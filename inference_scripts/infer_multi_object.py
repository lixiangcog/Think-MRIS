import argparse
import os
from pathlib import Path
from typing import Optional


_CACHE_ROOT = Path(os.environ.get("THINK_MRIS_CACHE_ROOT", "/workspace/.cache"))
os.environ["HF_HOME"] = os.environ.get("THINK_MRIS_HF_HOME", str(_CACHE_ROOT / "huggingface"))
os.environ["HF_HUB_CACHE"] = os.environ.get(
    "THINK_MRIS_HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub")
)
os.environ["TRANSFORMERS_CACHE"] = os.environ.get(
    "THINK_MRIS_TRANSFORMERS_CACHE", os.environ["HF_HUB_CACHE"]
)

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as PILImage
from qwen_vl_utils import process_vision_info
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from think_mris.modeling import build_vlki_prior
from think_mris.prompting import ThinkMRISPromptTemplate
from think_mris.utils import extract_think_answer, scale_answer_objects


def parse_args():
    parser = argparse.ArgumentParser(description="Think-MRIS reasoning + VLKI + SAM2 inference")
    parser.add_argument("--reasoning_model_path", default="pretrained_models/Think-MRIS-7B")
    parser.add_argument(
        "--processor_path",
        default=None,
        help="Tokenizer/processor artifact; defaults to --reasoning_model_path.",
    )
    parser.add_argument("--segmentation_model_path", default="facebook/sam2-hiera-large")
    parser.add_argument("--text", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_path", default="inference_outputs/think_mris_output.png")
    parser.add_argument("--max_side_length", type=int, default=640)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--disable_vlki", action="store_true")
    parser.add_argument(
        "--response_override",
        default=None,
        help=(
            "Use a canonical <think>/<answer> response instead of model.generate. "
            "This is intended only for end-to-end integration tests; hidden states, "
            "VLKI, and SAM2 still run normally."
        ),
    )
    return parser.parse_args()


def resize_max_side(image: PILImage.Image, max_side_length: int) -> PILImage.Image:
    if max(image.size) <= max_side_length:
        return image
    scale = max_side_length / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, PILImage.Resampling.BILINEAR)


def save_visualization(image: PILImage.Image, mask: np.ndarray, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[1].imshow(image)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask] = (1.0, 0.0, 0.0, 0.4)
    axes[1].imshow(overlay)
    axes[1].set_title("Think-MRIS mask")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map={"": device},
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.processor_path or args.reasoning_model_path,
        padding_side="left",
    )

    original_image = PILImage.open(args.image_path).convert("RGB")
    model_image = resize_max_side(original_image, args.max_side_length)
    prompt_template = ThinkMRISPromptTemplate()
    messages = [
        {"role": "system", "content": prompt_template.system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": model_image},
                {
                    "type": "text",
                    "text": prompt_template.format_user_prompt(args.text, include_image_token=False),
                },
            ],
        },
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    if args.response_override is None:
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
    else:
        response_override_ids = processor.tokenizer(
            args.response_override,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(device)
        generated_ids = torch.cat((inputs.input_ids, response_override_ids), dim=1)
    response_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    output_text = processor.batch_decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print(output_text)
    think_text, answer_objects = extract_think_answer(output_text)
    scaled_objects = scale_answer_objects(answer_objects, model_image.size, original_image.size)
    print("Thinking process:", think_text)
    print("Grounding:", scaled_objects)

    dense_prior: Optional[torch.Tensor] = None
    if not args.disable_vlki:
        dense_prior = build_vlki_prior(model, processor, inputs, generated_ids)

    predictor = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path, device=str(device))
    mask_all = np.zeros((original_image.height, original_image.width), dtype=bool)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(np.asarray(original_image))
        for item in scaled_objects:
            prediction_kwargs = {
                "point_coords": np.asarray(item["points_2d"], dtype=np.float32),
                "point_labels": np.ones(2, dtype=np.int32),
                "box": np.asarray(item["bbox_2d"], dtype=np.float32),
                "multimask_output": True,
            }
            if dense_prior is not None:
                prediction_kwargs["mask_input"] = dense_prior[0].float().cpu().numpy()
            masks, scores, _ = predictor.predict(**prediction_kwargs)
            mask_all |= masks[int(np.argmax(scores))].astype(bool)

    save_visualization(original_image, mask_all, args.output_path)
    print(f"Saved inference visualization to {Path(args.output_path).resolve()}")


if __name__ == "__main__":
    main()
