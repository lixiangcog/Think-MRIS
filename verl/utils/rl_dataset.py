# Copyright 2024 Anonymous contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from think_mris.prompting import ThinkMRISPromptTemplate
from verl.models.transformers.qwen2_5_vl import get_rope_index


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        if key not in ["pixel_values", "image_grid_thw"]:
            tensors[key] = torch.stack(value, dim=0)

    return {**tensors, **non_tensors}


def image_processor_size_kwargs(min_pixels: Optional[int], max_pixels: Optional[int]) -> Dict[str, int]:
    """Keep dataset-side visual tokenization aligned with the rollout engine."""
    kwargs = {}
    if min_pixels is not None:
        kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        kwargs["max_pixels"] = max_pixels
    return kwargs


def process_image(
    image: ImageObject,
    max_pixels: int,
    min_pixels: int,
    max_side_length: Optional[int] = None,
) -> ImageObject:
    if max_side_length and max(image.width, image.height) > max_side_length:
        resize_factor = max_side_length / max(image.width, image.height)
        width, height = max(1, round(image.width * resize_factor)), max(1, round(image.height * resize_factor))
        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def scale_solution_geometry(solution: str, x_scale: float, y_scale: float, width: int, height: int) -> str:
    """Apply the same resize to sparse annotations and their coordinate frame."""
    try:
        payload = json.loads(solution)
    except (TypeError, ValueError, json.JSONDecodeError):
        return solution
    metadata = payload if isinstance(payload, dict) else {}
    objects = metadata.get("solution", []) if metadata else payload
    if not isinstance(objects, list):
        return solution
    for item in objects:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_2d")
        if isinstance(bbox, list) and len(bbox) == 4:
            item["bbox_2d"] = [
                round(bbox[0] * x_scale),
                round(bbox[1] * y_scale),
                round(bbox[2] * x_scale),
                round(bbox[3] * y_scale),
            ]
        for key in ("points_2d", "points"):
            points = item.get(key)
            if isinstance(points, list):
                item[key] = [
                    [round(point[0] * x_scale), round(point[1] * y_scale)]
                    for point in points
                    if isinstance(point, list) and len(point) == 2
                ]
        point = item.get("point_2d")
        if isinstance(point, list) and len(point) == 2:
            item["point_2d"] = [round(point[0] * x_scale), round(point[1] * y_scale)]
    if metadata:
        metadata["img_width"] = width
        metadata["img_height"] = height
    return json.dumps(payload, ensure_ascii=False)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key="prompt",
        max_prompt_length=1024,
        truncation="error",
        system_prompt=None,
        max_pixels=None,
        min_pixels=None,
        max_side_length=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_side_length = max_side_length

        #self.dataset = load_dataset(data_path)['train']
        self.dataset = load_from_disk(data_path)['train'] # you can load from disk if you have already downloaded the dataset
        
        ################ Old Version ################
        # self.user_prompt = "<image>" \
        #     "Please find '{Question}' with bbox and points." \
        #     "Compare the difference between objects and find the most closely matched one." \
        #     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
        #     "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
        #     "i.e., <think> thinking process here </think>" \
        #     "<answer>{Answer}</answer>"
        ################ Old Version ################
        
        prompt_template = ThinkMRISPromptTemplate()
        system_prompt = self.system_prompt or prompt_template.system_prompt
        self.prompt_template = ThinkMRISPromptTemplate(system_prompt=system_prompt)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataset[index]
        
        ################ Old Version ################
        # messages = [
        #     {"role": "system", "content": self.system_prompt},
        #     {"role": "user", "content": self.user_prompt.format(Question=row_dict["problem"].lower().strip("."),
        #                                                         Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}")},
        # ]
        ################ Old Version ################
        
        messages = [
            {"role": "system", "content": self.prompt_template.system_prompt},
            {"role": "user", "content": self.prompt_template.format_user_prompt(row_dict["problem"])},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if "image" in row_dict:
            row_dict["images"] = [row_dict["image"]]
        if "images" in row_dict:  # expand image token
            raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            processed_images = []
            for image_index, image in enumerate(row_dict["images"]):
                original_width, original_height = image.size
                processed = process_image(
                    image,
                    self.max_pixels,
                    self.min_pixels,
                    max_side_length=self.max_side_length,
                )
                processed_images.append(processed)
                if image_index == 0 and "solution" in row_dict:
                    row_dict["solution"] = scale_solution_geometry(
                        row_dict["solution"],
                        processed.width / original_width,
                        processed.height / original_height,
                        processed.width,
                        processed.height,
                    )
                    row_dict["img_width"] = processed.width
                    row_dict["img_height"] = processed.height
            row_dict["images"] = processed_images
            image_inputs = self.processor.image_processor(
                row_dict["images"],
                return_tensors="pt",
                **image_processor_size_kwargs(self.min_pixels, self.max_pixels),
            )
            image_grid_thw = image_inputs["image_grid_thw"]
            row_dict.update(image_inputs)

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while "<image>" in prompt:
                    prompt = prompt.replace(
                        "<image>",
                        "<|vision_start|>"
                        + "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length)
                        + "<|vision_end|>",
                        1,
                    )
                    index += 1

                prompt = prompt.replace("<|placeholder|>", self.processor.image_token)
        else:
            raw_prompt = prompt

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if "images" in row_dict:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )  # (3, seq_len)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seqlen,)

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        return row_dict

