# Think-MRIS

Think-MRIS is an implementation of reward-optimized knowledge injection for
medical referring image segmentation.

Dataset: https://github.com/lixiangcog/Think-MRIS/blob/master/dataset/DATA_SOURCES.md

The associated manuscript is currently under submission. The full dataset, code, and detailed metadata will be released after the review process.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Training

```bash
bash training_scripts/run_think_mris_7b.sh
```

The main configuration is `training_scripts/think_mris_7b.yaml`.

## Inference

```bash
python inference_scripts/infer_multi_object.py \
  --reasoning_model_path /path/to/model \
  --image_path /path/to/image.png \
  --text "segment the enhancing tumor core" \
  --output_path /path/to/output.png
```

## Evaluation

```bash
REASONING_MODEL_PATH=/path/to/model \
  bash evaluation_scripts/eval_calibrated_full.sh
```
