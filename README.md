# 🧠 Think-MRIS: Rethinking Medical Referring Image Segmentation via Reward-Optimized Knowledge Injection

<p align="center">
  <img width="512" height="256" alt="logo" src="https://github.com/user-attachments/assets/a1c65268-cbff-415c-9194-66b0ae9aaf71" />
</p>

<p align="center">
  Reward-optimized knowledge injection for medical referring image segmentation.
</p>

Dataset: https://huggingface.co/datasets/lixiang007666/MRIS-Bench


> 📌 **Note**  
> To maintain compliance with the double-blind review policy, certain components of the code (e.g., data paths, model checkpoints, and scripts) have been intentionally removed or obfuscated.  
> These omissions do not affect the understanding of the method. A fully runnable version will be released after the review process.

---

## 🚀 Training

Use the Think-MRIS configuration:

```bash
bash training_scripts/run_think_mris_7b.sh
````

Core config:

```bash
training_scripts/think_mris_7b.yaml
```

The reward entry is:

```yaml
worker:
  reward:
    compute_score: think_mris
```

---

## 🔍 Inference

The multi-object inference script now defaults to a Think-MRIS model path and uses a medical referring prompt style:

```bash
python inference_scripts/infer_multi_object.py \
  --reasoning_model_path pretrained_models/Think-MRIS-7B \
  --image_path your_image.png \
  --text "segment the enhancing tumor core"
```

---

## 📊 Evaluation

Use:

```bash
bash evaluation_scripts/eval_think_mris.sh
```
