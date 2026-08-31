# MRIS-Bench paper-aligned v4

This configuration contains 25,809 rows.

## Splits

| Split | Rows | Fraction |
|---|---:|---:|
| train | 18,066 | 70% |
| validation | 2,581 | 10% |
| test | 5,162 | 20% |

The test split contains all 14 target categories, all five modalities, and all
eight source datasets present in this public snapshot.

The only recovered `right_atrium` row is in the test split. Adding it to train or
validation would require duplication and would introduce leakage.

The three difficulty factors use integer ratings from 1 to 10. The `difficulty`
field is their arithmetic mean, rounded to two decimal places.

Load this configuration with:

```python
from datasets import load_dataset

dataset = load_dataset("lixiangcog/MRIS-Bench", "paper_aligned_v4")
```
