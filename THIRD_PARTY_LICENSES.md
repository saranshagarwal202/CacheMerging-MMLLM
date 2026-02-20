# Third-Party Licenses

## MileBench Evaluation Framework

The `evaluation/` directory contains code adapted from the
[MileBench](https://github.com/MileBench/MileBench) repository,
used under the **Apache License 2.0**.

### Files used without modification (original MileBench code)

- `evaluation/generate.py`
- `evaluation/evaluate.py`
- `evaluation/utils.py`
- `evaluation/score.py`
- `evaluation/workers/baseworker.py`
- `evaluation/configs/accelerate_configs.yaml`

### Files modified from MileBench originals

- `evaluation/workers/model_workers.py` — added `LLaVA_oneVision_cam` and `LLaVA_oneVision_Sink` worker classes
- `evaluation/configs/model_configs.yaml` — added `LLaVA_oneVision_cam` and `LLaVA_oneVision_Sink` model configs
- `evaluation/generate_with_summary.py` — single-GPU variant adapted from `generate.py`
- `evaluation/aggregate_scores.py` — extended with MILEBENCH category score aggregation

### Apache License 2.0

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## MileBench Dataset

The dataset used for evaluation (`datasets/MileBench/`) is subject to the
MileBench DATA_LICENSE. Please refer to the original repository for dataset
licensing terms: https://github.com/MileBench/MileBench
