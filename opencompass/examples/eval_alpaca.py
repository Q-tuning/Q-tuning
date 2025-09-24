from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.collections.base_medium_llama import (
        mmlu_datasets, bbh_datasets, drop_datasets, humaneval_datasets, gsm8k_datasets, math_datasets)
    from opencompass.configs.models.hf_llama.alpaca import models

datasets = [*mmlu_datasets, *bbh_datasets, *drop_datasets, *humaneval_datasets, *gsm8k_datasets, *math_datasets]