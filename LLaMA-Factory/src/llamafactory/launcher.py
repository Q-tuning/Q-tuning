# Copyright 2025 the LlamaFactory team.
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

import os
import warnings

from llamafactory.train.tuner import run_exp  # use absolute import

# # Silence common noisy warnings and reduce third-party logs
# warnings.filterwarnings("ignore", message="Trainer.tokenizer is now deprecated")
# warnings.filterwarnings("ignore", category=UserWarning)

# # transformers / datasets logging control
# os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def launch():
    run_exp()


if __name__ == "__main__":
    launch()
