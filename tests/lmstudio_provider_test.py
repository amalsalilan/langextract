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
"""Tests for the LM Studio provider."""

import langextract as lx
from langextract.providers.lmstudio import LMStudioLanguageModel


def test_lmstudio_provider_resolution():
  config = lx.factory.ModelConfig(model_id="lmstudio:dummy")
  model = lx.factory.create_model(config)
  assert isinstance(model, LMStudioLanguageModel)
