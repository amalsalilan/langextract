# Copyright 2025 Google LLC.
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
"""LM Studio provider for LangExtract."""

from __future__ import annotations

import dataclasses
from typing import Any, Iterator, Sequence

import requests

from langextract import data
from langextract import exceptions
from langextract import inference
from langextract import schema
from langextract.providers import registry

_DEFAULT_BASE_URL = "http://localhost:1234/v1"
_DEFAULT_TIMEOUT = 60


@registry.register(r"^lmstudio:")
@dataclasses.dataclass(init=False)
class LMStudioLanguageModel(inference.BaseLanguageModel):
  """Language model inference using an LM Studio server."""

  model_id: str = "lmstudio:unknown"
  base_url: str = _DEFAULT_BASE_URL
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float | None = None
  timeout: int = _DEFAULT_TIMEOUT
  _requests: Any = dataclasses.field(default=requests, repr=False, compare=False)
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  @property
  def requires_fence_output(self) -> bool:
    if self.format_type == data.FormatType.JSON:
      return False
    return super().requires_fence_output

  def __init__(
      self,
      model_id: str,
      base_url: str = _DEFAULT_BASE_URL,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float | None = None,
      timeout: int = _DEFAULT_TIMEOUT,
      **kwargs,
  ) -> None:
    """Initialize the LM Studio language model.

    Args:
      model_id: The LM Studio model ID prefixed with ``lmstudio:``.
      base_url: Base URL of the LM Studio server.
      format_type: Desired output format (JSON or YAML).
      temperature: Sampling temperature.
      timeout: Request timeout in seconds.
      **kwargs: Additional unused parameters.
    """
    self.model_id = model_id
    self.base_url = base_url
    self.format_type = format_type
    self.temperature = temperature
    self.timeout = timeout
    self._requests = requests

    self._actual_model = model_id.split(":", 1)[1] if ":" in model_id else model_id

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )
    self._extra_kwargs = kwargs or {}

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[inference.ScoredOutput]]:
    timeout = kwargs.get("timeout", self.timeout)
    for prompt in batch_prompts:
      payload: dict[str, Any] = {
          "model": self._actual_model,
          "messages": [{"role": "user", "content": prompt}],
      }
      if self.temperature is not None:
        payload["temperature"] = self.temperature
      if self.format_type == data.FormatType.JSON:
        payload["response_format"] = {"type": "json_object"}

      try:
        response = self._requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
      except self._requests.exceptions.RequestException as e:
        raise exceptions.InferenceRuntimeError(
            f"LM Studio request failed: {e}", provider="LMStudio"
        ) from e

      yield [inference.ScoredOutput(score=1.0, output=content)]
