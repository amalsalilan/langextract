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
"""Quick-start example for using LM Studio with langextract."""

import langextract as lx


def run_extraction(model_id: str = "lmstudio:Phi-3-mini") -> lx.data.AnnotatedDocument:
  """Run a simple extraction example using an LM Studio endpoint."""
  input_text = "Bob enjoys hiking in the mountains."
  prompt = "Extract the person's name and their hobby."

  examples = [
      lx.data.ExampleData(
          text="Alice likes painting landscapes.",
          extractions=[
              lx.data.Extraction(
                  extraction_class="person_hobby",
                  extraction_text="Alice likes painting landscapes.",
                  attributes={"name": "Alice", "hobby": "painting"},
              )
          ],
      )
  ]

  model_config = lx.factory.ModelConfig(
      model_id=model_id,
      provider_kwargs={"base_url": "http://localhost:1234/v1"},
  )

  return lx.extract(
      text_or_documents=input_text,
      prompt_description=prompt,
      examples=examples,
      config=model_config,
      use_schema_constraints=True,
  )


def main() -> None:
  result = run_extraction()
  for extraction in result.extractions:
    print(extraction)


if __name__ == "__main__":
  main()
