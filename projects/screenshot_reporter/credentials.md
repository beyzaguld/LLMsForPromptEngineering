---
provider: openrouter
api_key_env: OPENROUTER_API_KEY
base_url: https://openrouter.ai/api/v1
target_models:
  - nvidia/nemotron-nano-12b-v2-vl:free
  - nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
optimizer_model: openai/gpt-oss-120b:free
max_iterations: 12
pass_threshold: 1.0
call_delay: 3.0
---
