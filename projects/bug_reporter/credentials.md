---
provider: openrouter
api_key_env: OPENROUTER_API_KEY
base_url: https://openrouter.ai/api/v1
target_models:
  - nvidia/nemotron-3-nano-30b-a3b:free
  - openai/gpt-oss-20b:free
optimizer_model: openai/gpt-oss-120b:free
max_iterations: 12
pass_threshold: 1.0
call_delay: 3.0
---
