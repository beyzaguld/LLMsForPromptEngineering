---
prompt: prompt.md
credentials: credentials.md
tests: tests/
output: results/optimized_prompt.md
input_type: text
split:
  train: [tc_001, tc_002, tc_003]
  validation: [val_001]
---

# Tool Call Router

Given a user request and a list of available tools, output the correct
tool_call as a JSON object with fields: tool_name, arguments.
