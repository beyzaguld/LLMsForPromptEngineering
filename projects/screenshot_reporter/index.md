---
prompt: prompt.md
credentials: credentials.md
tests: tests/
output: results/optimized_prompt.md
input_type: image
split:
  train: [tc_001, tc_002, tc_003]
  validation: [val_001]
---

# Screenshot Bug Reporter

Given a screenshot of a mobile or web application showing a bug or error,
extract structured information about the issue.

Input: PNG screenshot of a crashed/frozen/erroring app UI.
Output: JSON with fields: severity, affected_component, affected_platform,
        error_type, reproducibility.
