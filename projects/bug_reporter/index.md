---
prompt: prompt.md
credentials: credentials.md
tests: tests/
output: results/optimized_prompt.md
input_type: text
split:
  train: [tc_001, tc_002, tc_003, tc_004]
  validation: [val_001, val_002]
---

# Bug Report JSON Extractor

Extract structured information from free-text software bug reports.
The output must be a JSON object with fields: severity, affected_component,
affected_platform, condition, reproducibility.
