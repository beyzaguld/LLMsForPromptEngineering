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

# Locale-Aware Table Query Analyzer

Analyze tabular data containing numbers in mixed locale formats (US: 1,000.50 and
Turkish: 1.000,50). Answer natural language queries about the table data correctly
by first detecting and normalizing the locale of each number.

Output must be a JSON object with keys: answer, reasoning, locale_detected.
