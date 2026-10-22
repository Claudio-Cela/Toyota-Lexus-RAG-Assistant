# Evaluation Results

The CSV files in this folder contain the complete pre-RAG, post-RAG, retrieval-method, and generation-engine comparison outputs.

The project-specific overall score is:

```text
overall_score =
    0.55 * keyword_coverage
  + 0.30 * format_score
  + 0.15 * (1 - unsupported_flag)
```

- `keyword_coverage` measures the fraction of expected terms found in the response.
- `format_score` measures adherence to the five required response sections.
- `unsupported_flag` identifies explicitly prohibited statements defined for each test case.

These scores support controlled comparison within this project and should not be interpreted as universal automotive accuracy percentages.