# NeurIPS 2025 corpus analysis

- **Total submissions (records)**: 5542
- **Labeled accept/reject (from decision/venue)**: 5542
  - accepted labeled: 5287
  - rejected labeled: 255
- **Papers with local PDF→MD available (pdf2md/markdown_output)**: 52

## Outputs

- `paper_features.csv`: one row per submission with metadata + optional MD-quality features
- `primary_area_stats.csv`: counts + labeled acceptance rate per primary area
- `keyword_stats_top.csv`: top keywords with labeled acceptance rate (when available)
- `rating_bins.csv`: acceptance rate vs average review rating bins (when available)

## Notes / limitations

- Some decision/review summaries may be missing due to OpenReview rate limiting (HTTP 429).
- If you need full coverage of decision/review summaries, rerun the export with a larger `--sleep` (e.g. 1.0–2.0) or rerun later.
- Review ratings/confidence may be missing if official reviews are not public in the forum head notes; in that case, prediction uses only metadata (keywords/area/etc.).
- The MD-quality features are heuristics; they’re useful for trends, but they’re not a substitute for a human quality assessment.

