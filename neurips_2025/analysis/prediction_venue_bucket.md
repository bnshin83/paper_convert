# Baseline prediction: venue bucket

This predicts `oral/spotlight/poster/submitted` using a simple Multinomial Naive Bayes model over:
- keywords (full keyword + split words)
- primary_area
- abstract length bin
- presence of a code-host link (github/gitlab/bitbucket) in title/abstract/decision_comment

- Train size: 4433
- Test size: 1109
- Accuracy: 0.775

## Confusion matrix (rows=true, cols=pred)

| true \ pred | oral | poster | spotlight | submitted |
|---|---|---|---|---|
| oral | 0 | 14 | 1 | 0 |
| poster | 0 | 856 | 46 | 3 |
| spotlight | 0 | 131 | 4 | 2 |
| submitted | 0 | 49 | 3 | 0 |

