# Model evaluation report (temporal holdout)

- Users evaluated: 594
- K = 10, relevant = held-out rating >= 3.5
- Held-out relevant items never seen in training (unreachable for ALS/content/NCF): 1252 of 12117

| Strategy | Precision@10 | Recall@10 | F1@10 | NDCG@10 | Hit@10 | Coverage | Pop. percentile | Accuracy | Fallback users |
|---|---|---|---|---|---|---|---|---|---|
| popularity | 0.0202 | 0.0131 | 0.0122 | 0.0226 | 0.1465 | 0.0127 | 0.0031 | 0.9969 | 2 |
| als | 0.0865 | 0.0886 | 0.0673 | 0.1072 | 0.4832 | 0.0779 | 0.1564 | 0.997 | 2 |
| content | 0.012 | 0.0137 | 0.0101 | 0.0134 | 0.1128 | 0.1353 | 0.4197 | 0.9969 | 2 |
| ncf | 0.0643 | 0.0425 | 0.0395 | 0.0799 | 0.3384 | 0.0098 | 0.0254 | 0.997 | 2 |

## NDCG@10 by user activity (training ratings)

| Strategy | low | medium | high |
|---|---|---|---|
| popularity | 0.015 | 0.0132 | 0.0393 |
| als | 0.0999 | 0.095 | 0.1262 |
| content | 0.0138 | 0.0116 | 0.0148 |
| ncf | 0.0528 | 0.0571 | 0.1286 |
