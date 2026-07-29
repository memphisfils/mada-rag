# Holdout evaluation set

`holdout.jsonl` is an immutable, separate evaluation set for the Madagascar
snapshot at revision `1365949107` (source revision timestamp:
`2026-07-25T10:41:18Z`). It uses only exact chunks from the committed local
corpus for that revision.

These cases were not used to tune retrieval, reranking, chunking, sufficiency
thresholds, prompts, or abstention behaviour. Keep them separate from
`questions.jsonl`: evaluate them only after implementation choices are frozen,
and report their results independently.
