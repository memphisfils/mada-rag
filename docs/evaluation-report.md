# Rapport d'évaluation

## Référence de release et traçabilité

La référence de release est le run **calibré après l'ajout du gate d'ancrage**.
Elle s'appuie exclusivement sur les deux artefacts suivants :

| Artefact final | Modes | Horodatage exact (`generated_at`) |
|---|---|---|
| [`evaluation-calibrated-dense-hybrid.json`](../artifacts/reports/evaluation-calibrated-dense-hybrid.json) | `dense`, `hybrid` | `2026-07-29T09:36:43.487927+00:00` |
| [`evaluation-calibrated-hybrid-rerank.json`](../artifacts/reports/evaluation-calibrated-hybrid-rerank.json) | `hybrid-rerank` | `2026-07-29T09:40:18.273216+00:00` |

Les deux rapports de baseline pré-calibration sont conservés sans modification
pour la traçabilité :
[`evaluation-dense-hybrid.json`](../artifacts/reports/evaluation-dense-hybrid.json)
et
[`evaluation-hybrid-rerank.json`](../artifacts/reports/evaluation-hybrid-rerank.json).
Ils décrivent un système antérieur au gate d'ancrage ; ils ne sont donc **pas
comparables comme si le système était inchangé** et ne constituent pas les
résultats de release.

Le gate d'ancrage a été ajusté et testé contre ce même jeu versionné de 25 cas,
avec ses chunks attendus, avant la relance calibrée. Les résultats finaux
démontrent donc une **régression contrôlée et la conformité de release** sur ce
jeu ; ils ne constituent pas une estimation non biaisée de la généralisation.
Une évaluation future doit séparer un jeu de calibration et un holdout final
inédit avant toute conclusion de performance généralisable.

Chaque run final porte sur 25 cas, dont 22 answerable pour les métriques de
retrieval, et ne contient aucune erreur enregistrée. Les trois pièges hors
périmètre sont `abstained` avec zéro citation dans chaque mode final.

| Champ | Valeur |
|---|---|
| Page | Wikipedia anglaise `Madagascar` |
| Révision | `1365949107` |
| Horodatage de révision | `2026-07-25T10:41:18Z` |
| Horodatage de capture | `2026-07-28T15:29:12.618740Z` |
| HTML / snapshot SHA-256 | `c54a3df9ca9650a99b717e0b235bb2845593b69a96d744becaea6c0eac4e3a4a` |
| Corpus chunks SHA-256 | `99e27de0805b3a201d443da3f0e7b4324dd459f360a866e166d215e74f239a05` |
| Index dense SHA-256 | `5d94f75a4c00b34f497bc7c457f275139564daca702ddd6a91eb708523dfbe61` |
| Provenance | [`data/raw/manifest.json`](../data/raw/manifest.json) |

## Jeu d'évaluation et protocole

Le jeu versionné [`data/eval/questions.jsonl`](../data/eval/questions.jsonl)
contient 25 cas : 4 faits simples, 7 chiffres précis, 5 lectures de tableau,
2 multi-passages, 3 ambiguïtés temporelles, 3 pièges hors périmètre et 1 cas de
couverture partielle. Chaque cas fixe langue, révision, réponse attendue quand
elle est answerable et preuves acceptables.

Les runs finaux utilisent `k=5`, le même snapshot, corpus et index pour les
trois modes. Le gate d'ancrage a été calibré sur ce même jeu afin de ne pas
accepter une question sur un simple mot générique partagé ; il ne s'agit donc
pas d'une séparation calibration/final indépendante. Pour reproduire :

```bash
uv run mada-rag evaluate --mode dense --mode hybrid --top-k 5 \
  --output artifacts/reports/evaluation-calibrated-dense-hybrid.json

MADA_RAG_RERANKER_ENABLED=true uv run mada-rag evaluate \
  --mode hybrid-rerank --top-k 5 \
  --output artifacts/reports/evaluation-calibrated-hybrid-rerank.json
```

## Définitions et limites des métriques

| Mesure | Définition |
|---|---|
| Recall@5 | part des 22 cas answerable dont une preuve attendue est récupérée dans les cinq premiers chunks |
| MRR / nDCG@5 | rang du premier chunk de preuve / qualité du classement par rapport aux chunks attendus |
| Précision citations | proxy basé sur l'intersection entre citations et chunks/extraits attendus |
| Validité citations | validité structurelle : chunk retourné, même révision, offsets, extrait, section et provenance exacts |
| Exactitude d'abstention | statut `answered`/`abstained` correct sur les 25 cas versionnés |
| Latence p50/p95 | millisecondes mesurées dans les artefacts, sans extrapolation à une autre machine |

Les métriques de contenu sont des **proxies** fondés sur chunks, extraits et
citations de référence. Elles ne constituent ni une vérité sémantique LLM, ni
une revue humaine d'entailment ; une citation structurellement valide ne prouve
pas à elle seule que toute réponse est complète ou sémantiquement correcte.

## Résultats finaux calibrés

| Mode | Recall@5 | MRR | nDCG@5 | Précision citations (proxy) | Validité citations | Abstention | p50/p95 retrieval (ms) | p50/p95 answer (ms) | Erreurs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0.772727 | 0.810606 | 0.739043 | 0.564516 | 1.000000 | 1.000000 (25/25) | 423.760 / 2 096.993 | 555.299 / 1 095.291 | 0 |
| Hybride RRF | 0.871212 | 0.954545 | 0.859030 | 0.532258 | 1.000000 | 1.000000 (25/25) | 429.164 / 1 165.835 | 588.917 / 1 442.539 | 0 |
| Hybride RRF + reranker | 0.863636 | 0.799242 | 0.776049 | 0.483871 | 1.000000 | 1.000000 (25/25) | 21 756.536 / 26 181.638 | 21 801.078 / 26 566.471 | 0 |

Le mode **hybride RRF est le défaut recommandé** : il atteint les meilleurs
Recall@5, MRR et nDCG@5 du run final tout en gardant des latences p95 de l'ordre
de 1 à 1,5 seconde. Le reranker reste un opt-in expérimental : son p95 de
retrieval est de 26,2 secondes et son p95 de réponse de 26,6 secondes, pour des
proxies de classement inférieurs à l'hybride RRF sur ce run.

## Historique pré-calibration

Les valeurs ci-dessous sont retenues comme audit historique des artefacts de
baseline, pas comme comparaison causale avec les résultats finaux calibrés.

| Mode | Artefact baseline | Recall@5 | MRR | nDCG@5 | Abstention | p50/p95 retrieval (ms) |
|---|---|---:|---:|---:|---:|---:|
| Dense | `evaluation-dense-hybrid.json` | 0.772727 | 0.810606 | 0.739043 | 0.560000 (14/25) | 1 140.906 / 2 742.389 |
| Hybride RRF | `evaluation-dense-hybrid.json` | 0.810606 | 0.844697 | 0.762778 | 0.600000 (15/25) | 824.984 / 1 811.131 |
| Hybride RRF + reranker | `evaluation-hybrid-rerank.json` | 0.840909 | 0.796970 | 0.764046 | 0.600000 (15/25) | 27 255.403 / 52 129.900 |

## Réflexion : deux semaines supplémentaires

Avec deux semaines supplémentaires, je commencerais par une revue humaine
d'entailment des réponses, des citations et des pièges, puis par une analyse
d'erreurs par langue, catégorie et type de tableau. Elle permettrait de
différencier une amélioration de retrieval d'une simple correspondance de
chunks attendus, et de confirmer que le gate calibré reste prudent sur des
formulations inédites.

Je consacrerais ensuite le temps restant à une génération sous contraintes plus
riche mais vérifiable, au cache/versionnement des modèles, aux mesures cold/warm
sur une machine cible et à une politique de délai pour le reranker. Enfin,
j'automatiserais le clone propre, l'entailment ciblé et la conservation des
artefacts de release afin de préserver la reproductibilité lors des évolutions
de la page ou des dépendances.
