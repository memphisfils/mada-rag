# Rapport d'évaluation

## Statut de ce rapport

Le jeu de test versionné contient **25 cas** dans
[`data/eval/questions.jsonl`](../data/eval/questions.jsonl). Les résultats
ci-dessous proviennent exclusivement de deux artefacts versionnés :

| Artefact | Modes | Horodatage exact (`generated_at`) |
|---|---|---|
| [`evaluation-dense-hybrid.json`](../artifacts/reports/evaluation-dense-hybrid.json) | `dense`, `hybrid` | `2026-07-28T21:20:38.010076+00:00` |
| [`evaluation-hybrid-rerank.json`](../artifacts/reports/evaluation-hybrid-rerank.json) | `hybrid-rerank` | `2026-07-29T08:06:01.239708+00:00` |

Chaque run porte sur 25 cas, dont 22 answerable pour les métriques de retrieval,
et ne contient aucune erreur enregistrée. Les trois modes partagent la même
révision, le même SHA-256 de snapshot et les mêmes hashes de corpus/index
indiqués ci-dessous.

Le seul snapshot autorisé est :

| Champ | Valeur |
|---|---|
| Page | Wikipedia anglaise `Madagascar` |
| Révision | `1365949107` |
| Horodatage de révision | `2026-07-25T10:41:18Z` |
| Horodatage de capture | `2026-07-28T15:29:12.618740Z` |
| HTML SHA-256 | `c54a3df9ca9650a99b717e0b235bb2845593b69a96d744becaea6c0eac4e3a4a` |
| Snapshot SHA-256 dans les runs | `c54a3df9ca9650a99b717e0b235bb2845593b69a96d744becaea6c0eac4e3a4a` |
| Corpus chunks SHA-256 | `99e27de0805b3a201d443da3f0e7b4324dd459f360a866e166d215e74f239a05` |
| Index dense SHA-256 | `5d94f75a4c00b34f497bc7c457f275139564daca702ddd6a91eb708523dfbe61` |
| Provenance | [`data/raw/manifest.json`](../data/raw/manifest.json) |

## Jeu d'évaluation

Les 25 cas couvrent les catégories exigées :

| Catégorie | Cas |
|---|---:|
| Fait simple | 4 |
| Chiffre précis | 7 |
| Lecture de tableau | 5 |
| Raisonnement multi-passages | 2 |
| Ambiguïté temporelle | 3 |
| Hors périmètre / piège | 3 |
| Couverture partielle | 1 |

Chaque ligne JSONL fixe la langue, la réponse attendue lorsqu'elle est
answerable, les chemins de sections, les IDs de chunks ou extraits de preuve,
ainsi que les pièges d'abstention. Les cas temporels doivent être interprétés
relativement à la révision ci-dessus.

## Protocole reproduit par les runs

1. Créer un environnement propre avec Python 3.12 et `uv sync --frozen
   --all-extras --group dev`.
2. Vérifier le manifeste et reconstruire l'index à partir de `data/raw/`, sans
   relancer l'ingestion : `uv run mada-rag index --overwrite`.
3. Geler les paramètres : modèle E5, modèle de reranking éventuel, `top_k`,
   pool BM25, `rrf_k`, seuil d'abstention, machine/OS et versions.
4. Exécuter exactement les mêmes 25 questions pour `dense`, `hybrid` et
   `hybrid-rerank`, en séparant les latences cold et warm. Le reranker est
   facultatif à l'exécution mais doit être explicitement marqué indisponible si
   son modèle ne peut pas être chargé.
5. Sauvegarder pour chaque mode les sorties JSON, la configuration, la date et
   les hashes du corpus/index dans `artifacts/reports/`.
6. Calculer les métriques ci-dessous et ne publier que les chiffres attachés à
   ces artefacts. Une revue manuelle d'entailment reste nécessaire avant toute
   conclusion sémantique sur les réponses.

## Définitions de métriques

| Mesure | Définition |
|---|---|
| Recall@k | part des cas answerable dont au moins un chunk de preuve attendu est dans les `k` premiers résultats ; préciser le `k` |
| MRR | moyenne de l'inverse du rang du premier chunk de preuve attendu |
| nDCG@k | pertinence graduée des chunks de preuve, avec gain/normalisation documentés |
| Exactitude de réponse | part des réponses answerable dont le contenu satisfait la réponse attendue et ses conditions temporelles ; non calculée par les artefacts actuels |
| Précision des citations | proxy : citations qui recoupent les chunks/extraits de référence attendus ; ce n'est pas une validation humaine d'entailment |
| Validité des citations | citations structurellement valides par rapport aux chunks retournés et à leurs extraits exacts |
| Exactitude d'abstention | part des cas non answerable ou insuffisants correctement refusés |
| Faux positifs pièges | part des cas `out-of-scope` recevant une affirmation au lieu d'une abstention |
| Latence p50/p95 | percentiles mesurés séparément pour retrieval, reranking éventuel et réponse, avec cold/warm documenté |

La réponse extractive peut comporter plusieurs claims. Les métriques de contenu
ci-dessous sont des **proxies** calculés sur les chunks, extraits et citations
attendus du jeu versionné ; elles ne constituent ni une vérité sémantique LLM,
ni une revue humaine d'entailment. Les citations doivent toujours être vérifiées
contre les offsets/extraits exacts, pas seulement contre la section.

## Résultats réels

`k=5`, avec 22 cas answerable pour le retrieval et 25 cas pour l'abstention et
les latences. Les latences sont en millisecondes ; elles sont les p50/p95
enregistrés dans les artefacts, sans extrapolation à une autre machine.

| Mode | Artefact | Recall@5 | MRR | nDCG@5 | Précision citations (proxy) | Validité citations | Abstention | p50/p95 retrieval | p50/p95 answer | Erreurs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | `evaluation-dense-hybrid.json` | 0.772727 | 0.810606 | 0.739043 | 0.516129 (16/31) | 1.000000 (31/31) | 0.560000 (14/25) | 1 140.906 / 2 742.389 | 714.873 / 1 568.903 | 0 |
| Hybride RRF | `evaluation-dense-hybrid.json` | 0.810606 | 0.844697 | 0.762778 | 0.500000 (17/34) | 1.000000 (34/34) | 0.600000 (15/25) | 824.984 / 1 811.131 | 733.799 / 1 236.832 | 0 |
| Hybride RRF + reranker | `evaluation-hybrid-rerank.json` | 0.840909 | 0.796970 | 0.764046 | 0.500000 (17/34) | 1.000000 (34/34) | 0.600000 (15/25) | 27 255.403 / 52 129.900 | 24 700.280 / 45 007.858 | 0 |

Les artefacts ne calculent pas séparément l'exactitude sémantique des réponses
ni le taux de faux positifs limité aux seuls pièges ; ces deux valeurs ne sont
donc volontairement pas déduites du tableau.

## Lecture et limites des résultats

L'hybride RRF améliore les trois proxies de retrieval par rapport au dense sur
ce run (`Recall@5` 0.810606 contre 0.772727 ; `MRR` 0.844697 contre 0.810606 ;
`nDCG@5` 0.762778 contre 0.739043), avec des p50/p95 de retrieval inférieurs
dans cet échantillon. Le reranker obtient le Recall@5 et le nDCG@5 les plus
élevés (0.840909 et 0.764046), mais son MRR est inférieur à celui de l'hybride
RRF et sa latence est très supérieure. Ces comparaisons sont descriptives des
artefacts enregistrés, non une généralisation à tout matériel ou toute charge.

La validité structurale des citations est de 1.0 dans les trois runs, tandis
que la précision de citations proxy reste entre 0.5 et 0.516129. Cela démontre
que le graphe de preuves est cohérent selon le validateur, pas que chaque claim
est sémantiquement vrai ou complet. Une revue humaine/entailment des cas
échoués, une mesure distincte des pièges et une séparation calibration/final
restent nécessaires avant toute affirmation de qualité finale.

## Réflexion : deux semaines supplémentaires

Avec deux semaines supplémentaires, je livrerais d'abord le harnais manquant,
des runs versionnés et une analyse d'erreurs par langue, catégorie et type de
tableau. La priorité serait de distinguer une amélioration de retrieval d'une
simple amélioration de style de réponse : les mesures de couverture des preuves,
des citations et des abstentions seraient donc publiées avant toute conclusion
sur le reranker. Je mettrais en place un jeu de calibration séparé, puis je
gèlerais les seuils et les prompts avant le jeu final.

Je consacrerais ensuite le temps restant à une génération sous contraintes plus
riche, mais toujours vérifiable : schéma de claims atomiques, validation
d'entailment/revue humaine ciblée et tests adversariaux sur les dates, unités,
tables et reformulations bilingues. Enfin, j'automatiserais un clone propre,
le cache/versionnement des modèles et la production d'artefacts signés afin de
conserver une comparaison fiable quand la page source ou les dépendances
évoluent.
