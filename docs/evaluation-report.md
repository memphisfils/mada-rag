# Rapport d'évaluation

## Statut de ce rapport

Ce document sépare strictement le protocole des résultats. Le jeu de test
versionné contient **25 cas** dans
[`data/eval/questions.jsonl`](../data/eval/questions.jsonl), mais le harnais
d'évaluation et la commande CLI `evaluate` ne sont pas encore livrés. Par
conséquent, aucune métrique, latence, comparaison dense/hybride/rerank ou
conclusion de performance n'est renseignée ici. Les champs `non mesuré` ne
sont pas des zéros et ne doivent pas être agrégés.

Le seul snapshot autorisé est :

| Champ | Valeur |
|---|---|
| Page | Wikipedia anglaise `Madagascar` |
| Révision | `1365949107` |
| Horodatage de révision | `2026-07-25T10:41:18Z` |
| Horodatage de capture | `2026-07-28T15:29:12.618740Z` |
| HTML SHA-256 | `c54a3df9ca9650a99b717e0b235bb2845593b69a96d744becaea6c0eac4e3a4a` |
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

## Protocole à exécuter avant de remplir les résultats

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
5. Sauvegarder pour chaque mode les sorties JSON, la configuration, la date,
   le hash du corpus/index et le journal de commande dans `artifacts/reports/`.
6. Calculer les métriques ci-dessous, effectuer la revue manuelle des réponses
   erronées et ne publier que les chiffres attachés à ces artefacts.

Une future commande `mada-rag evaluate` devra automatiser ce protocole; son
absence actuelle est un travail restant, pas une mesure négative ou positive.

## Définitions de métriques

| Mesure | Définition |
|---|---|
| Recall@k | part des cas answerable dont au moins un chunk de preuve attendu est dans les `k` premiers résultats ; préciser le `k` |
| MRR | moyenne de l'inverse du rang du premier chunk de preuve attendu |
| nDCG@k | pertinence graduée des chunks de preuve, avec gain/normalisation documentés |
| Exactitude de réponse | part des réponses answerable dont le contenu satisfait la réponse attendue et ses conditions temporelles |
| Précision des citations | part des citations qui sont à la fois exactes dans leur chunk et probantes pour le claim associé |
| Exactitude d'abstention | part des cas non answerable ou insuffisants correctement refusés |
| Faux positifs pièges | part des cas `out-of-scope` recevant une affirmation au lieu d'une abstention |
| Latence p50/p95 | percentiles mesurés séparément pour retrieval, reranking éventuel et réponse, avec cold/warm documenté |

La réponse extractive peut comporter plusieurs claims. Une réponse n'est exacte
que si aucun claim ne dépasse les preuves fournies. Les citations doivent être
vérifiées contre les offsets/extraits exacts, pas seulement contre la section.

## Tableau de résultats à remplir après exécution réelle

| Mode | Run / artefact | Recall@k | MRR | nDCG@k | Exactitude réponse | Précision citations | Exactitude abstention | Faux positifs pièges | p50 / p95 | Statut |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dense | non mesuré | non mesuré | non mesuré | non mesuré | non mesuré | non mesuré | non mesuré | non mesuré | non mesuré | en attente du harnais |
| Hybride RRF | non mesuré | non mesuré | non mesuré | non mesuré | non mesuré | non mesuré | non mesuré | non mesuré | non mesuré | en attente du harnais |
| Hybride RRF + reranker | non mesuré | non mesuré | non mesuré | non mesuré | non mesuré | non mesuré | non mesuré | non mesuré | non mesuré | reranker non mesuré |

## Observations vérifiées avant mesure

Les tests G3 vérifient, sans modèle ni réseau, la fusion RRF, la déduplication,
la provenance des rangs, le chargement paresseux et le fail-closed du reranker,
l'expansion de contexte tabulaire, l'API et des cas de passage réel tels que
la ligne Analamanga (`198.0`) et Michael Randrianirina. Ce sont des garanties
de comportement unitaire ; elles ne constituent pas des métriques de retrieval
ni une comparaison expérimentale.

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
