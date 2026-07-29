# Preuves de release

Ce document est le registre de preuve de la release. Une ligne ne peut passer à
`validée` qu'avec une sortie réellement obtenue, datée et liée au commit
publié. Il ne remplace ni les rapports JSON ni la CI et ne contient aucun
secret.

## Périmètre figé

| Élément | Valeur vérifiable |
|---|---|
| Source de connaissance | Page Wikipedia anglaise `Madagascar` uniquement |
| Révision | `1365949107` |
| Horodatage de la révision | `2026-07-25T10:41:18Z` |
| Snapshot | `data/raw/manifest.json` |
| Calibrage | `data/eval/questions.jsonl` |
| Holdout | `data/eval/holdout.jsonl`, jamais utilisé pour calibrer retrieval, reranking, prompts ni seuils |

## Contrat de génération

Le défaut `extractive` est local et ne demande aucune clé. Les fournisseurs
`openai` et `openai-compatible` sont facultatifs ; ils nécessitent
`uv sync --extra openai`, `MADA_RAG_GENERATION_PROVIDER`,
`MADA_RAG_GENERATION_MODEL` et `MADA_RAG_LLM_API_KEY`. Le second exige aussi
`MADA_RAG_LLM_BASE_URL`.

Une réponse en français ou en anglais conserve la langue demandée pour son
texte et ses claims. Les citations restent toutefois des extraits anglais
caractère pour caractère issus des chunks récupérés. Toute citation hors
contexte, extrait altéré, JSON invalide, erreur du fournisseur ou preuve
insuffisante aboutit à une abstention sans claim factuel.

## Métriques publiées

| Mesure | Définition et limite |
|---|---|
| `evidence_recall_at_k` | Part moyenne des chunks attendus récupérés dans les `k` premiers résultats pour les cas answerable. |
| `hit_rate_at_k` | Part des cas answerable ayant au moins un chunk attendu dans les `k` premiers résultats. |
| `complete_evidence_rate_at_k` | Part des cas answerable dont tous les chunks attendus sont dans les `k` premiers résultats. |
| `answer_accuracy` | Égalité **exacte normalisée** entre `answer_text` et `expected_answer` : Unicode NFKC, minuscules et espaces compactés. Ce n'est pas une mesure de similarité sémantique, d'entailment ou de style. |
| `answerability_status_accuracy` | Exactitude du statut `answered` ou `abstained` face à la réponse attendue. |
| `trap_false_positive_rate` | Part des cas hors périmètre auxquels le système répond tout de même. |
| Validité/précision des citations | Contrôles de provenance/offset/extrait et proxy fondé sur les preuves attendues ; ce n'est pas une revue humaine d'entailment. |
| Latences | `cold_ms` est le premier échantillon, les champs `warm_*` les suivants, et `max_ms` le maximum observé. Elles dépendent de la machine, du cache et des modèles. |

## Tableau de preuves à compléter par le release manager

| Gate | Commande ou lien de preuve | Statut | À renseigner après exécution réelle |
|---|---|---|---|
| Qualité locale | `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src`, `uv run pytest` | À compléter | Commit, date, sorties et code de retour |
| CI GitHub | [workflow CI](https://github.com/memphisfils/mada-rag/actions/workflows/ci.yml) | À compléter | URL du run, SHA du commit, conclusion et horodatage |
| Clone propre | `git clone … && uv sync --frozen --all-extras --group dev` | À compléter | OS/Python/uv, SHA, log d'installation et commandes de validation |
| Snapshot | Vérification de `data/raw/manifest.json` et de son SHA-256 | À compléter | SHA observé, commande et date |
| Évaluation calibrage | `uv run mada-rag evaluate …` avec rapports JSON versionnés | À compléter | Commande exacte, SHA des rapports, timestamp, paramètres et matériel |
| Évaluation holdout | Même protocole, après gel de la configuration | À compléter | Confirmation écrite d'absence de calibrage, rapport séparé et métriques réellement mesurées |
| Secrets | Scan du dépôt et de l'historique autorisé par la procédure de release | À compléter | Outil/commande, périmètre, date et résultat sans exposer de valeur sensible |

Ne renseignez jamais une métrique, une CI verte ou une réussite de clone propre
par anticipation. Les valeurs non exécutées restent « À compléter ».
