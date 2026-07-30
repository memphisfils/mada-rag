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

## Preuves observées avant l'archivage

Les commandes et résultats ci-dessous ont été exécutés le 29 juillet 2026 UTC
sur le commit `5f77b58cb31a6fc7ec4d09b934dbfce071617d53`. Les changements
documentaires postérieurs nécessitent leur propre CI ; cette section ne les
présente pas par anticipation comme déjà validés.

| Gate | Commande ou lien de preuve | Résultat réellement observé |
|---|---|---|
| Qualité locale | `uv run ruff check .`; `uv run ruff format --check .`; `uv run mypy src`; `uv run pytest -q` | Ruff et format OK, mypy OK, `188 passed` (un avertissement externe Starlette/httpx). |
| CI GitHub | [run 30492765067](https://github.com/memphisfils/mada-rag/actions/runs/30492765067) | `success`, déclenché manuellement sur `5f77b58`, terminé à `2026-07-29T21:33:39Z`; lint, format, mypy et tests verts. |
| Publication `main` | `git push origin HEAD:refs/heads/main` puis `git ls-remote --heads origin main` | Le SHA local et `refs/heads/main` étaient tous deux `5f77b58cb31a6fc7ec4d09b934dbfce071617d53`; aucun push forcé. |
| Visibilité publique | `curl -I -H "Authorization:" https://github.com/memphisfils/mada-rag` et lectures raw de `README.md` / `.github/workflows/ci.yml` | HTTP `200`; README et workflow visibles sans en-tête d'autorisation. |
| Clone propre | `git clone --depth 1 --branch main https://github.com/memphisfils/mada-rag.git`; `uv sync --frozen --all-extras --group dev`; Ruff, format, `pytest -q`, `mada-rag --help` | Clone sur `5f77b58`; installation gelée réussie (extra OpenAI inclus); Ruff/format OK et `188 passed` (même avertissement externe). |
| Snapshot | `data/raw/manifest.json` | Révision `1365949107`, SHA-256 HTML `c54a3df9ca9650a99b717e0b235bb2845593b69a96d744becaea6c0eac4e3a4a`. |

## Exécution holdout réellement mesurée

Configuration gelée : fallback `extractive`, aucune clé définie, `top_k=5`,
snapshot `1365949107`, index SHA-256
`5d94f75a4c00b34f497bc7c457f275139564daca702ddd6a91eb708523dfbe61`.
Les commandes ont été exécutées séparément par mode afin d'isoler un crash
natif Windows observé lors du lancement combiné dense+hybrid (`-1073741819`,
aucun rapport combiné écrit). Les deux rapports suivants sont donc les seules
mesures publiées :

| Mode | Commande | Rapport et SHA-256 | Résultat |
|---|---|---|---|
| dense | `uv run mada-rag evaluate --file data/eval/holdout.jsonl --mode dense --top-k 5 --output artifacts/reports/evaluation-holdout-extractive-dense.json` | `evaluation-holdout-extractive-dense.json` — `7C7400C8C311C7EC2BA883A0F6B067021A4C2A2CA701613DD5F563A4F7BE5E6C` | evidence recall/hit/complete `1.0`; citation validity `1.0`; answer accuracy `0.0`; answerability status accuracy `0.6667`; trap false-positive rate `1.0`. |
| hybrid | `uv run mada-rag evaluate --file data/eval/holdout.jsonl --mode hybrid --top-k 5 --output artifacts/reports/evaluation-holdout-extractive-hybrid.json` | `evaluation-holdout-extractive-hybrid.json` — `B2C16C68F426B2B2063B0B91C4D35E5635A932CFE028260C5F6C909487CC122F` | evidence recall/hit/complete `1.0`; citation validity `1.0`; answer accuracy `0.0`; answerability status accuracy `0.6667`; trap false-positive rate `1.0`. |

Le holdout est inédit et n'a servi à aucun réglage. Ces résultats montrent que
le fallback extractif récupère les preuves du holdout mais ne satisfait pas le
seuil de qualité de réponse ni l'abstention sur son piège. Il ne faut donc pas
déclarer une qualification fonctionnelle de ce fallback sur ce holdout. Les
baselines historiques sont conservées et ne sont ni remplacées ni recalibrées
à partir de ces six cas.

## Validation Gemini via le provider OpenAI-compatible

Gemini a servi uniquement à valider l'interface générique
`openai-compatible` avec `gemini-3.6-flash` et l'endpoint compatible Google.
Le dépôt n'ajoute ni SDK Gemini ni provider Gemini spécifique : les providers
`extractive`, `openai` et `openai-compatible` restent disponibles. Un
recruteur peut fournir sa propre clé OpenAI par environnement sans modifier le
code ; aucune clé n'est stockée dans Git.

Les six cas de `data/eval/holdout.jsonl` sont diagnostiques, pas un holdout
indépendant destiné au réglage. Le run réel est archivé dans
`artifacts/reports/evaluation-gemini-3.6-flash-hybrid.json` et sa validation
manuelle dans `docs/gemini-validation.md` : quatre réponses factuellement
correctes sur cinq, une abstention answerable, statut correct sur cinq cas sur
six, aucun faux positif sur le piège et quatre citations valides sur quatre.
La métrique automatique `answer_accuracy` reste à `0.0` à cause de son
comparateur volontairement textuel strict ; elle ne mesure pas la fidélité
sémantique de ce run.
