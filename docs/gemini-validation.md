# Validation Gemini via OpenAI-compatible

- Date UTC : `2026-07-30T08:15:52Z`
- SHA testé : `be863f3788e7457bbfcc390122301b7d59f58380`
- Provider : `openai-compatible`
- Modèle : `gemini-3.6-flash`
- Endpoint compatible Google : `https://generativelanguage.googleapis.com/v1beta/openai/`
- Sécurité : aucune clé n'est versionnée ; `.env` reste ignoré par Git.

## Commandes exécutées

```powershell
uv sync --frozen --extra openai --group dev
uv run mada-rag ask "Quelle est la capitale de Madagascar ?" --language fr --mode hybrid
uv run mada-rag evaluate --file data/eval/holdout.jsonl --mode hybrid --top-k 5 --output artifacts/reports/evaluation-gemini-3.6-flash-hybrid.json
```

Le smoke test est `PASS` : statut `answered`, réponse française contenant
`Antananarivo`, provider/modèle attendus, une claim liée à une citation, et
un extrait anglais exact du chunk récupéré.

## Validation manuelle des six cas diagnostiques

| Cas | Statut | Langue | Faits requis | Citations valides | Verdict | Justification |
|---|---|---|---|---|---|---|
| `holdout-highest-point-en` | `answered` | EN | Oui | Oui | PASS | Maromokotro, Tsaratanana Massif et `2,876 m (9,436 ft)` sont explicitement fournis. |
| `holdout-baobab-species-fr` | `abstained` | FR | Non | N/A | FAIL | Le chunk attendu est récupéré mais le générateur s'abstient au lieu d'indiquer six espèces sur neuf. |
| `holdout-vakinankaratra-capital-en` | `answered` | EN | Oui | Oui | PASS | La réponse cite la ligne de tableau indiquant Antsirabe. |
| `holdout-antsirabe-rank-fr` | `answered` | FR | Oui | Oui | PASS | La réponse donne le 3e rang et 245 592 habitants. |
| `holdout-cuisine-components-en` | `answered` | EN | Oui | Oui | PASS | La réponse donne rice/vary et accompaniment/laoka. |
| `holdout-space-agency-en` | `abstained` | EN | Oui | Oui (zéro citation) | PASS | Refus conforme, sans claim ni citation factuelle. |

## Métriques observées

- `manual_factual_accuracy` : `4/5 = 0.80`
- `answerability_status_accuracy` : `5/6 = 0.8333`
- `trap_false_positive_rate` : `0/1 = 0.0`
- `citation_validity` : `4/4 = 1.0`
- Erreurs fournisseur : `0`
- Erreurs de validation locale : `0`

Le rapport automatique conserve `answer_accuracy=0.0`, car cette métrique
existante compare une égalité textuelle normalisée à la réponse attendue ; elle
n'est ni modifiée ni utilisée comme jugement sémantique ici. Les six cas sont
diagnostiques : aucun prompt, seuil, index ou paramètre n'a été réglé à partir
d'eux.
