# Registre des risques

| ID | Risque | Probabilité | Impact | Signal de détection | Réduction | Responsable | État |
|---|---|---:|---:|---|---|---|---|
| R001 | Une actualisation silencieuse de Wikipedia change les réponses. | Haute | Critique | Révision ou SHA-256 différent | Révision figée, manifeste, commande d'actualisation explicite | Terra | Ouvert |
| R002 | Une autre page ou la mémoire du LLM contamine la connaissance. | Moyenne | Critique | Citation sans chunk local, appel réseau pendant `ask` | Allowlist source, runtime hors ligne, validation du graphe de preuves | Terra | Ouvert |
| R003 | `rowspan`/`colspan` produit des lignes de tableau incorrectes. | Haute | Élevé | Largeur différente, valeurs décalées | Normalisation rectangulaire et fixtures dédiées | Terra/Luna | Ouvert |
| R004 | Les chunks dépassent la fenêtre effective d'E5. | Moyenne | Élevé | Troncature tokenizer | Comptage token-aware, partitions de table, tests de limite | Terra | Ouvert |
| R005 | BM25 retrouve mal une question française sur un corpus anglais. | Haute | Moyen | Recall lexical faible par langue | E5 multilingue, RRF et métriques séparées FR/EN | Terra | Ouvert |
| R006 | Une citation existe mais ne prouve pas réellement l'affirmation. | Moyenne | Critique | Revue humaine ou entailment négatif | Claims atomiques, extraits exacts, revue adversariale | Luna | Ouvert |
| R007 | Les seuils d'abstention sont surajustés aux cas finaux. | Moyenne | Élevé | Écart calibration/final | Séparation des jeux, paramètres gelés avant mesure finale | Root | Ouvert |
| R008 | FAISS ou PyTorch est indisponible sur une plateforme. | Moyenne | Élevé | Échec `uv sync` ou import | Python 3.12 verrouillé, CI Linux et clone propre | Terra | Ouvert |
| R009 | Le reranker excède le budget mémoire ou latence. | Moyenne | Moyen | Timeout, OOM, p95 élevé | Composant optionnel, pool borné, latences cold/warm | Terra | Ouvert |
| R010 | Un secret est commité ou journalisé. | Faible | Critique | Scan de secrets positif | `.env` ignoré, `SecretStr`, génération désactivée par défaut | Root | Ouvert |
| R011 | Un chiffre d'évaluation est publié sans exécution réelle. | Faible | Critique | Absence d'artefact ou commande reproductible | Rapports issus du harnais, `non mesuré` sinon | Root | Ouvert |
| R012 | Deux agents modifient le même fichier. | Moyenne | Élevé | Conflit ou diff inattendu | Verrou d'affectation et revue du diff avant fusion | Root | Ouvert |
