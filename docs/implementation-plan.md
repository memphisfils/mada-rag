# Plan d'implémentation en trois jours

## Règles d'exécution

- `root` orchestre, arbitre, intègre et gère la release.
- Terra réalise les composants principaux et les intégrations.
- Luna réalise fixtures, tests répétitifs, documentation et petites corrections.
- Un fichier n'a qu'un propriétaire actif. `root` tient le verrou d'affectation avant chaque tâche.
- Toute fusion exige tests ciblés, suite complète disponible, Ruff et mypy.
- Chaque phase est revue par une personne qui n'en a pas écrit le composant principal.
- Les résultats d'évaluation ne sont commités qu'après une commande réelle et reproductible.
- Toute décision ou dérive met à jour les journaux ci-dessous avant poursuite.

## Plan par jour

| Jour | Objectif | Sortie vérifiable |
|---|---|---|
| Jour 1 matin | Contrats, snapshot, parsing et fixtures | HTML+manifeste, sections/tableaux validés |
| Jour 1 après-midi | Parcours vertical minimal dense | `ingest -> index -> ask`, citation/refus, CLI |
| Jour 2 matin | Tables robustes, BM25, RRF | retrieval hybride testé et explicable |
| Jour 2 après-midi | Reranking, API et jeu d'évaluation | trois modes, endpoints, au moins 15 cas |
| Jour 3 matin | Exécutions et analyse | rapport réel dense/hybride/rerank |
| Jour 3 après-midi | Clone propre et release | CI verte, revue adversariale, dépôt public |

## Phases, tâches et responsabilités

| ID | Tâche et fichiers réservés | Propriétaire | Dépend de | Tests/gate | Revue |
|---|---|---|---|---|---|
| P0.1 | Architecture et plan dans `docs/` | Luna | cadrage | cohérence exigences | root |
| P0.2 | Scaffolding, dépendances, CI, configuration | Terra | P0.1 | `uv sync`, commandes qualité | root |
| P0.3 | Matrice exigences et critères de release | root | P0.1 | audit documentaire | Luna |
| P1.1 | Modèles Pydantic et manifeste | Terra | P0.2 | sérialisation, validation | root |
| P1.2 | Snapshot MediaWiki révisionné | Terra | P1.1 | hash, révision, allowlist | Luna |
| P1.3 | Fixtures HTML sections/tableaux | Luna | contrat P1.1 | fixture chargeable | Terra |
| P1.4 | Parseur sections et nettoyage | Terra | P1.2-P1.3 | tests unitaires ciblés | root |
| P1.5 | Parseur tableaux global et ligne | Terra | P1.3 | 23 régions, en-têtes, unités | Luna |
| P1.6 | Tests répétitifs de parsing/cas limites | Luna | P1.4-P1.5 figés | suite parsing verte | root |
| P2.1 | Chunking token-aware et IDs stables | Terra | P1 | tailles, overlap, stabilité | Luna |
| P2.2 | Embeddings E5 et FAISS | Terra | P2.1 | dimension, normalisation, reload | root |
| P2.3 | Retrieval dense et contexte | Terra | P2.2 | top-k et preuves attendues | Luna |
| P2.4 | Génération structurée, citations, abstention | Terra | P2.3 | refus et IDs invalides | root |
| P2.5 | CLI vertical minimal | Terra | P2.4 | smoke test bout en bout | Luna |
| P2.6 | Tests d'intégration verticaux | Luna | P2.5 stable | scénario online figé/offline | root |
| P3.1 | BM25 et normalisation lexicale | Terra | P2.1 | noms, nombres, dates | Luna |
| P3.2 | RRF, déduplication et expansion tables | Terra | P3.1-P2.3 | ordre et provenance | root |
| P3.3 | Reranker configurable | Terra | P3.2 | fallback et classement | Luna |
| P3.4 | FastAPI sur service partagé | Terra | P2.4 | health/ask/retrieve | root |
| P3.5 | Tests API et retrieval paramétrés | Luna | P3.2-P3.4 stables | suite sans réseau | Terra |
| P4.1 | Jeu d'au moins 15 cas bilingues | Luna | snapshot P1.2 | catégories et preuves | root |
| P4.2 | Harnais et métriques d'évaluation | Terra | P3, P4.1 | calculs sur fixture | root |
| P4.3 | Exécution des trois modes | root | P4.2 | artefacts horodatés | Terra |
| P4.4 | Rapport d'évaluation, limites, réflexion | Luna | résultats P4.3 | aucun placeholder chiffré | root |
| P5.1 | Installation depuis clone propre | root | P0-P4 | procédure README exacte | Luna |
| P5.2 | Revue adversariale FR/EN, pièges, citations | Luna | P5.1 | journal des défauts | Terra |
| P5.3 | Corrections bloquantes par propriétaire de fichier | Terra/Luna | P5.2 | non-régression complète | root |
| P5.4 | Scan secrets, licence, tag et publication | root | P5.3 | gate release verte | Luna |

Les tâches Luna portant sur les tests commencent seulement après gel du contrat Terra correspondant. Terra ne modifie pas les fichiers de tests affectés à Luna ; Luna ne corrige pas le code applicatif sans réaffectation explicite par `root`.

## Gates de fusion et revues indépendantes

| Gate | Conditions minimales | Relecteur indépendant |
|---|---|---|
| G0 Architecture | décisions, risques, DoD approuvés | root |
| G1 Données | snapshot traçable, parsing texte/tableaux vert | Luna |
| G2 Vertical | question dense sourcée et piège refusé via CLI | root |
| G3 Retrieval | dense, hybride et rerank testés | Luna |
| G4 Produit | API, CLI et jeu d'évaluation complets | root |
| G5 Release | mesures réelles, clone propre, sécurité, licence | Luna |

Avant fusion, l'auteur fournit la commande exécutée et son résultat. Une revue peut approuver, demander une correction au propriétaire, ou bloquer. Aucun résultat « attendu » n'est présenté comme résultat « obtenu ».

## Stratégie de tests

- Unitaires : modèles, nettoyage, sections, tables, chunking, RRF, citations.
- Propriétés : IDs déterministes, ordre stable, aucun chunk vide, citations incluses dans le contexte.
- Intégration : snapshot fixture vers index, question vers réponse, abstention hors périmètre.
- API/CLI : codes de sortie, schémas, erreurs sans secret, parité de comportement.
- Évaluation : métriques testées sur petits exemples calculables à la main.
- Release : clone dans un répertoire vierge, `uv sync --frozen`, tests, lint, types, smoke tests.

Les tests CI n'appellent ni Wikipedia ni un LLM payant. Les tests réseau et génération réelle sont des jobs explicites ; leurs artefacts portent la configuration utilisée.

## Journal des décisions

| ID | Décision | Statut |
|---|---|---|
| D1 | Révision MediaWiki figée et hashée | accepté |
| D2 | Chunks de tableau global plus ligne | accepté |
| D3 | E5 multilingue sans traduction | accepté |
| D4 | FAISS FlatIP normalisé et BM25 fusionnés par RRF | accepté |
| D5 | Reranker optionnel à l'exécution, obligatoire dans la comparaison finale | accepté |
| D6 | Sortie par affirmations citées et abstention fail-closed | accepté |
| D7 | Typer et FastAPI partagent le service applicatif | accepté |

## Registre des risques

| Risque | Signal | Responsable | Réponse |
|---|---|---|---|
| Révision cible ne contient pas un exemple 2025 | cas attendu introuvable | root | documenter le snapshot, ne jamais enrichir |
| Tableau mal interprété (`rowspan`/`colspan`) | nombre ou en-têtes incohérents | Terra | normalisation et fixture dédiée |
| Modèle trop lourd pour CI | timeout/mémoire | root | mocks CI, smoke réel documenté |
| Seuil d'abstention surajusté | écart calibration/final | Terra | séparation des cas et paramètres gelés |
| Faux support par simple citation | citation non probante | Luna | revue manuelle adversariale |
| Dépendance LLM indisponible | génération impossible | Terra | adaptateur et erreur/abstention propre |
| Secret dans historique | scan positif | root | blocage release et rotation du secret |
| Travail concurrent sur un fichier | conflit ou écrasement | root | verrou d'affectation et réattribution |

## Travaux restants

- [ ] Approuver G0 et créer le scaffolding.
- [ ] Fixer puis valider le snapshot et son manifeste.
- [ ] Livrer le parcours dense minimal avant les optimisations.
- [ ] Livrer hybride puis reranking avec tests séparés.
- [ ] Geler les questions et preuves avant l'exécution finale.
- [ ] Générer le rapport à partir des sorties réelles.
- [ ] Compléter README, limites et réflexion de deux paragraphes.
- [ ] Réussir clone propre, revue adversariale et scan des secrets.
- [ ] Publier le dépôt et préparer le lien de remise.

## Definition of Done de release

G0 à G5 sont approuvés ; chaque tâche fusionnée possède ses preuves de test ; aucun fichier n'a été édité simultanément ; les trois variantes sont comparées sans chiffres inventés ; toutes les affirmations de démonstration sont citées ou refusées ; le dépôt public s'installe et s'exécute depuis un clone propre ; risques résiduels, décisions et limites sont explicitement documentés.
