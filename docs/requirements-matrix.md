# Matrice de traçabilité des exigences

## Convention de statut

- `G0 validé` : décision ou document approuvé à la gate d'architecture.
- `Fixture prête` : donnée synthétique disponible, sans présumer du comportement du code.
- `À implémenter` : code ou artefact de produit non encore validé.
- `À mesurer` : résultat interdit à estimer ; une exécution réelle est requise.
- `Release` : contrôle final à réaliser sur le dépôt public.

La fixture HTML associée est explicitement synthétique. Elle ne constitue ni un snapshot de
production ni une source de vérité pour les réponses d'évaluation.

## Cadrage et livraison

| ID | Exigence | Artefact ou preuve | Validation attendue | Statut initial |
|---|---|---|---|---|
| DEL-01 | Livrer en trois jours calendaires | `docs/implementation-plan.md` | jalons, priorités et gates explicites | G0 validé |
| DEL-02 | Dépôt Git public | URL GitHub de release | clone anonyme possible | Release |
| DEL-03 | Code complet et exécutable | package, lockfile, commandes | installation depuis clone propre | À implémenter |
| DEL-04 | README d'installation et lancement local | `README.md` | commandes copiées dans un environnement vierge | À implémenter |
| DEL-05 | Démo utilisable | CLI et petite API | une question sourcée et un refus démontrables | À implémenter |
| DEL-06 | Envoi du lien avec objet `TEST IA` | remise à `test_ia_072026@tikasa.net` | lien public transmis par l'utilisateur | Release |

## Source, snapshot et périmètre

| ID | Exigence | Artefact ou preuve | Validation attendue | Statut initial |
|---|---|---|---|---|
| SRC-01 | Une seule source de connaissance | allowlist et module d'ingestion | seule la page anglaise `Madagascar` est acceptée | À implémenter |
| SRC-02 | Inclure texte et tableaux de la page | snapshot HTML brut | sections et tables présentes | À implémenter |
| SRC-03 | Ne suivre aucun lien interne | parseur et test de transport | aucune requête vers la cible du lien | Fixture prête |
| SRC-04 | Ne faire aucun enrichissement Web | architecture et journal d'exécution | aucune seconde URL de connaissance | G0 validé |
| SRC-05 | Enregistrer la révision | manifeste du snapshot | `revision_id` non vide et affiché | À implémenter |
| SRC-06 | Enregistrer les dates | manifeste et réponse | `revision_timestamp` et `fetched_at` ISO 8601 | À implémenter |
| SRC-07 | Détecter un changement silencieux | SHA-256 du HTML | hash recalculé identique au manifeste | À implémenter |
| SRC-08 | Actualisation uniquement explicite | commande CLI dédiée | aucun refresh pendant `ask` ou `index` | À implémenter |
| SRC-09 | Ne pas utiliser la fixture comme vérité | marqueurs HTML et documentation | contenu identifié `synthetic` et exclu des données d'évaluation | Fixture prête |

## Ingestion, parsing et chunking

| ID | Exigence | Artefact ou preuve | Validation attendue | Statut initial |
|---|---|---|---|---|
| ING-01 | Acquisition MediaWiki justifiée | client API et documentation | titre/révision de la réponse vérifiés | À implémenter |
| ING-02 | Nettoyer les références `[n]` | parseur BeautifulSoup | marqueurs et section References exclus | Fixture prête |
| ING-03 | Nettoyer scripts/navigation/édition | parseur | `script`, `navbox`, `mw-editsection` absents des chunks | Fixture prête |
| ING-04 | Conserver le texte visible des liens | parseur | ancre présente, URL non utilisée comme connaissance | Fixture prête |
| ING-05 | Préserver les sections imbriquées | `SectionRecord.section_path` | chemin H2/H3 exact et ordre stable | Fixture prête |
| ING-06 | Préserver dates, nombres et unités | texte normalisé | valeurs non tronquées et exposants normalisés | Fixture prête |
| ING-07 | Gérer l'infobox | `TableRecord` ou champs structurés | labels et valeurs accessibles | Fixture prête |
| ING-08 | Gérer caption, thead et tbody | parseur de tableaux | légende et en-têtes attachés aux cellules | Fixture prête |
| ING-09 | Résoudre `rowspan` et `colspan` | grille normalisée | zone propagée, en-têtes groupés sans décalage | Fixture prête |
| ING-10 | Rendre un tableau globalement interrogeable | chunk `table_global` | légende, en-têtes et toutes lignes reliées | À implémenter |
| ING-11 | Rendre chaque ligne interrogeable | chunks `table_row` | une région récupérable avec ses en-têtes | À implémenter |
| ING-12 | Définir taille et chevauchement | configuration de chunking | comptage tokenizer, limites respectées | G0 validé |
| ING-13 | Découper sur frontières sémantiques | chunker section/paragraphe | aucune fusion arbitraire de sections | À implémenter |
| ING-14 | Produire des IDs stables | fonction d'identification | mêmes entrées, mêmes IDs et ordres | À implémenter |
| ING-15 | Valider par contrats Pydantic | modèles typés | donnée invalide refusée explicitement | À implémenter |

## Indexation et retrieval multilingue

| ID | Exigence | Artefact ou preuve | Validation attendue | Statut initial |
|---|---|---|---|---|
| RET-01 | Justifier le modèle d'embedding | architecture et README | compromis multilingue/latence documenté | G0 validé |
| RET-02 | Utiliser `multilingual-e5-base` correctement | encodeur | préfixes `query:` et `passage:` vérifiés | À implémenter |
| RET-03 | Répondre à des questions FR et EN | jeu d'évaluation bilingue | preuves pertinentes dans les deux langues | À implémenter |
| RET-04 | Index dense FAISS | index et métadonnées | normalisation, recherche et reload testés | À implémenter |
| RET-05 | Recherche BM25 | index lexical | noms, nombres, unités et dates conservés | À implémenter |
| RET-06 | Recherche hybride par RRF | fusionneur | rangs fusionnés, dédupliqués et traçables | À implémenter |
| RET-07 | Reranking configurable | cross-encoder/adaptateur | fallback propre et nouveau rang exposé | À implémenter |
| RET-08 | Expansion contrôlée des tables | assembleur de contexte | ligne et table globale reliées sans source externe | À implémenter |
| RET-09 | Comparer dense, hybride et rerank | harnais d'évaluation | corpus, questions et paramètres communs | À mesurer |
| RET-10 | Justifier les paramètres | configuration et rapport | top-k, RRF, seuils et budget consignés | À mesurer |

## Génération, citations et abstention

| ID | Exigence | Artefact ou preuve | Validation attendue | Statut initial |
|---|---|---|---|---|
| GEN-01 | Générer uniquement depuis les chunks récupérés | prompt et service | contexte borné et IDs visibles au modèle | À implémenter |
| GEN-02 | Ne pas stocker la connaissance dans le LLM | architecture | aucun fine-tuning ni mémoire externe | G0 validé |
| GEN-03 | Citer chaque affirmation | modèle `Answer`/`Claim` | toute affirmation a au moins une citation | À implémenter |
| GEN-04 | Rendre les citations vérifiables | `Citation` | chunk, section, extrait et révision retrouvables | À implémenter |
| GEN-05 | Rejeter les citations inventées | validateur post-génération | ID hors contexte provoque un refus | À implémenter |
| GEN-06 | S'abstenir sans preuve suffisante | gate de suffisance | question piège non affirmée | À implémenter |
| GEN-07 | Répondre prudemment aux cas partiels | statut et limite explicite | seule la partie démontrée est affirmative | À implémenter |
| GEN-08 | Désambiguïser le temps | réponse et citation | « actuel » rattaché à la date du snapshot | À implémenter |
| GEN-09 | Échouer de façon sûre | politique fail-closed | erreur LLM/validation retourne une abstention | À implémenter |

## Interfaces

| ID | Exigence | Artefact ou preuve | Validation attendue | Statut initial |
|---|---|---|---|---|
| INT-01 | CLI Typer | commandes `ingest/index/ask/retrieve/evaluate/serve` | aide et codes de sortie stables | À implémenter |
| INT-02 | Petite API FastAPI | `/healthz`, `/v1/ask`, `/v1/retrieve` | schémas et erreurs documentés | À implémenter |
| INT-03 | Logique commune CLI/API | service applicatif | réponses équivalentes à entrée identique | À implémenter |
| INT-04 | Exposer provenance et latence | schéma de réponse | statut, citations, révision, timing présents | À implémenter |
| INT-05 | Limiter les entrées API | configuration | requête trop grande refusée proprement | À implémenter |

## Évaluation

| ID | Exigence | Artefact ou preuve | Validation attendue | Statut initial |
|---|---|---|---|---|
| EVA-01 | Au moins 15 questions | dataset versionné | nombre de cas supérieur ou égal à 15 | À implémenter |
| EVA-02 | Couvrir le fait simple | cas annoté | attendu et preuve source présents | À implémenter |
| EVA-03 | Couvrir le chiffre précis | cas annoté | valeur et unité vérifiables | À implémenter |
| EVA-04 | Couvrir la lecture de tableau | cas annoté | preuve globale/ligne indiquée | À implémenter |
| EVA-05 | Couvrir le multi-passages | cas annoté | plusieurs preuves attendues | À implémenter |
| EVA-06 | Couvrir l'ambiguïté temporelle | cas annoté | réponse rattachée au snapshot | À implémenter |
| EVA-07 | Couvrir le hors périmètre | cas `answerable=false` | abstention attendue | À implémenter |
| EVA-08 | Couvrir la question partielle | cas annoté | éléments démontrables et limites indiqués | À implémenter |
| EVA-09 | Fournir réponses et sources attendues | schéma `EvalCase` | annotations relues avant mesure | À implémenter |
| EVA-10 | Mesurer le taux de bonnes réponses | rapport généré | formule, numérateur et dénominateur visibles | À mesurer |
| EVA-11 | Mesurer les faux positifs des pièges | rapport généré | réponses affirmatives injustifiées comptées | À mesurer |
| EVA-12 | Mesurer le temps de réponse | rapport généré | p50/p95, cold/warm et matériel indiqués | À mesurer |
| EVA-13 | Mesurer le retrieval | rapport généré | Recall@k, MRR/nDCG selon annotations | À mesurer |
| EVA-14 | Mesurer la qualité des citations/refus | rapport généré | précision citations et exactitude abstention | À mesurer |
| EVA-15 | Ne rien inventer | artefacts d'exécution | non exécuté affiché `non mesuré` | À mesurer |

## Qualité, sécurité, documentation et release

| ID | Exigence | Artefact ou preuve | Validation attendue | Statut initial |
|---|---|---|---|---|
| QUA-01 | Code propre et modulaire | package `src/mada_rag` | responsabilités séparées | À implémenter |
| QUA-02 | Gérer erreurs et cas limites | exceptions et réponses typées | échecs explicites, sans données silencieusement perdues | À implémenter |
| QUA-03 | Tests automatisés | pytest | unitaires et intégration verts | À implémenter |
| QUA-04 | Lint et types | Ruff, mypy | commandes CI vertes | À implémenter |
| QUA-05 | CI GitHub Actions | workflow | installation, tests, lint et types | À implémenter |
| SEC-01 | Aucun secret commité | `.env.example`, ignore et scan | scan du dépôt et de l'historique propre | Release |
| SEC-02 | Variables d'environnement pour clés | configuration | absence de clé par défaut dans le code | À implémenter |
| SEC-03 | Contenu HTML non fiable | parseur et limites | scripts et instructions parasites exclus | Fixture prête |
| LIC-01 | Attribuer le contenu Wikipedia | snapshot/README | URL, révision et CC BY-SA indiquées | À implémenter |
| LIC-02 | Séparer licence code/contenu | fichiers de licence | périmètres clairement distingués | Release |
| DOC-01 | Architecture et choix justifiés | `docs/architecture.md` | embeddings, chunking, stores, LLM et compromis | G0 validé |
| DOC-02 | Plan, risques, décisions, restant | `docs/implementation-plan.md` | propriétaires, dépendances et gates | G0 validé |
| DOC-03 | Limites connues | README et rapport | limites concrètes, non promotionnelles | À implémenter |
| DOC-04 | Résultats d'évaluation | `docs/evaluation-report.md` | générés à partir d'une exécution réelle | À mesurer |
| DOC-05 | Réflexion de deux paragraphes | README ou rapport | améliorations réalistes pour deux semaines | À implémenter |
| REL-01 | Tests avant chaque fusion | preuve CI/commande | gate verte avant intégration | À implémenter |
| REL-02 | Revue indépendante par phase | journal des gates | relecteur différent de l'auteur principal | À implémenter |
| REL-03 | Installation depuis clone propre | journal de release | parcours complet reproduit | Release |
| REL-04 | Revue adversariale finale | checklist et défauts | aucun défaut bloquant restant | Release |
