# Architecture du système Mada RAG

## Statut et objectifs

Ce document fixe l'architecture de référence avant l'implémentation. Le système répond en français ou en anglais uniquement à partir d'un snapshot révisionné de la page Wikipedia anglaise `Madagascar`. Une réponse affirmative doit être démontrable par les chunks récupérés ; sinon le système s'abstient.

Principes non négociables :

- une seule source de connaissance, sans suivi des liens internes ni recherche secondaire ;
- snapshot immuable, reproductible et attribuable ;
- même corpus et même jeu d'évaluation pour toutes les variantes de retrieval ;
- citations structurées et vérifiables pour chaque affirmation ;
- aucun résultat d'évaluation publié sans exécution réelle ;
- fonctionnement local documenté, secrets uniquement par variables d'environnement.

## Décisions d'architecture

| Sujet | Décision | Justification |
|---|---|---|
| Runtime | Python 3.12, `uv`, Pydantic | Environnement rapide à reproduire et contrats typés |
| Acquisition | MediaWiki API, page `Madagascar`, révision figée | Capture le HTML et les métadonnées sans consulter d'autre page |
| Parsing | BeautifulSoup, parcours DOM par sections | Préserve la hiérarchie et permet un traitement dédié des tableaux |
| Chunking texte | Frontières de section et paragraphe, environ 350 tokens, overlap 50 | Reste sous la limite E5 et conserve le contexte local |
| Chunking tableaux | représentation globale plus un chunk par ligne | Permet comparaisons globales et recherches par entité |
| Embeddings | `intfloat/multilingual-e5-base`, préfixes `query:`/`passage:` | Retrieval direct français-anglais sans traduction générative |
| Index dense | FAISS `IndexFlatIP`, vecteurs normalisés | Recherche cosinus exacte, adaptée au petit corpus |
| Index lexical | BM25 sur texte normalisé, nombres et unités conservés | Renforce noms propres, dates et valeurs exactes |
| Fusion | Reciprocal Rank Fusion, constante initiale 60 | Fusionne des rangs hétérogènes sans calibrer leurs scores bruts |
| Reranking | cross-encoder multilingue configurable sur les candidats fusionnés | Variante mesurable, désactivable pour coût ou latence |
| Génération | adaptateur LLM configurable, sortie JSON Pydantic | Sépare le fournisseur et rend citations/refus validables |
| Interfaces | Typer et FastAPI sur le même service applicatif | Évite deux comportements métier divergents |

Les valeurs de `top_k`, overlap, seuils d'abstention, constante RRF et taille du contexte sont configurables. Elles sont calibrées sur un sous-ensemble prévu à cet effet, jamais ajustées sur les résultats finaux.

## Flux de bout en bout

```text
MediaWiki API (Madagascar uniquement)
  -> snapshot HTML + manifeste de révision + SHA-256
  -> sections, paragraphes et tableaux normalisés
  -> chunks texte + table globale + lignes de table
  -> embeddings E5/FAISS et corpus BM25
  -> dense | dense+BM25/RRF | dense+BM25/RRF+reranking
  -> expansion contrôlée des lignes/tableaux et assemblage du contexte
  -> contrôle de suffisance
  -> génération structurée
  -> validation affirmation-citations
  -> réponse sourcée ou abstention
```

L'ingestion est la seule étape qui accède à Wikipedia. L'indexation et les réponses utilisent le snapshot local. Les appels éventuels au LLM ne peuvent apporter aucune connaissance : le prompt interdit explicitement tout fait absent du contexte.

## Modèles de données

| Modèle | Champs essentiels |
|---|---|
| `SnapshotManifest` | URL canonique, titre, `revision_id`, `revision_timestamp`, `fetched_at`, SHA-256, version du parseur |
| `SectionRecord` | identifiant, chemin de titres, ordre, texte normalisé |
| `TableRecord` | `table_id`, section, légende, en-têtes, lignes, ordre |
| `Chunk` | ID stable, type, texte, section, table/ligne éventuelle, révision, ordinal, token count |
| `RetrievedChunk` | chunk, rangs dense/BM25/RRF, score reranker, provenance |
| `Citation` | chunk ID, section, table/ligne, extrait justificatif, révision |
| `Answer` | statut `answered` ou `abstained`, texte, affirmations et citations, motif de refus |
| `EvalCase` | question, langue, catégorie, attendu, answerable, preuves attendues |

Les IDs de chunks sont dérivés de la révision, du chemin source, du type et de l'ordinal. Ils restent stables pour un même snapshot.

## Snapshot, parsing et chunking

L'ingestion enregistre côte à côte le HTML brut et un manifeste JSON. Elle vérifie le titre et la révision retournés, calcule le hash et refuse un changement silencieux. Une commande explicite est nécessaire pour actualiser le snapshot.

Le parseur :

- ignore navigation, scripts, styles, édition, références et notes de bas de page ;
- conserve le texte visible des liens mais ne suit jamais leur cible ;
- conserve titres, paragraphes, listes, légendes, en-têtes, unités et dates ;
- normalise les espaces et marque les valeurs manquantes sans les inventer ;
- traite l'infobox et les tableaux de contenu avec le même modèle tabulaire.

Chaque tableau produit un chunk global compact contenant légende, en-têtes et lignes, plus un chunk autonome par ligne répétant les en-têtes. Si le tableau global dépasse la limite du modèle, il est partitionné avec le même `table_id`; récupérer une partition déclenche l'expansion de ses partitions sœurs dans la limite du budget de contexte.

## Retrieval

Le mode dense encode les requêtes avec le préfixe E5 `query:` et les chunks avec `passage:`. Le mode hybride récupère des candidats FAISS et BM25, déduplique par chunk ID puis applique RRF. Le troisième mode reranke ce pool avec un cross-encoder multilingue.

Les lignes de tableau récupérées peuvent entraîner l'ajout du chunk global ; un chunk global peut entraîner l'ajout des lignes les plus pertinentes. Cette expansion est tracée et n'introduit aucun texte extérieur. Les trois modes exposent leurs rangs et latences afin que l'évaluation reste explicable.

## Génération, citations et abstention

Le générateur reçoit la question, la date/révision du snapshot et des passages balisés par ID. Il doit produire des affirmations séparées, chacune associée à au moins une citation. Pour une question temporelle, les dates présentes dans les preuves sont conservées ; le mot « actuel » signifie « actuel dans ce snapshot ».

Avant génération, un contrôle calibré refuse les contextes trop faibles. Après génération, le validateur vérifie :

- que chaque citation appartient aux chunks effectivement récupérés ;
- que chaque affirmation factuelle possède une citation ;
- que l'extrait cité existe dans le chunk ;
- qu'aucun identifiant ou passage non fourni n'a été créé.

Tout échec produit une abstention sûre. Une question partiellement couverte reçoit uniquement la partie démontrable et signale explicitement la limite. Le motif de refus est observable sans exposer de secret ni de prompt interne.

## Interfaces

- CLI : `ingest`, `index`, `ask`, `retrieve`, `evaluate`, `serve`.
- API : `GET /healthz`, `POST /v1/ask`, `POST /v1/retrieve`.
- Les réponses contiennent statut, texte, citations, révision et latence.
- L'ingestion et la reconstruction d'index restent des opérations CLI explicites.

## Évaluation

Le jeu versionné contient au moins 15 questions : fait simple, chiffre précis, tableau, multi-passages, ambiguïté temporelle, hors périmètre et couverture partielle, en français et en anglais. Chaque cas indique la réponse attendue et les preuves acceptables.

Les trois modes sont mesurés sur le même snapshot et la même configuration : Recall@k, MRR/nDCG lorsque pertinent, exactitude de réponse, précision des citations, exactitude d'abstention, taux de faux positifs sur pièges, latences p50/p95. Le rapport consigne matériel, modèles, paramètres et timestamp. Une valeur non exécutée est marquée `non mesuré`.

## Sécurité, licence et structure

Les clés sont lues depuis l'environnement ; seul `.env.example` est versionné. Les entrées API ont des limites de taille, les erreurs sont neutralisées et le HTML est traité comme contenu non fiable. Un scan de secrets précède la release.

Le snapshot conserve URL et révision pour l'attribution Wikipedia. La documentation rappelle la licence CC BY-SA applicable au contenu ; la licence du code est déclarée séparément.

```text
src/mada_rag/{config,models,ingestion,parsing,chunking,indexing,retrieval,generation,service,cli,api,evaluation}.py
tests/{fixtures,unit,integration}
data/{raw,processed,eval}
artifacts/{indexes,reports}
docs/{architecture.md,implementation-plan.md,evaluation-report.md}
```

## Risques majeurs

| Risque | Réduction |
|---|---|
| Page évolutive ou « président actuel » ambigu | révision et date visibles dans toute réponse |
| Perte de structure des tableaux | fixtures réelles, chunks global/ligne et tests de comparaison |
| Faiblesse BM25 en français | embeddings multilingues et comparaison par langue |
| Citation présente mais non probante | sortie par affirmation, extraits exacts et revue adversariale |
| Abstention trop stricte ou permissive | calibration séparée, pièges et cas partiels |
| Premier lancement lent | cache documenté et séparation cold/warm latency |
| Fuite de secret ou connaissance externe | environnement, scan, source allowlistée, pas de suivi de liens |

## Definition of Done

Le dépôt est terminé lorsque : un clone public propre s'installe avec `uv`; le snapshot et sa révision sont vérifiables; ingestion, parsing, tables, indexation et trois retrievals s'exécutent; CLI et API répondent avec citations ou abstention; les tests, Ruff et mypy passent; au moins 15 cas réels sont évalués sans résultat inventé; le rapport, les limites et la réflexion de deux paragraphes sont présents; aucun secret n'est détecté; une revue indépendante et une revue adversariale ne laissent aucun défaut bloquant.
