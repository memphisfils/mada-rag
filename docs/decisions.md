# Journal des décisions techniques

Ce journal complète l'architecture de référence. Une décision n'est modifiée
qu'explicitement ; les expériences de retrieval ne changent pas silencieusement
le corpus, le snapshot ou le jeu d'évaluation.

| ID | Décision | Statut | Conséquence vérifiable |
|---|---|---|---|
| D001 | Le runtime est limité à Python 3.12 et les dépendances sont verrouillées par `uv.lock`. | Acceptée | CI et clone propre utilisent `.python-version` et `uv sync --frozen`. |
| D002 | Le fournisseur de génération est désactivé par défaut. | Acceptée | Une installation fraîche ne demande aucun secret et n'appelle aucun LLM. |
| D003 | Le seul titre source accepté par le manifeste est `Madagascar`. | Acceptée | `SnapshotManifest` rejette tout autre titre et fixe `source_count` à 1. |
| D004 | Les objets de domaine Pydantic sont stricts, immuables et refusent les champs supplémentaires. | Acceptée | Les coercitions ou métadonnées inconnues échouent avant indexation. |
| D005 | Les dates de révision et de capture doivent inclure un fuseau horaire. | Acceptée | Un manifeste à date naïve est invalide. |
| D006 | Les tableaux ont des lignes rectangulaires après normalisation. | Acceptée | `TableRecord` rejette une ligne dont la largeur diffère des en-têtes. |
| D007 | Une réponse affirmative est un graphe cohérent question → claims → citations → chunks récupérés. | Acceptée | `Answer` rejette une affirmation sans citation, une citation inventée ou un chunk non récupéré. |
| D008 | L'abstention est fail-closed et ne contient aucune affirmation factuelle. | Acceptée | Une abstention exige un motif et interdit claims/citations. |
| D009 | Les index, caches et données transformées sont régénérables ; snapshots et rapports restent versionnables. | Acceptée | `.gitignore` exclut les premiers mais permet `data/raw` et `artifacts/reports`. |
| D010 | La licence MIT couvre le code ; le contenu Wikipedia garde sa provenance et sa licence propres. | Acceptée | Le manifeste expose nom et URL de licence du snapshot. |
