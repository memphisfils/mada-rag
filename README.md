# Mada RAG

RAG local, bilingue (français/anglais), limité à **une seule révision** de la
page Wikipédia anglaise [Madagascar](https://en.wikipedia.org/wiki/Madagascar).
Il répond par extraits exacts accompagnés de citations structurées, ou refuse
de répondre lorsque les chunks récupérés ne suffisent pas.

Le snapshot versionné est la révision `1365949107`, datée du
`2026-07-25T10:41:18Z` et capturée le `2026-07-28T15:29:12.618740Z`. Son
manifeste et son SHA-256 se trouvent dans
[`data/raw/manifest.json`](data/raw/manifest.json). Les réponses dites
« actuelles » signifient donc « actuelles dans ce snapshot », jamais dans le
Web en direct.

## Garanties de périmètre

- La seule connaissance métier autorisée est le HTML de cette page et cette
  révision. L'ingestion ne suit aucun lien et ne consulte aucune page annexe.
- `ask` et `retrieve` travaillent sur les artefacts locaux ; ils n'appellent
  pas Wikipedia ni un moteur de recherche.
- Chaque réponse affirmative contient des citations d'extraits présents dans
  les chunks récupérés. Le validateur rejette une citation ou un ID forgé.
- Une preuve trop faible, partielle ou contradictoire produit une abstention,
  sans claim ni citation factuelle.

L'architecture détaillée est dans [`docs/architecture.md`](docs/architecture.md).

## Architecture

```text
HTML Wikipedia révisionné
  -> sections + tableaux normalisés
  -> chunks texte, partitions et lignes de tableau
  -> E5/FAISS dense + BM25 lexical
  -> RRF, puis reranker optionnel
  -> expansion contrôlée de contexte tabulaire
  -> contrôle de suffisance
  -> extraits exacts + citations, ou abstention
```

Les tableaux sont représentés à la fois globalement (ou en partitions) et par
ligne. Cela permet une recherche par entité et des comparaisons, par exemple
sur les 23 régions. Les questions françaises interrogent directement le corpus
anglais grâce à `intfloat/multilingual-e5-base`; aucune traduction générative
n'ajoute de connaissance.

## Prérequis et installation

- Python **3.12** (voir [`.python-version`](.python-version))
- [`uv`](https://docs.astral.sh/uv/)
- mémoire et disque suffisants pour FAISS et les modèles Sentence Transformers
  lors du premier indexage/usage du reranker

```bash
git clone https://github.com/memphisfils/mada-rag.git
cd mada-rag
uv sync --frozen --all-extras --group dev
uv run pytest
uv run ruff check .
uv run mypy src
```

Le HTML brut et son manifeste sont versionnés sous `data/raw/`. Les index et
fichiers transformés sont reconstruis localement et ne sont pas requis dans le
dépôt.

## Configuration et secrets

Copiez éventuellement l'exemple, sans jamais versionner `.env` :

```bash
cp .env.example .env
```

Sous PowerShell, remplacez `cp` par `Copy-Item .env.example .env`. La
configuration utilise le préfixe `MADA_RAG_`; les paramètres de retrieval,
ports, limites de requête et chemins sont documentés dans
[`.env.example`](.env.example).

Le chemin de génération livré est **extractif et sans secret** :
`MADA_RAG_GENERATION_PROVIDER=extractive`. Les variables `MADA_RAG_LLM_API_KEY`
et `MADA_RAG_LLM_BASE_URL` sont réservées à une intégration future ; le CLI
actuel refuse tout fournisseur autre qu'extractif. Ne placez aucune clé dans le
code, le dépôt, les commandes shell historisées ou les rapports.

## Commandes

Toutes les commandes s'exécutent avec `uv run mada-rag`.

### Ingestion et index

```bash
# Commande réseau : récupère uniquement la page Madagascar via MediaWiki.
uv run mada-rag ingest

# Recrée sections, chunks et l'index E5/FAISS depuis le snapshot local.
uv run mada-rag index
# Pour remplacer un index existant :
uv run mada-rag index --overwrite
```

Pour reproduire exactement la release, ne relancez pas `ingest` : il peut
obtenir une révision plus récente. Reprenez plutôt le HTML et le manifeste
versionnés, puis lancez `index`. Le premier indexage télécharge le modèle E5 si
son cache est vide ; ce téléchargement ne constitue pas une source de
connaissance.

### Retrieval et réponses

```bash
uv run mada-rag retrieve "Which region has the highest population density?" --mode hybrid
uv run mada-rag ask "Quelle est la capitale de Madagascar ?" --language fr --mode hybrid
uv run mada-rag ask "What is Madagascar's official flower?" --mode dense
```

Modes disponibles : `dense`, `hybrid`, `hybrid-rerank`. Le mode `hybrid` combine
E5/FAISS et BM25 par Reciprocal Rank Fusion. `hybrid-rerank` charge à la demande
le cross-encoder configuré par `MADA_RAG_RERANKER_MODEL`; il peut donc demander
un téléchargement de modèle et augmenter la latence. Il a fait l'objet d'un
run calibré de release : son `Recall@5` est de `0.863636`, mais ses latences
retrieval p50/p95 (`21 756.536` / `26 181.638` ms) restent très supérieures à
celles du mode hybride RRF. **`hybrid` est le défaut recommandé** ;
`hybrid-rerank` est un opt-in expérimental. Consulter les limites, les baselines
historiques et les artefacts dans le
[rapport d'évaluation](docs/evaluation-report.md) avant de le choisir.

`retrieve` émet les chunks avec rangs et provenance dense/BM25/RRF/reranker.
`ask` émet un objet JSON `Answer` : `answered` porte claims et citations;
`abstained` porte un `refusal_reason` et aucune affirmation. Une citation
contient l'ID du chunk, le chemin de section, l'extrait exact, la révision et,
le cas échéant, le tableau/la ligne.

### Évaluation

Le jeu d'évaluation versionné est
[`data/eval/questions.jsonl`](data/eval/questions.jsonl) : 25 cas bilingues,
avec réponses/preuves attendues et catégories (faits, chiffres, tableaux,
multi-passages, ambiguïtés temporelles, pièges et couverture partielle).

Les runs finaux calibrés dense, hybride et hybride+reranker sont archivés sous
[`artifacts/reports/`](artifacts/reports/) ; les baselines pré-calibration sont
conservées dans le même dossier pour audit. Les métriques, limites et la règle
de non-comparabilité des baselines sont documentées dans
[`docs/evaluation-report.md`](docs/evaluation-report.md).

```bash
uv run mada-rag evaluate --mode dense --mode hybrid --top-k 5 \
  --output artifacts/reports/evaluation-calibrated-dense-hybrid.json

MADA_RAG_RERANKER_ENABLED=true uv run mada-rag evaluate \
  --mode hybrid-rerank --top-k 5 \
  --output artifacts/reports/evaluation-calibrated-hybrid-rerank.json
```

Le second run charge le cross-encoder optionnel. Conserver les rapports JSON,
leurs horodatages et leurs hashes pour toute comparaison reproductible.

### API locale

```bash
uv run mada-rag serve --host 127.0.0.1 --port 8000
```

L'API FastAPI charge le service de manière paresseuse : `GET /healthz` ne
charge pas de modèle. Les endpoints sont `POST /v1/retrieve` et `POST /v1/ask`;
ils limitent les questions, refusent les champs inconnus et retournent des
schémas Pydantic validés.

```bash
curl http://127.0.0.1:8000/healthz
curl -X POST http://127.0.0.1:8000/v1/ask \
  -H 'content-type: application/json' \
  -d '{"question":"Qui est le président actuel selon ce snapshot ?","language":"fr"}'
```

## Tests et reproductibilité

La CI exécute le lockfile, Ruff, le contrôle de formatage, mypy et pytest.
Les tests unitaires n'ont ni accès réseau ni besoin de modèle téléchargé. Les
tests G3 emploient une fixture versionnée, indépendante des artefacts ignorés
générés localement, y compris pour les scénarios de tableaux et API.

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

## Limites connues

- La connaissance est volontairement bornée à une page et une révision : une
  information absente, postérieure ou présente seulement dans un lien doit être
  refusée.
- La réponse extractive est sûre et traçable, mais moins naturelle qu'une
  synthèse LLM contrôlée.
- Le gate d'ancrage est calibré pour le run de release ; sa robustesse doit être
  réévaluée sur des formulations inédites et une séparation de calibration. Le
  même jeu versionné de 25 cas et ses chunks attendus ont servi à l'ajustement
  puis aux runs calibrés : ces résultats prouvent une régression contrôlée et
  la conformité de release, pas une estimation non biaisée de généralisation.
- Les artefacts finaux montrent une forte pénalité de latence pour le reranker
  (p95 retrieval `26 181.638` ms) : le traiter comme un opt-in expérimental,
  non comme un défaut. Les métriques de contenu restent des proxies fondés sur
  chunks/citations attendus, pas une validation sémantique LLM.

## Licence et attribution

Le code de ce dépôt est sous [licence MIT](LICENSE). Le contenu de Wikipédia
reste sous [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Toute redistribution du snapshot ou de sorties substantielles doit conserver
l'attribution à la page *Madagascar*, son URL, la révision `1365949107` et les
obligations de partage à l'identique applicables. Le manifeste conserve ces
informations de provenance et de licence.

## Avec deux semaines de plus

Avec deux semaines supplémentaires, je commencerais par une revue humaine
d'entailment, par catégorie et par langue, afin de compléter les proxies de
chunks/citations des runs calibrés. Je testerais le gate d'ancrage sur une
séparation de calibration et des reformulations inédites, et je mesurerais les
latences cold/warm sur une machine cible au lieu de les résumer en une moyenne.

Je renforcerais ensuite l'usage produit : test d'entailment ou revue humaine
des citations, génération structurée sous contexte strict avec un adaptateur
LLM local ou API, cache/versionnement des modèles, mesures de robustesse aux
formulations FR/EN et CI de clone propre complète. Je documenterais également
un processus d'actualisation explicitement versionné, afin que la fraîcheur
du snapshot n'efface jamais la reproductibilité des résultats passés.
