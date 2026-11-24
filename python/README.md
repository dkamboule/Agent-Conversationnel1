# Assistant DGECC — Version Python (RAG Hybride, OCR & Garde-fous)

Ce dossier contient le portage complet en **Python 3.11+** du moteur conversationnel RAG hybride de la **DGECC** (*Direction Générale des Examens, Concours et de la Certification*), aligné sur l'architecture TypeScript/Node.js du projet.

## Architecture modulaire

- `dgecc_assistant/core/` : Configuration, organigramme officiel DGECC (5 directions & services), sécurité (`scrypt`, JWT HMAC-SHA256, anti-injection, rate-limiting) et client Gemini résilient (retry exponentiel + fallback `gemini-3.1-flash-lite` / `gemini-4-hydrogen`).
- `dgecc_assistant/database/` : Connexion SQLite WAL compatible avec `data/dgecc.db` et journalisation d'audit.
- `dgecc_assistant/document_processing/` : Extraction multi-formats (`PDF`, `PNG`, `JPG`, `WEBP`, `DOCX`, `HTML`, `TXT`), OCR multimodal Gemini Vision pour documents scannés, normalisation et découpage (chunking) avec chevauchement.
- `dgecc_assistant/embeddings/` : Génération et mise en cache LRU des vecteurs unitaires (`gemini-embedding-2`).
- `dgecc_assistant/search/` : Index corpus en mémoire, recherche lexicale **BM25** française et recherche sémantique vectorielle.
- `dgecc_assistant/nlp/` : Analyse déterministe des intentions, entités, dates et orientation vers le service compétent de l'organigramme DGECC (+ reformulation anaphorique multi-tours).
- `dgecc_assistant/rag/` : Fusion **RRF** (*Reciprocal Rank Fusion*), construction du contexte avec la **consigne système stricte imposée**, et orchestrateur RAG hybride avec cache instantané.
- `dgecc_assistant/guardrails/` : Vérification numérique/factuelle, détection d'expiration et messages de refus normalisés.
- `dgecc_assistant/main.py` : Serveur REST & SSE **FastAPI**.
- `dgecc_assistant/cli.py` : Interface en ligne de commande pour interroger l'assistant directement en Python.

## Installation

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Définissez la variable d'environnement `GEMINI_API_KEY` côté serveur (sans jamais l'exposer côté client) :

```bash
export GEMINI_API_KEY="votre_cle_api"
```

## Démarrage du serveur FastAPI

```bash
cd python
uvicorn dgecc_assistant.main:app --host 0.0.0.0 --port 8000 --reload
```

## Utilisation en ligne de commande (CLI)

```bash
cd python
python3 -m dgecc_assistant.cli "Quelles sont les dates des épreuves écrites du Baccalauréat 2026 ?"
```

## Export de la base documentaire (vecteurs + métadonnées) et squelette RAG

```bash
python3 python/scripts/export_rag_bundle.py \
  --db-path data/dgecc.db \
  --output-dir python/exports/dgecc_rag_bundle \
  --zip
```

Cet utilitaire génère :
- `config/rag_config.json` (paramètres RAG, modèles Gemini, organigramme DGECC, consigne système stricte)
- `corpus/documents_metadata.json` et `corpus/chunks_vectors.jsonl` (métadonnées et vecteurs d'embeddings)
- `corpus/manifest.json` (sommaire et empreintes SHA-256)
- `skeleton/dgecc_rag_skeleton.py` (squelette autonome Python du moteur RAG hybride prêt à l'emploi)

## Exécution des tests unitaires

```bash
cd python
python3 -m unittest discover -s tests -v
```
