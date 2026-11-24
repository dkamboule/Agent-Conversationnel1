# Squelette autonome du moteur RAG hybride DGECC

Ce répertoire est généré automatiquement par `export_rag_bundle.py`.

## Contenu
- `../config/rag_config.json` : Paramètres du moteur RAG, modèles Gemini, organigramme DGECC et consigne système stricte.
- `../corpus/documents_metadata.json` : Métadonnées documentaires exportées (sans données personnelles ni secrets).
- `../corpus/chunks_vectors.jsonl` : Segments et vecteurs d'embeddings (format JSONL ligne par ligne).
- `../corpus/manifest.json` : Sommaire statistique et empreintes SHA-256 d'intégrité.
- `dgecc_rag_skeleton.py` : Moteur RAG hybride autonome (BM25 + Vecteurs + RRF + routage organigramme DGECC).

## Exécution rapide
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="votre_cle_api"
python3 skeleton/dgecc_rag_skeleton.py "Quelles sont les dates des épreuves écrites du Baccalauréat 2026 ?"
```
