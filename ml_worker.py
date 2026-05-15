"""
ml_worker.py — Proceso hijo efímero para embeddings con SentenceTransformer.

Uso:
    python ml_worker.py <query_json_base64>

Imprime en stdout un JSON con el embedding y termina.
El proceso padre (app.py) lo lanza con subprocess y lee el resultado.
Al terminar, el SO recupera TODO el RAM de PyTorch inmediatamente.
"""
import sys
import os
import json
import base64
import numpy as np


def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "uso: ml_worker.py <query_b64>"}))
        sys.exit(1)

    try:
        payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
        query   = payload["query"]
    except Exception as e:
        print(json.dumps({"error": f"payload inválido: {e}"}))
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer
        import torch

        torch.set_num_threads(1)

        model_cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME", "./model_cache")
        os.makedirs(model_cache_dir, exist_ok=True)

        model  = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=model_cache_dir)
        vector = model.encode(query, convert_to_numpy=True).astype(np.float32)

        print(json.dumps({"embedding": vector.tolist()}))
        sys.exit(0)

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
