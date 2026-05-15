from flask import Flask, render_template, request, jsonify
import sqlite3
import numpy as np
from pathlib import Path
import unicodedata
import re
import os
import ctypes
from openai import AzureOpenAI
import httpx
import time
from threading import Lock, Timer
import sys
import gc
import faiss

# ------------------------------------------------------------
# Rate limiter con limpieza automática de entradas antiguas
# ------------------------------------------------------------
class RateLimiter:
    def __init__(self, max_calls=30, period=60):
        self.max_calls = max_calls
        self.period = period
        self.calls = {}
        self.lock = Lock()

    def _cleanup(self, now):
        expired = []
        for client_id, timestamps in self.calls.items():
            valid = [t for t in timestamps if now - t < self.period]
            if valid:
                self.calls[client_id] = valid
            else:
                expired.append(client_id)
        for client_id in expired:
            del self.calls[client_id]

    def is_allowed(self, client_id):
        with self.lock:
            now = time.time()
            self._cleanup(now)
            if client_id not in self.calls:
                self.calls[client_id] = []
            if len(self.calls[client_id]) >= self.max_calls:
                return False
            self.calls[client_id].append(now)
            return True

# ------------------------------------------------------------
# Cache con TTL (reemplaza lru_cache estático que retiene self)
# ------------------------------------------------------------
class TTLCache:
    """Cache clave→valor con expiración por tiempo de vida."""
    def __init__(self, maxsize=200, ttl=3600):
        self.maxsize = maxsize
        self.ttl     = ttl
        self._cache  = {}       # key → (value, timestamp)
        self._lock   = Lock()

    def get(self, key):
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            value, ts = entry
            if time.time() - ts > self.ttl:
                del self._cache[key]
                return None
            return value

    def set(self, key, value):
        with self._lock:
            if len(self._cache) >= self.maxsize:
                oldest = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest]
            self._cache[key] = (value, time.time())

    def clear(self):
        with self._lock:
            self._cache.clear()

    def __len__(self):
        return len(self._cache)

# ------------------------------------------------------------
# Traductor principal
# ------------------------------------------------------------
class YoremnokkilTranslator:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_lock = Lock()
        self._initialized = False

        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key='embedding_dim'")
        self.embedding_dim = int(cursor.fetchone()[0])
        cursor.execute("SELECT value FROM metadata WHERE key='total_pairs'")
        self.total_pairs = int(cursor.fetchone()[0])

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        azure_key = os.getenv("AZURE_OPENAI_KEY", "")
        azure_deployment = os.getenv("AZURE_DEPLOYMENT", "gpt-4.1-mini")
        azure_model = os.getenv("AZURE_MODEL", "gpt-4.1-mini")

        self.azure_rate_limiter = RateLimiter(max_calls=30, period=60)
        self.azure_client    = None
        self._azure_http     = None   # httpx.Client propio — se cierra en _azure_close()
        self.azure_deployment = str(azure_deployment)
        self.azure_model      = str(azure_model)
        self._azure_cache     = TTLCache(maxsize=200, ttl=3600)
        self._last_request_ts = 0.0
        self._idle_timer      = None
        self._idle_lock       = Lock()
        # Tiempo de inactividad (segundos) tras el cual se cierra el cliente Azure
        self._idle_timeout    = int(os.getenv("AZURE_IDLE_TIMEOUT", 300))

        if azure_endpoint and azure_key:
            try:
                endpoint_clean = str(azure_endpoint).strip().rstrip('/')
                if '/openai' in endpoint_clean:
                    endpoint_clean = endpoint_clean.split('/openai')[0]
                if not endpoint_clean.endswith('/'):
                    endpoint_clean += '/'
                if endpoint_clean and azure_key:
                    self._azure_endpoint = endpoint_clean
                    self._azure_key      = azure_key
                    self._init_azure_client()
                    print(f"✓ Azure OpenAI configurado (deployment: {self.azure_deployment})")
            except Exception as e:
                print(f"⚠ Error configurando Azure OpenAI: {e}")
        else:
            print("⚠ Azure OpenAI no configurado (sin corrección gramatical)")

        self.espanol_raw = None
        self.yoremnokki_raw = None
        self.espanol_norm = None
        self.yoremnokki_norm = None
        self.esp_index = None
        self.yor_index = None
        self.model = None

    # --------------------------------------------------------
    # Gestión del cliente Azure (init, idle timeout, close)
    # --------------------------------------------------------
    def _init_azure_client(self):
        """Instancia httpx.Client y AzureOpenAI con connection pool controlado."""
        self._azure_http = httpx.Client(
            timeout=httpx.Timeout(10.0),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        )
        self.azure_client = AzureOpenAI(
            api_key=self._azure_key,
            api_version="2024-12-01-preview",
            azure_endpoint=self._azure_endpoint,
            http_client=self._azure_http,
        )

    def _azure_close(self):
        """Cierra el connection pool de httpx y libera el cliente Azure."""
        with self._idle_lock:
            if self.azure_client is not None:
                try:
                    self._azure_http.close()
                except Exception:
                    pass
                self.azure_client  = None
                self._azure_http   = None
                gc.collect()
                print("♻ Cliente Azure cerrado por inactividad (RAM liberada)")

    def _reset_idle_timer(self):
        """Reinicia el temporizador de inactividad tras cada llamada a Azure."""
        with self._idle_lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            self._idle_timer = Timer(self._idle_timeout, self._azure_close)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _ensure_azure_client(self):
        """Re-crea el cliente si fue cerrado por idle y se necesita de nuevo."""
        with self._idle_lock:
            if self.azure_client is None and hasattr(self, '_azure_key'):
                print("🔄 Reconectando cliente Azure...")
                self._init_azure_client()

    # --------------------------------------------------------
    # Paths para índices FAISS serializados
    # --------------------------------------------------------
    def _index_paths(self):
        base = Path(self.db_path).parent
        return (
            base / "faiss_esp.index",
            base / "faiss_yor.index",
            base / "faiss_texts.npz"
        )

    # --------------------------------------------------------
    # Carga lazy: intenta leer índices desde disco, si no los
    # construye UNA sola vez y los persiste para arranques futuros
    # --------------------------------------------------------
    def _lazy_init(self):
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return

            esp_path, yor_path, txt_path = self._index_paths()
            indexes_on_disk = esp_path.exists() and yor_path.exists() and txt_path.exists()

            if indexes_on_disk:
                print("📂 Cargando índices FAISS desde disco (sin reconstruir)...")
                self.esp_index = faiss.read_index(str(esp_path))
                self.yor_index = faiss.read_index(str(yor_path))

                data = np.load(str(txt_path), allow_pickle=True)
                self.espanol_raw    = data["espanol_raw"].tolist()
                self.yoremnokki_raw = data["yoremnokki_raw"].tolist()
                self.espanol_norm   = data["espanol_norm"].tolist()
                self.yoremnokki_norm= data["yoremnokki_norm"].tolist()
                print(f"✓ Índices cargados desde disco ({self.total_pairs} pares)")

                # Importar SentenceTransformer solo cuando hace falta
                self._load_sentence_transformer()

                self._initialized = True
                return

            # Primera vez: construir y persistir
            self._build_and_persist_indexes(esp_path, yor_path, txt_path)

    @staticmethod
    def _malloc_trim():
        """
        Devuelve al SO la RAM que glibc/PyTorch tienen en su heap pero marcada
        como libre. Sin esta llamada el SO (y Railway) la sigue contando como
        consumo aunque Python ya no la use.
        """
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass  # no-op en macOS / Windows

    def _load_sentence_transformer(self):
        """Importa y carga SentenceTransformer de forma diferida."""
        if self.model is not None:
            return
        # Importación diferida: PyTorch no se carga hasta este momento
        from sentence_transformers import SentenceTransformer as ST
        import torch

        # Desactivar el cache interno de memoria de PyTorch en CPU.
        # Por defecto PyTorch retiene bloques libres en su propio allocator
        # y nunca los devuelve al SO; esto lo deshabilita.
        try:
            torch.set_num_threads(1)              # limitar threads innecesarios
        except Exception:
            pass

        model_cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME", "./model_cache")
        os.makedirs(model_cache_dir, exist_ok=True)
        print("🔧 Cargando modelo SentenceTransformer (all-MiniLM-L6-v2)...")
        try:
            self.model = ST('all-MiniLM-L6-v2', cache_folder=model_cache_dir)
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise RuntimeError("No se pudo cargar el modelo.") from e
        print("✓ Modelo cargado")

    def _build_and_persist_indexes(self, esp_path, yor_path, txt_path):
        """Construye índices FAISS desde la DB y los guarda en disco."""
        self._load_sentence_transformer()

        print("📖 Construyendo índices FAISS desde base de datos...")
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT espanol, yoremnokki, esp_embedding, yor_embedding FROM traducciones"
        )

        esp_raw_list = []
        yor_raw_list = []
        esp_norm_list = []
        yor_norm_list = []
        esp_emb_list = []
        yor_emb_list = []

        for row in cursor.fetchall():
            esp_raw, yor_raw, esp_blob, yor_blob = row
            esp_raw_list.append(esp_raw)
            yor_raw_list.append(yor_raw)
            esp_norm_list.append(self.normalize_text(esp_raw))
            yor_norm_list.append(self.normalize_text(yor_raw))
            esp_emb_list.append(np.frombuffer(esp_blob, dtype=np.float32))
            yor_emb_list.append(np.frombuffer(yor_blob, dtype=np.float32))

        self.espanol_raw    = esp_raw_list
        self.yoremnokki_raw = yor_raw_list
        self.espanol_norm   = esp_norm_list
        self.yoremnokki_norm= yor_norm_list

        esp_embs = np.array(esp_emb_list, dtype=np.float32)
        yor_embs = np.array(yor_emb_list, dtype=np.float32)
        del esp_emb_list, yor_emb_list
        gc.collect()

        faiss.normalize_L2(esp_embs)
        faiss.normalize_L2(yor_embs)

        dim = self.embedding_dim
        ids = np.arange(len(esp_raw_list), dtype=np.int64)

        self.esp_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.yor_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.esp_index.add_with_ids(esp_embs, ids)
        self.yor_index.add_with_ids(yor_embs, ids)
        del esp_embs, yor_embs
        gc.collect()

        # Persistir índices en disco para arranques futuros
        try:
            faiss.write_index(self.esp_index, str(esp_path))
            faiss.write_index(self.yor_index, str(yor_path))
            np.savez_compressed(
                str(txt_path),
                espanol_raw    = np.array(esp_raw_list,  dtype=object),
                yoremnokki_raw = np.array(yor_raw_list,  dtype=object),
                espanol_norm   = np.array(esp_norm_list, dtype=object),
                yoremnokki_norm= np.array(yor_norm_list, dtype=object),
            )
            print("✓ Índices persistidos en disco (próximos arranques serán más rápidos y ligeros)")
        except Exception as e:
            print(f"⚠ No se pudieron guardar índices en disco: {e}")

        self._initialized = True
        print(f"✓ Índices FAISS construidos ({self.total_pairs} pares)")

    # --------------------------------------------------------
    # Métodos auxiliares
    # --------------------------------------------------------
    def normalize_text(self, text):
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    # --------------------------------------------------------
    # Búsqueda exacta
    # --------------------------------------------------------
    def exact_match(self, query, direction='es2yor', allow_typos=False, max_edits=1):
        self._lazy_init()
        query_norm = self.normalize_text(query)
        source_raw  = self.espanol_raw    if direction == 'es2yor' else self.yoremnokki_raw
        source_norm = self.espanol_norm   if direction == 'es2yor' else self.yoremnokki_norm
        target_raw  = self.yoremnokki_raw if direction == 'es2yor' else self.espanol_raw

        matches = []
        for idx, txt in enumerate(source_raw):
            if txt.lower() == query.lower():
                matches.append({'id': idx, 'source': txt, 'target': target_raw[idx],
                                'edit_distance': 0, 'match_type': 'original_acento'})
        if matches:
            return matches[:3]

        for idx, norm in enumerate(source_norm):
            if norm == query_norm:
                matches.append({'id': idx, 'source': source_raw[idx], 'target': target_raw[idx],
                                'edit_distance': 0, 'match_type': 'normalized'})
        if matches:
            return matches[:3]

        if allow_typos:
            for idx, norm in enumerate(source_norm):
                len_diff = abs(len(norm) - len(query_norm))
                if len_diff <= max_edits:
                    dist = self.levenshtein_distance(norm, query_norm)
                    if dist <= max_edits:
                        matches.append({'id': idx, 'source': source_raw[idx], 'target': target_raw[idx],
                                        'edit_distance': dist, 'match_type': 'typo'})
            matches.sort(key=lambda x: x['edit_distance'])
            return matches[:3]
        return []

    # --------------------------------------------------------
    # División morfológica
    # --------------------------------------------------------
    def morphological_split_search(self, word, direction='yor2es', min_ratio=0.45, fuzzy_threshold=0.90):
        if direction != 'yor2es':
            return {'found': False, 'splits': []}
        self._lazy_init()

        word_norm = self.normalize_text(word)
        word_len  = len(word_norm)
        if word_len < 4:
            return {'found': False, 'splits': []}

        valid_splits = []
        source_norm = self.yoremnokki_norm
        target_raw  = self.espanol_raw

        for split_pos in range(int(word_len * min_ratio), int(word_len * 0.95) + 1):
            part1_norm = word_norm[:split_pos]
            part2_norm = word_norm[split_pos:]
            if len(part1_norm) < 1 or len(part2_norm) < 1:
                continue

            part2_indices = [i for i, n in enumerate(source_norm) if n == part2_norm]
            if not part2_indices:
                continue

            part1_candidates = []
            for idx, norm in enumerate(source_norm):
                if norm == part1_norm:
                    part1_candidates.append({'idx': idx, 'similarity': 1.0, 'exact': True})
                else:
                    max_len = max(len(norm), len(part1_norm))
                    if max_len == 0:
                        continue
                    sim = 1.0 - (self.levenshtein_distance(norm, part1_norm) / max_len)
                    if sim >= fuzzy_threshold:
                        part1_candidates.append({'idx': idx, 'similarity': sim, 'exact': False})

            for p1 in part1_candidates:
                for p2_idx in part2_indices:
                    combined_trans = f"{target_raw[p1['idx']]} {target_raw[p2_idx]}"
                    score = p1['similarity'] * 0.7 + 0.3
                    valid_splits.append({
                        'part1': source_norm[p1['idx']],
                        'part2': source_norm[p2_idx],
                        'part1_translation': target_raw[p1['idx']],
                        'part2_translation': target_raw[p2_idx],
                        'combined_translation': combined_trans,
                        'part1_similarity': p1['similarity'],
                        'part1_exact': p1['exact'],
                        'part2_exact': True,
                        'score': score,
                        'split_position': split_pos
                    })

        valid_splits.sort(key=lambda x: x['score'], reverse=True)
        return {'found': len(valid_splits) > 0, 'splits': valid_splits[:3]}

    # --------------------------------------------------------
    # Fuzzy token match (Jaccard)
    # --------------------------------------------------------
    def fuzzy_token_match(self, query, direction='es2yor', threshold=0.3):
        self._lazy_init()
        query_norm   = self.normalize_text(query)
        query_tokens = set(query_norm.split())
        source_norm  = self.espanol_norm   if direction == 'es2yor' else self.yoremnokki_norm
        target_raw   = self.yoremnokki_raw if direction == 'es2yor' else self.espanol_raw
        source_raw   = self.espanol_raw    if direction == 'es2yor' else self.yoremnokki_raw

        results = []
        for idx, item_norm in enumerate(source_norm):
            item_tokens = set(item_norm.split())
            inter   = query_tokens & item_tokens
            union   = query_tokens | item_tokens
            jaccard = len(inter) / len(union) if union else 0
            if jaccard >= threshold:
                results.append({'id': idx, 'source': source_raw[idx],
                                'target': target_raw[idx], 'jaccard_score': jaccard})
        results.sort(key=lambda x: x['jaccard_score'], reverse=True)
        return results

    # --------------------------------------------------------
    # Búsqueda por embeddings (FAISS)
    # --------------------------------------------------------
    def embedding_search(self, query, direction='es2yor', top_k=10, min_similarity=0.5):
        self._lazy_init()
        query_emb = self.model.encode(query, convert_to_numpy=True).reshape(1, -1).astype(np.float32)
        gc.collect()
        self._malloc_trim()   # devolver al SO la RAM libre del heap de PyTorch/glibc
        faiss.normalize_L2(query_emb)
        index = self.esp_index if direction == 'es2yor' else self.yor_index
        lims, D, I = index.range_search(query_emb, min_similarity)

        results = []
        for i in range(len(lims) - 1):
            start, end = lims[i], lims[i + 1]
            for j in range(start, end):
                idx   = I[j]
                score = float(D[j])
                results.append({
                    'id': idx,
                    'espanol':     self.espanol_raw[idx],
                    'yoremnokki':  self.yoremnokki_raw[idx],
                    'embedding_score': score
                })
        results.sort(key=lambda x: x['embedding_score'], reverse=True)
        return results[:top_k]

    # --------------------------------------------------------
    # Traducción composicional (ngrams)
    # --------------------------------------------------------
    def ngram_windows(self, tokens, max_n=3):
        windows = []
        n = len(tokens)
        for m in range(min(max_n, n), 0, -1):
            for i in range(n - m + 1):
                windows.append((i, i + m, ' '.join(tokens[i:i + m])))
        return windows

    def compositional_translate(self, query, direction='es2yor', max_window=3):
        self._lazy_init()
        tokens_orig = query.split()
        tokens_norm = self.normalize_text(query).split()
        n = len(tokens_norm)
        if n == 0:
            return {'success': False, 'chunks': []}

        covered = [False] * n
        chunks   = []
        windows  = self.ngram_windows(tokens_norm, max_window)

        for start, end, chunk_norm in windows:
            if any(covered[start:end]):
                continue
            chunk_orig = ' '.join(tokens_orig[start:end])
            matches    = self.exact_match(chunk_orig, direction, allow_typos=True, max_edits=1)
            if matches:
                best   = matches[0]
                target = best['target']
                chunks.append({
                    'start': start, 'end': end,
                    'source': chunk_orig,
                    'translation': target,
                    'match_type': best['match_type'],
                    'confidence': 1.0 - (best.get('edit_distance', 0) * 0.1)
                })
                for i in range(start, end):
                    covered[i] = True

        for i in range(n):
            if not covered[i]:
                token_orig = tokens_orig[i]
                if direction == 'yor2es':
                    morph = self.morphological_split_search(token_orig, direction='yor2es')
                    if morph['found']:
                        best_split = morph['splits'][0]
                        chunks.append({
                            'start': i, 'end': i + 1,
                            'source': token_orig,
                            'translation': best_split['combined_translation'],
                            'match_type': 'morphological',
                            'confidence': best_split['score'],
                            'morphological_detail': {
                                'part1': best_split['part1'],
                                'part2': best_split['part2'],
                                'part1_translation': best_split['part1_translation'],
                                'part2_translation': best_split['part2_translation']
                            }
                        })
                        covered[i] = True
                        continue
                chunks.append({
                    'start': i, 'end': i + 1,
                    'source': token_orig,
                    'translation': f"[{token_orig}]",
                    'match_type': 'unknown',
                    'confidence': 0.0
                })

        chunks.sort(key=lambda x: x['start'])
        success = any(c['match_type'] != 'unknown' for c in chunks)
        return {'success': success, 'chunks': chunks}

    # --------------------------------------------------------
    # Corrección gramatical con Azure
    # --------------------------------------------------------
    def corregir_gramatica_azure(self, texto_desordenado, client_id='default'):
        # Consultar TTLCache antes de llamar a la API
        cached = self._azure_cache.get(texto_desordenado)
        if cached is not None:
            return cached

        self._ensure_azure_client()
        if not self.azure_client:
            return {'corregida': None, 'original': texto_desordenado, 'error': 'Azure no configurado'}
        if not self.azure_rate_limiter.is_allowed(client_id):
            return {'corregida': None, 'original': texto_desordenado, 'error': 'Rate limit excedido'}

        try:
            texto_clean = str(texto_desordenado).strip()
            if not texto_clean:
                return {'corregida': None, 'original': texto_desordenado, 'error': 'Texto vacío'}

            prompt = f"""Eres un corrector gramatical especializado en español.
Tu tarea es reorganizar palabras desordenadas en frases gramaticalmente correctas.
Reglas:
- si dice él mujer se refiere a ella, si dice ella hombre se refiere a él.
- reorganiza y agrega artículos/preposiciones necesarios (a, de, el, la, etc.)
- Si hay símbolos como '+' interprétalos como separadores
- Responde SOLO con la frase corregida, sin explicaciones

Corrige esta frase: {texto_clean}"""
            response = self.azure_client.chat.completions.create(
                model=self.azure_deployment,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=100,
                timeout=10
            )
            corregida = response.choices[0].message.content.strip()
            result = {'corregida': corregida, 'original': texto_desordenado, 'error': None}
        except Exception:
            result = {'corregida': None, 'original': texto_desordenado, 'error': 'Error en Azure OpenAI'}

        # Guardar en cache y reiniciar temporizador de inactividad
        self._azure_cache.set(texto_desordenado, result)
        self._reset_idle_timer()
        return result

    # --------------------------------------------------------
    # Búsqueda híbrida (punto de entrada principal)
    # --------------------------------------------------------
    def hybrid_search(self, query, direction='es2yor', top_k=5, client_id='default'):
        results = {
            'query': query,
            'direction': direction,
            'exact_matches': [],
            'compositional': None,
            'morphological': None,
            'alternatives': []
        }

        exact = self.exact_match(query, direction, allow_typos=True, max_edits=1)
        if exact:
            for m in exact[:3]:
                results['exact_matches'].append({
                    'source': m['source'],
                    'target': m['target'],
                    'match_type': m['match_type'],
                    'confidence': 1.0
                })
            return results

        if len(query.split()) > 1:
            comp = self.compositional_translate(query, direction, max_window=3)
            if comp['success']:
                assembled  = ' '.join(c['translation'] for c in comp['chunks'])
                correccion = None
                if direction == 'yor2es' and self.azure_client:
                    correccion = self.corregir_gramatica_azure(assembled, client_id)
                results['compositional'] = {
                    'translation': assembled,
                    'translation_corrected': correccion['corregida'] if correccion else None,
                    'azure_error': correccion['error'] if correccion else None,
                    'chunks': comp['chunks'],
                    'has_unknowns': any(c['match_type'] == 'unknown' for c in comp['chunks'])
                }
                return results

        if direction == 'yor2es' and len(query.split()) == 1:
            morph = self.morphological_split_search(query, direction='yor2es')
            if morph['found']:
                best = morph['splits'][0]
                results['morphological'] = {
                    'translation': best['combined_translation'],
                    'part1': best['part1'],
                    'part2': best['part2'],
                    'part1_translation': best['part1_translation'],
                    'part2_translation': best['part2_translation'],
                    'part1_similarity': best['part1_similarity'],
                    'part1_exact': best['part1_exact'],
                    'score': best['score']
                }
                return results

        fuzzy = self.fuzzy_token_match(query, direction, threshold=0.4)
        emb   = self.embedding_search(query, direction, top_k=top_k * 2, min_similarity=0.5)

        combined = {}
        for item in fuzzy[:10]:
            combined[item['id']] = {
                'source': item['source'],
                'target': item['target'],
                'score': 0.7 + (item['jaccard_score'] * 0.3),
                'match_type': 'fuzzy_token'
            }
        for item in emb:
            if item['id'] not in combined:
                combined[item['id']] = {
                    'source': item['espanol'] if direction == 'es2yor' else item['yoremnokki'],
                    'target': item['yoremnokki'] if direction == 'es2yor' else item['espanol'],
                    'score': item['embedding_score'] * 0.6,
                    'match_type': 'embedding'
                }

        alts = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        alts = [a for a in alts if a['score'] >= 0.5][:top_k]
        results['alternatives'] = alts
        return results


# ------------------------------------------------------------
# Aplicación Flask
# ------------------------------------------------------------
app = Flask(__name__)
translator      = None
request_limiter = RateLimiter(max_calls=100, period=60)

def get_client_ip():
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0]
    return request.remote_addr

@app.before_request
def rate_limit_check():
    client_ip = get_client_ip()
    if not request_limiter.is_allowed(client_ip):
        return jsonify({'error': 'Demasiadas peticiones. Espera un minuto.'}), 429

@app.route('/')
def index():
    return render_template('index.html',
                           total_pairs=translator.total_pairs,
                           embedding_dim=translator.embedding_dim)

@app.route('/translate', methods=['POST'])
def translate():
    data      = request.get_json()
    query     = data.get('query', '').strip()
    direction = data.get('direction', 'es2yor')
    if not query:
        return jsonify({'error': 'Query vacío'}), 400
    if len(query) > 500:
        return jsonify({'error': 'Query demasiado largo'}), 400
    client_ip = get_client_ip()
    results   = translator.hybrid_search(query, direction=direction, top_k=5, client_id=client_ip)
    return jsonify(results)

@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({
        'total_pairs':    translator.total_pairs,
        'embedding_dim':  translator.embedding_dim,
        'model_loaded':   translator._initialized,
        'azure_active':   translator.azure_client is not None,
        'azure_cache_entries': len(translator._azure_cache),
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'yoremnokki-translator'}), 200

# ------------------------------------------------------------
# Inicialización
# ------------------------------------------------------------
possible_paths = [
    'traductor_assets/traductor_yoremnokki.db',
    'traductor_yoremnokki.db',
    './traductor_yoremnokki.db',
    os.getenv("DATABASE_PATH", ""),
]
db_path = None
for path in possible_paths:
    if path and Path(path).exists():
        db_path = path
        break

if not db_path:
    print("❌ Error: No se encuentra la base de datos")
    sys.exit(1)

print("Inicializando traductor (modo lazy, solo metadatos)...")
translator = YoremnokkilTranslator(db_path)
print(f"✓ Base de datos conectada. {translator.total_pairs} pares, dimensión {translator.embedding_dim}")
print("⏳ El modelo y los índices se cargarán en la primera petición real.")

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)