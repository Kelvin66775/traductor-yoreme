from flask import Flask, render_template, request, jsonify
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import unicodedata
import re
import os
from openai import AzureOpenAI
from functools import lru_cache
import time
from threading import Lock
import sys
from collections import OrderedDict
from queue import Queue
import threading
from contextlib import contextmanager
import faiss


class RateLimiter:
    """Rate limiter con límite de IPs en memoria para prevenir memory leaks"""
    
    def __init__(self, max_calls=30, period=60, max_ips=1000):
        self.max_calls = max_calls
        self.period = period
        self.max_ips = max_ips
        self.calls = OrderedDict()  # Mantiene orden de inserción (LRU)
        self.lock = Lock()
        self.last_cleanup = time.time()
    
    def _cleanup_old_ips(self):
        """Elimina IPs inactivas periódicamente"""
        now = time.time()
        
        # Cleanup cada 5 minutos
        if now - self.last_cleanup < 300:
            return
            
        self.last_cleanup = now
        
        # Eliminar IPs con todos los timestamps expirados
        expired_ips = []
        for ip, timestamps in list(self.calls.items()):
            valid_timestamps = [t for t in timestamps if now - t < self.period]
            if not valid_timestamps:
                expired_ips.append(ip)
            else:
                self.calls[ip] = valid_timestamps
        
        for ip in expired_ips:
            del self.calls[ip]
        
        # Si aún hay demasiadas IPs, eliminar las más antiguas (LRU)
        while len(self.calls) > self.max_ips:
            self.calls.popitem(last=False)
        
        print(f"🧹 Cleanup: {len(expired_ips)} IPs expiradas, {len(self.calls)} activas")
    
    def is_allowed(self, client_id):
        with self.lock:
            now = time.time()
            
            # Cleanup periódico
            self._cleanup_old_ips()
            
            if client_id not in self.calls:
                self.calls[client_id] = []
            
            # Limpiar llamadas antiguas de esta IP
            self.calls[client_id] = [
                t for t in self.calls[client_id] if now - t < self.period
            ]
            
            if len(self.calls[client_id]) >= self.max_calls:
                return False
            
            self.calls[client_id].append(now)
            
            # Mover al final (marca como "usado recientemente")
            self.calls.move_to_end(client_id)
            
            return True


class ConnectionPool:
    """Pool de conexiones SQLite thread-safe"""
    
    def __init__(self, db_path, pool_size=3):
        self.db_path = db_path
        self.pool_size = pool_size
        self.pool = Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        
        # Pre-crear conexiones
        for _ in range(pool_size):
            conn = sqlite3.connect(db_path, check_same_thread=True, timeout=30.0)
            conn.row_factory = sqlite3.Row
            self.pool.put(conn)
        
        print(f"✓ Connection pool creado: {pool_size} conexiones")
    
    @contextmanager
    def get_connection(self):
        """Context manager para obtener conexión del pool"""
        conn = self.pool.get(timeout=10)  # Espera máximo 10s
        try:
            yield conn
        finally:
            self.pool.put(conn)
    
    def close_all(self):
        """Cierra todas las conexiones del pool"""
        while not self.pool.empty():
            conn = self.pool.get()
            conn.close()


class YoremnokkilTranslator:
    def __init__(self, db_path):
        self.db_path = db_path
        
        # Connection pool en lugar de conexión única
        self.db_pool = ConnectionPool(db_path, pool_size=3)

        # Configurar caché para el modelo
        model_cache_dir = os.getenv(
            "SENTENCE_TRANSFORMERS_HOME", "./model_cache")
        os.makedirs(model_cache_dir, exist_ok=True)

        print(f"📦 Cargando modelo Sentence Transformers...")
        self.model = SentenceTransformer(
            'all-MiniLM-L6-v2', cache_folder=model_cache_dir)
        print(f"✓ Modelo cargado correctamente")

        # Leer credenciales de Azure desde variables de entorno
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        azure_key = os.getenv("AZURE_OPENAI_KEY", "")
        azure_deployment = os.getenv("AZURE_DEPLOYMENT", "gpt-4.1-mini")
        azure_model = os.getenv("AZURE_MODEL", "gpt-4.1-mini")

        # Rate limiter para Azure API con límite de IPs
        self.azure_rate_limiter = RateLimiter(max_calls=30, period=60, max_ips=500)

        self.azure_client = None
        self.azure_deployment = str(azure_deployment)
        self.azure_model = str(azure_model)

        if azure_endpoint and azure_key:
            try:
                endpoint_clean = str(azure_endpoint).strip().rstrip('/')
                key_clean = str(azure_key).strip()

                # Limpiar endpoint si tiene /openai al final
                if '/openai' in endpoint_clean:
                    parts = endpoint_clean.split('/openai')
                    endpoint_clean = parts[0]

                if not endpoint_clean.endswith('/'):
                    endpoint_clean += '/'

                if not endpoint_clean or not key_clean:
                    print("⚠ Azure OpenAI: credenciales vacías")
                else:
                    self.azure_client = AzureOpenAI(
                        api_key=key_clean,
                        api_version="2024-12-01-preview",
                        azure_endpoint=endpoint_clean
                    )
                    print(f"✓ Azure OpenAI configurado")
                    print(f"  Endpoint: {endpoint_clean}")
                    print(f"  Deployment: {self.azure_deployment}")
            except Exception as e:
                print(f"⚠ Error configurando Azure OpenAI: {e}")
        else:
            print("⚠ Azure OpenAI no configurado (sin corrección gramatical)")

        # 1. Cargar datos en caché
        self.cache_data()

        # 2. Obtener metadatos
        with self.db_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM metadata WHERE key='embedding_dim'")
            self.embedding_dim = int(cursor.fetchone()[0])
            cursor.execute("SELECT value FROM metadata WHERE key='total_pairs'")
            self.total_pairs = int(cursor.fetchone()[0])

        # 3. Construir índices FAISS
        self._build_faiss_indexes()

        print(f"✓ Base de datos cargada: {self.total_pairs} pares")

    def _build_faiss_indexes(self):
        """Construye índices FAISS para búsqueda eficiente por similitud coseno."""
        print("🔧 Construyendo índices FAISS...")
        # Extraer embeddings y normalizarlos
        esp_embs = np.array([item['esp_embedding']
                            for item in self.data_cache])
        yor_embs = np.array([item['yor_embedding']
                            for item in self.data_cache])

        # Normalizar (norma L2) para que el producto interno sea coseno
        faiss.normalize_L2(esp_embs)
        faiss.normalize_L2(yor_embs)

        dim = self.embedding_dim
        self.esp_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.yor_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

        ids = np.arange(len(self.data_cache))
        self.esp_index.add_with_ids(esp_embs, ids)
        self.yor_index.add_with_ids(yor_embs, ids)
        print("✓ Índices FAISS creados")

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

    def normalize_text(self, text):
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join(
            char for char in text if unicodedata.category(char) != 'Mn')
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def cache_data(self):
        with self.db_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, espanol, yoremnokki, esp_embedding, yor_embedding FROM traducciones")

            self.data_cache = []
            for row in cursor.fetchall():
                db_id, espanol, yoremnokki, esp_blob, yor_blob = row
                self.data_cache.append({
                    'id': db_id,
                    'espanol': espanol,
                    'yoremnokki': yoremnokki,
                    'espanol_norm': self.normalize_text(espanol),
                    'yoremnokki_norm': self.normalize_text(yoremnokki),
                    'esp_embedding': np.frombuffer(esp_blob, dtype=np.float32),
                    'yor_embedding': np.frombuffer(yor_blob, dtype=np.float32)
                })

    def exact_match(self, query, direction='es2yor', allow_typos=False, max_edits=1):
        query_lower = query.lower()
        query_norm = self.normalize_text(query)
        source_key = 'espanol' if direction == 'es2yor' else 'yoremnokki'
        source_norm_key = 'espanol_norm' if direction == 'es2yor' else 'yoremnokki_norm'

        matches = []

        for item in self.data_cache:
            if item[source_key].lower() == query_lower:
                matches.append({**item, 'edit_distance': 0,
                               'match_type_original': 'original_acento'})

        if matches:
            return matches

        for item in self.data_cache:
            if item[source_norm_key] == query_norm:
                matches.append({**item, 'edit_distance': 0,
                               'match_type_original': 'normalized'})
            elif allow_typos:
                len_diff = abs(len(item[source_norm_key]) - len(query_norm))
                if len_diff <= max_edits:
                    dist = self.levenshtein_distance(
                        item[source_norm_key], query_norm)
                    if dist <= max_edits:
                        matches.append(
                            {**item, 'edit_distance': dist, 'match_type_original': 'typo'})

        matches_sorted = sorted(matches, key=lambda x: x['edit_distance'])
        return matches_sorted

    def fuzzy_token_match(self, query, direction='es2yor', threshold=0.4):
        query_tokens = set(query.lower().split())
        source_key = 'espanol' if direction == 'es2yor' else 'yoremnokki'

        candidates = []
        for item in self.data_cache:
            source_tokens = set(item[source_key].lower().split())
            intersection = len(query_tokens & source_tokens)
            union = len(query_tokens | source_tokens)
            if union > 0:
                jaccard = intersection / union
                if jaccard >= threshold:
                    candidates.append({**item, 'jaccard_score': jaccard})

        candidates_sorted = sorted(
            candidates, key=lambda x: x['jaccard_score'], reverse=True)
        return candidates_sorted

    def embedding_search(self, query, direction='es2yor', top_k=5):
        query_emb = self.model.encode([query])[0]
        query_emb = query_emb.reshape(1, -1)
        faiss.normalize_L2(query_emb)

        index = self.esp_index if direction == 'es2yor' else self.yor_index
        D, I = index.search(query_emb, top_k * 3)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            item = self.data_cache[int(idx)]
            if dist >= 0.5:
                results.append({**item, 'embedding_score': float(dist)})

        results_sorted = sorted(
            results, key=lambda x: x['embedding_score'], reverse=True)
        return results_sorted[:top_k]

    def compositional_translate(self, query, direction='es2yor', max_window=3):
        words = query.split()
        n = len(words)

        chunks = []
        i = 0
        while i < n:
            best_chunk = None
            best_window = 1

            for w in range(max_window, 0, -1):
                if i + w > n:
                    continue
                window_text = ' '.join(words[i:i+w])
                match = self.exact_match(window_text, direction, allow_typos=True)
                if match:
                    target_key = 'yoremnokki' if direction == 'es2yor' else 'espanol'
                    best_chunk = {
                        'original': window_text,
                        'translation': match[0][target_key],
                        'match_type': 'exact',
                        'window_size': w
                    }
                    best_window = w
                    break

            if not best_chunk:
                fuzzy_matches = self.fuzzy_token_match(
                    words[i], direction, threshold=0.5)
                if fuzzy_matches:
                    target_key = 'yoremnokki' if direction == 'es2yor' else 'espanol'
                    best_chunk = {
                        'original': words[i],
                        'translation': fuzzy_matches[0][target_key],
                        'match_type': 'fuzzy',
                        'window_size': 1
                    }
                else:
                    best_chunk = {
                        'original': words[i],
                        'translation': f"[?{words[i]}?]",
                        'match_type': 'unknown',
                        'window_size': 1
                    }

            chunks.append(best_chunk)
            i += best_window

        return {'success': True, 'chunks': chunks}

    def morphological_split_search(self, word, direction='yor2es', min_ratio=0.45, fuzzy_threshold=0.90):
        results = []
        n = len(word)

        for i in range(1, n):
            part1 = word[:i]
            part2 = word[i:]

            if len(part1) < 2 or len(part2) < 2:
                continue

            exact_p1 = self.exact_match(part1, direction, allow_typos=False)
            exact_p2 = self.exact_match(part2, direction, allow_typos=False)

            if exact_p1 and exact_p2:
                target_key = 'espanol' if direction == 'yor2es' else 'yoremnokki'
                p1_trans = exact_p1[0][target_key]
                p2_trans = exact_p2[0][target_key]

                results.append({
                    'part1': part1,
                    'part2': part2,
                    'part1_translation': p1_trans,
                    'part2_translation': p2_trans,
                    'combined_translation': f"{p1_trans} {p2_trans}",
                    'part1_similarity': 1.0,
                    'part1_exact': True,
                    'score': 1.0
                })
                continue

            if not exact_p1:
                fuzzy_p1 = self.embedding_search(part1, direction, top_k=1)
                if fuzzy_p1 and fuzzy_p1[0]['embedding_score'] >= fuzzy_threshold:
                    exact_p2 = self.exact_match(part2, direction, allow_typos=False)
                    if exact_p2:
                        target_key = 'espanol' if direction == 'yor2es' else 'yoremnokki'
                        p1_trans = fuzzy_p1[0][target_key]
                        p2_trans = exact_p2[0][target_key]

                        results.append({
                            'part1': part1,
                            'part2': part2,
                            'part1_translation': p1_trans,
                            'part2_translation': p2_trans,
                            'combined_translation': f"{p1_trans} {p2_trans}",
                            'part1_similarity': fuzzy_p1[0]['embedding_score'],
                            'part1_exact': False,
                            'score': fuzzy_p1[0]['embedding_score']
                        })

        if results:
            results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
            return {'found': True, 'splits': results_sorted}
        else:
            return {'found': False, 'splits': []}

    def corregir_gramatica_azure(self, texto_desordenado, client_id='default'):
        if not self.azure_client:
            return {'corregida': None, 'original': texto_desordenado, 'error': 'Azure no configurado'}

        if not self.azure_rate_limiter.is_allowed(client_id):
            return {
                'corregida': None,
                'original': texto_desordenado,
                'error': 'Rate limit excedido para Azure API'
            }

        try:
            prompt = f"""Eres un corrector gramatical del idioma español. Tu tarea es corregir SOLO la gramática y orden de las palabras.

REGLAS ESTRICTAS:
- NO traduzcas palabras entre idiomas
- NO reemplaces palabras indígenas con equivalentes en español
- SOLO corrige el orden de las palabras y la gramática
- Si una frase ya está gramaticalmente correcta, devuélvela sin cambios
- Mantén TODAS las palabras originales

Texto a corregir: "{texto_desordenado}"

Responde ÚNICAMENTE con la versión corregida, sin explicaciones adicionales."""

            response = self.azure_client.chat.completions.create(
                model=self.azure_deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )

            texto_corregido = response.choices[0].message.content.strip()
            return {
                'corregida': texto_corregido,
                'original': texto_desordenado,
                'error': None
            }

        except Exception as e:
            error_msg = str(e)
            if 'DeploymentNotFound' in error_msg or '404' in error_msg:
                return {
                    'corregida': None,
                    'original': texto_desordenado,
                    'error': f"Deployment '{self.azure_deployment}' no encontrado"
                }

            return {
                'corregida': None,
                'original': texto_desordenado,
                'error': 'Error en Azure OpenAI'
            }

    def hybrid_search(self, query, direction='es2yor', top_k=5, client_id='default'):
        results = {
            'query': query,
            'direction': direction,
            'exact_matches': [],
            'compositional': None,
            'morphological': None,
            'alternatives': []
        }

        exact_matches = self.exact_match(
            query, direction, allow_typos=True, max_edits=1)

        if exact_matches:
            for match in exact_matches[:3]:
                target_key = 'yoremnokki' if direction == 'es2yor' else 'espanol'
                results['exact_matches'].append({
                    'source': match['espanol'] if direction == 'es2yor' else match['yoremnokki'],
                    'target': match[target_key],
                    'match_type': match.get('match_type_original', 'exact'),
                    'confidence': 1.0
                })
            return results

        if len(query.split()) > 1:
            comp_result = self.compositional_translate(
                query, direction, max_window=3)
            if comp_result['success']:
                assembled = ' '.join([c['translation']
                                     for c in comp_result['chunks']])
                correccion = None
                if direction == 'yor2es' and self.azure_client:
                    correccion = self.corregir_gramatica_azure(
                        assembled, client_id)

                results['compositional'] = {
                    'translation': assembled,
                    'translation_corrected': correccion['corregida'] if correccion and correccion['corregida'] else None,
                    'azure_error': correccion['error'] if correccion else None,
                    'chunks': comp_result['chunks'],
                    'has_unknowns': any(c['match_type'] == 'unknown' for c in comp_result['chunks'])
                }
                return results

        if direction == 'yor2es' and len(query.split()) == 1:
            morph_result = self.morphological_split_search(
                query, direction, min_ratio=0.45, fuzzy_threshold=0.90)

            if morph_result['found']:
                best_split = morph_result['splits'][0]
                results['morphological'] = {
                    'translation': best_split['combined_translation'],
                    'part1': best_split['part1'],
                    'part2': best_split['part2'],
                    'part1_translation': best_split['part1_translation'],
                    'part2_translation': best_split['part2_translation'],
                    'part1_similarity': best_split['part1_similarity'],
                    'part1_exact': best_split['part1_exact'],
                    'score': best_split['score']
                }
                return results

        fuzzy_matches = self.fuzzy_token_match(query, direction, threshold=0.4)
        embedding_matches = self.embedding_search(
            query, direction, top_k=top_k*2)

        combined = {}

        for item in fuzzy_matches[:10]:
            key = item['id']
            combined[key] = {
                'id': key,
                'espanol': item['espanol'],
                'yoremnokki': item['yoremnokki'],
                'score': 0.7 + (item['jaccard_score'] * 0.3),
                'match_type': 'fuzzy_token'
            }

        for item in embedding_matches:
            key = item['id']
            if key not in combined:
                combined[key] = {
                    'id': key,
                    'espanol': item['espanol'],
                    'yoremnokki': item['yoremnokki'],
                    'score': item['embedding_score'] * 0.6,
                    'match_type': 'embedding'
                }

        alternatives = sorted(
            combined.values(), key=lambda x: x['score'], reverse=True)
        alternatives = [
            alt for alt in alternatives if alt['score'] >= 0.5][:top_k]

        for alt in alternatives:
            results['alternatives'].append({
                'source': alt['espanol'] if direction == 'es2yor' else alt['yoremnokki'],
                'target': alt['yoremnokki'] if direction == 'es2yor' else alt['espanol'],
                'score': alt['score'],
                'match_type': alt['match_type']
            })

        return results


# Inicializar Flask
app = Flask(__name__)

translator = None

# Rate limiter global con límite de IPs
request_limiter = RateLimiter(max_calls=100, period=60, max_ips=2000)


def get_client_ip():
    """Obtener IP del cliente (funciona detrás de proxies)"""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0]
    return request.remote_addr


@app.before_request
def rate_limit_check():
    """Rate limiting global por IP"""
    client_ip = get_client_ip()
    if not request_limiter.is_allowed(client_ip):
        return jsonify({'error': 'Demasiadas peticiones. Espera un minuto.'}), 429


@app.route('/')
def index():
    return render_template('index.html', total_pairs=translator.total_pairs, embedding_dim=translator.embedding_dim)


@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    query = data.get('query', '').strip()
    direction = data.get('direction', 'es2yor')

    if not query:
        return jsonify({'error': 'Query vacío'}), 400

    if len(query) > 500:
        return jsonify({'error': 'Query demasiado largo (máx 500 caracteres)'}), 400

    client_ip = get_client_ip()
    results = translator.hybrid_search(
        query, direction=direction, top_k=5, client_id=client_ip)
    return jsonify(results)


@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({
        'total_pairs': translator.total_pairs,
        'embedding_dim': translator.embedding_dim
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint para Railway"""
    return jsonify({'status': 'ok', 'service': 'yoremnokki-translator'}), 200


@app.route('/debug/memory', methods=['GET'])
def memory_usage():
    """Endpoint de debug para monitorear uso de memoria"""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        
        return jsonify({
            'rss_mb': round(mem_info.rss / 1024 / 1024, 2),
            'vms_mb': round(mem_info.vms / 1024 / 1024, 2),
            'percent': round(process.memory_percent(), 2),
            'rate_limiter_ips': len(request_limiter.calls),
            'azure_limiter_ips': len(translator.azure_rate_limiter.calls) if translator else 0
        }), 200
    except ImportError:
        return jsonify({'error': 'psutil no instalado'}), 500


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
    print(f"   Buscado en: {possible_paths}")
    sys.exit(1)

print("Inicializando traductor Yoremnokki...")
translator = YoremnokkilTranslator(db_path)

print("\n" + "="*60)
print("✓ Servidor listo (translator cargado)")
print("="*60)
print(f"\n📊 Total de pares: {translator.total_pairs}")
print(f"🧠 Modelo: all-MiniLM-L6-v2 ({translator.embedding_dim}D)")

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)