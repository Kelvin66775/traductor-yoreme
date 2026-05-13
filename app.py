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
import gc


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
        
        if expired_ips:
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
        print("\n" + "="*60)
        print("🚀 Inicializando traductor Yoremnokki (optimizado)")
        print("="*60)
        
        # Log memoria inicial
        try:
            import psutil
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            print(f"📊 Memoria inicial: {mem_before:.1f}MB")
        except ImportError:
            mem_before = None
        
        self.db_path = db_path
        
        # Connection pool en lugar de conexión única
        self.db_pool = ConnectionPool(db_path, pool_size=3)

        # ============================================================
        # OPTIMIZACIÓN 1: Configurar caché del modelo
        # ============================================================
        model_cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME", "./model_cache")
        os.makedirs(model_cache_dir, exist_ok=True)

        print(f"📦 Cargando modelo Sentence Transformers...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=model_cache_dir)
        print(f"✓ Modelo cargado correctamente")
        
        # Log memoria después de modelo
        if mem_before:
            mem_after_model = process.memory_info().rss / 1024 / 1024
            print(f"   Memoria después de modelo: {mem_after_model:.1f}MB (+{mem_after_model - mem_before:.1f}MB)")

        # ============================================================
        # OPTIMIZACIÓN 2: Azure con rate limiter reducido
        # ============================================================
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        azure_key = os.getenv("AZURE_OPENAI_KEY", "")
        azure_deployment = os.getenv("AZURE_DEPLOYMENT", "gpt-4.1-mini")
        azure_model = os.getenv("AZURE_MODEL", "gpt-4.1-mini")

        # Rate limiter más agresivo para Azure
        self.azure_rate_limiter = RateLimiter(max_calls=20, period=60, max_ips=200)

        self.azure_client = None
        self.azure_deployment = str(azure_deployment)
        self.azure_model = str(azure_model)

        if azure_endpoint and azure_key:
            try:
                endpoint_clean = str(azure_endpoint).strip().rstrip('/')
                key_clean = str(azure_key).strip()

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

        # ============================================================
        # OPTIMIZACIÓN 3: Cargar datos con memory profiling
        # ============================================================
        print(f"📥 Cargando base de datos...")
        self.cache_data()
        
        if mem_before:
            mem_after_cache = process.memory_info().rss / 1024 / 1024
            print(f"   Memoria después de cache: {mem_after_cache:.1f}MB (+{mem_after_cache - mem_after_model:.1f}MB)")

        # 2. Obtener metadatos
        with self.db_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM metadata WHERE key='embedding_dim'")
            self.embedding_dim = int(cursor.fetchone()[0])
            cursor.execute("SELECT value FROM metadata WHERE key='total_pairs'")
            self.total_pairs = int(cursor.fetchone()[0])

        # ============================================================
        # OPTIMIZACIÓN 4: Construir FAISS con garbage collection
        # ============================================================
        print(f"🔧 Construyendo índices FAISS...")
        self._build_faiss_indexes()
        
        # Force garbage collection después de construcción pesada
        gc.collect()
        
        if mem_before:
            mem_final = process.memory_info().rss / 1024 / 1024
            print(f"   Memoria final: {mem_final:.1f}MB (+{mem_final - mem_after_cache:.1f}MB)")
            print(f"   Memoria total usada: {mem_final:.1f}MB")

        print(f"\n✓ Base de datos cargada: {self.total_pairs} pares")
        print(f"✓ Inicialización completa")
        print("="*60 + "\n")

    def _build_faiss_indexes(self):
        """Construye índices FAISS para búsqueda eficiente por similitud coseno."""
        # Extraer embeddings y normalizarlos
        esp_embs = np.array([item['esp_embedding'] for item in self.data_cache], dtype=np.float32)
        yor_embs = np.array([item['yor_embedding'] for item in self.data_cache], dtype=np.float32)

        # Normalizar (norma L2) para que el producto interno sea coseno
        faiss.normalize_L2(esp_embs)
        faiss.normalize_L2(yor_embs)

        dim = self.embedding_dim
        self.esp_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.yor_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

        ids = np.arange(len(self.data_cache), dtype=np.int64)
        self.esp_index.add_with_ids(esp_embs, ids)
        self.yor_index.add_with_ids(yor_embs, ids)
        
        # Liberar arrays temporales
        del esp_embs, yor_embs, ids
        
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

    def remove_accents(self, input_str):
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        only_ascii = nfkd_form.encode('ASCII', 'ignore').decode('ASCII')
        return only_ascii.lower()

    def normalize_text(self, text):
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[¿?¡!]', '', text)
        text = self.remove_accents(text)
        return text

    def cache_data(self):
        """Carga todos los pares en memoria para búsqueda rápida."""
        with self.db_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, espanol, yoremnokki, esp_embedding, yor_embedding FROM translations")
            rows = cursor.fetchall()
            
            # Procesar datos con numpy para eficiencia
            self.data_cache = []
            for row in rows:
                esp_emb = np.frombuffer(row['esp_embedding'], dtype=np.float32)
                yor_emb = np.frombuffer(row['yor_embedding'], dtype=np.float32)
                
                self.data_cache.append({
                    'id': row['id'],
                    'espanol': row['espanol'],
                    'yoremnokki': row['yoremnokki'],
                    'esp_embedding': esp_emb,
                    'yor_embedding': yor_emb,
                    'espanol_norm': self.normalize_text(row['espanol']),
                    'yoremnokki_norm': self.normalize_text(row['yoremnokki'])
                })

    @lru_cache(maxsize=2000)
    def exact_match(self, query, direction='es2yor', allow_typos=False, max_edits=1):
        """Busca coincidencias exactas o con ligeros typos en la base de datos."""
        query_norm = self.normalize_text(query)
        
        if direction == 'es2yor':
            source_field = 'espanol_norm'
            matches = [item for item in self.data_cache if item[source_field] == query_norm]
        else:
            source_field = 'yoremnokki_norm'
            matches = [item for item in self.data_cache if item[source_field] == query_norm]
        
        if matches:
            return matches
        
        if allow_typos:
            fuzzy_matches = []
            for item in self.data_cache:
                dist = self.levenshtein_distance(query_norm, item[source_field])
                if dist <= max_edits:
                    fuzzy_matches.append({
                        **item,
                        'match_type_original': f'typo_{dist}edit',
                        'edit_distance': dist
                    })
            
            if fuzzy_matches:
                fuzzy_matches.sort(key=lambda x: x['edit_distance'])
                return fuzzy_matches
        
        return []

    def embedding_search(self, query, direction='es2yor', top_k=5):
        """Búsqueda por similitud usando embeddings con FAISS."""
        query_embedding = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(query_embedding)
        
        index = self.esp_index if direction == 'es2yor' else self.yor_index
        
        distances, indices = index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx != -1:
                item = self.data_cache[idx]
                results.append({
                    'id': item['id'],
                    'espanol': item['espanol'],
                    'yoremnokki': item['yoremnokki'],
                    'embedding_score': float(score)
                })
        
        return results

    def jaccard_similarity(self, str1, str2):
        """Calcula similitud de Jaccard entre dos cadenas (a nivel de tokens)."""
        tokens1 = set(str1.split())
        tokens2 = set(str2.split())
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        if not union:
            return 0.0
        return len(intersection) / len(union)

    def fuzzy_token_match(self, query, direction='es2yor', threshold=0.4):
        """Busca coincidencias basadas en tokens compartidos (Jaccard)."""
        query_norm = self.normalize_text(query)
        
        results = []
        for item in self.data_cache:
            source_field = 'espanol_norm' if direction == 'es2yor' else 'yoremnokki_norm'
            score = self.jaccard_similarity(query_norm, item[source_field])
            if score >= threshold:
                results.append({
                    'id': item['id'],
                    'espanol': item['espanol'],
                    'yoremnokki': item['yoremnokki'],
                    'jaccard_score': score
                })
        
        results.sort(key=lambda x: x['jaccard_score'], reverse=True)
        return results

    def compositional_translate(self, query, direction='es2yor', max_window=3):
        """Traduce texto dividiéndolo en ventanas de N palabras."""
        tokens = query.strip().split()
        chunks = []
        i = 0
        
        while i < len(tokens):
            found = False
            for window_size in range(min(max_window, len(tokens) - i), 0, -1):
                window_text = ' '.join(tokens[i:i + window_size])
                matches = self.exact_match(window_text, direction, allow_typos=True, max_edits=1)
                
                if matches:
                    best_match = matches[0]
                    target_key = 'yoremnokki' if direction == 'es2yor' else 'espanol'
                    
                    chunks.append({
                        'source': window_text,
                        'translation': best_match[target_key],
                        'window_size': window_size,
                        'match_type': best_match.get('match_type_original', 'exact')
                    })
                    i += window_size
                    found = True
                    break
            
            if not found:
                chunks.append({
                    'source': tokens[i],
                    'translation': f"[{tokens[i]}]",
                    'window_size': 1,
                    'match_type': 'unknown'
                })
                i += 1
        
        return {
            'success': len(chunks) > 0,
            'chunks': chunks
        }

    def morphological_split_search(self, word, direction='yor2es', min_ratio=0.45, fuzzy_threshold=0.90):
        """Divide palabras compuestas y busca coincidencias morfológicas."""
        if direction != 'yor2es':
            return {'found': False, 'splits': []}
        
        results = []
        word_norm = self.normalize_text(word)
        
        for split_pos in range(3, len(word_norm) - 2):
            part1 = word_norm[:split_pos]
            part2 = word_norm[split_pos:]
            
            if len(part2) / len(word_norm) < min_ratio:
                continue
            
            match1 = self.exact_match(part1, direction='yor2es', allow_typos=False)
            is_exact1 = len(match1) > 0
            
            if not is_exact1:
                embedding_results = self.embedding_search(part1, direction='yor2es', top_k=1)
                if embedding_results and embedding_results[0]['embedding_score'] >= fuzzy_threshold:
                    match1 = [embedding_results[0]]
            
            if match1:
                match2 = self.exact_match(part2, direction='yor2es', allow_typos=False)
                
                if match2:
                    combined = f"{match1[0]['espanol']} {match2[0]['espanol']}"
                    score = 1.0 if is_exact1 else embedding_results[0]['embedding_score']
                    
                    results.append({
                        'part1': part1,
                        'part2': part2,
                        'part1_translation': match1[0]['espanol'],
                        'part2_translation': match2[0]['espanol'],
                        'combined_translation': combined,
                        'score': score,
                        'part1_exact': is_exact1,
                        'part1_similarity': score
                    })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return {'found': len(results) > 0, 'splits': results}

    def corregir_gramatica_azure(self, texto_desordenado, client_id='default'):
        """Corrige gramática usando Azure OpenAI con rate limiting."""
        if not self.azure_client:
            return {'corregida': None, 'original': texto_desordenado, 'error': 'Azure no configurado'}
        
        if not self.azure_rate_limiter.is_allowed(client_id):
            return {
                'corregida': None,
                'original': texto_desordenado,
                'error': 'Rate limit excedido para corrección gramatical'
            }
        
        try:
            system_prompt = (
                "Eres un corrector gramatical de español. "
                "Corrige únicamente errores gramaticales (concordancia, artículos, preposiciones) "
                "sin cambiar el significado ni las palabras clave. "
                "Si el texto ya está correcto, devuélvelo sin cambios."
            )
            
            response = self.azure_client.chat.completions.create(
                model=self.azure_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Corrige: {texto_desordenado}"}
                ],
                temperature=0.3,
                max_tokens=100,
                timeout=10
            )
            
            texto_corregido = response.choices[0].message.content.strip()
            
            return {
                'corregida': texto_corregido,
                'original': texto_desordenado,
                'error': None
            }
        except Exception as e:
            print(f"Error en Azure OpenAI: {e}")
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

        exact_matches = self.exact_match(query, direction, allow_typos=True, max_edits=1)

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
            comp_result = self.compositional_translate(query, direction, max_window=3)
            if comp_result['success']:
                assembled = ' '.join([c['translation'] for c in comp_result['chunks']])
                correccion = None
                if direction == 'yor2es' and self.azure_client:
                    correccion = self.corregir_gramatica_azure(assembled, client_id)

                results['compositional'] = {
                    'translation': assembled,
                    'translation_corrected': correccion['corregida'] if correccion and correccion['corregida'] else None,
                    'azure_error': correccion['error'] if correccion else None,
                    'chunks': comp_result['chunks'],
                    'has_unknowns': any(c['match_type'] == 'unknown' for c in comp_result['chunks'])
                }
                return results

        if direction == 'yor2es' and len(query.split()) == 1:
            morph_result = self.morphological_split_search(query, direction, min_ratio=0.45, fuzzy_threshold=0.90)

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
        embedding_matches = self.embedding_search(query, direction, top_k=top_k*2)

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

        alternatives = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        alternatives = [alt for alt in alternatives if alt['score'] >= 0.5][:top_k]

        for alt in alternatives:
            results['alternatives'].append({
                'source': alt['espanol'] if direction == 'es2yor' else alt['yoremnokki'],
                'target': alt['yoremnokki'] if direction == 'es2yor' else alt['espanol'],
                'score': alt['score'],
                'match_type': alt['match_type']
            })

        return results


# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)

# Global translator (cargado una sola vez)
translator = None

# Rate limiter global con límite de IPs reducido
request_limiter = RateLimiter(max_calls=100, period=60, max_ips=1000)


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
    results = translator.hybrid_search(query, direction=direction, top_k=5, client_id=client_ip)
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


# ============================================================
# INICIALIZACIÓN (solo una vez)
# ============================================================
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

# Cargar traductor (solo una vez al inicio)
translator = YoremnokkilTranslator(db_path)

print("\n" + "="*60)
print("✅ Servidor listo para recibir requests")
print("="*60)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)