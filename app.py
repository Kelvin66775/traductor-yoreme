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
from pathlib import Path

class RateLimiter:
    """Rate limiter para prevenir abuso de la API"""
    def __init__(self, max_calls=30, period=60):
        self.max_calls = max_calls
        self.period = period
        self.calls = {}
        self.lock = Lock()
    
    def is_allowed(self, client_id):
        with self.lock:
            now = time.time()
            if client_id not in self.calls:
                self.calls[client_id] = []
            
            # Limpiar llamadas antiguas
            self.calls[client_id] = [t for t in self.calls[client_id] if now - t < self.period]
            
            if len(self.calls[client_id]) >= self.max_calls:
                return False
            
            self.calls[client_id].append(now)
            return True

class YoremnokkilTranslator:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # Configurar caché para el modelo
        model_cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME", "./model_cache")
        os.makedirs(model_cache_dir, exist_ok=True)
        
        print(f"📦 Cargando modelo Sentence Transformers...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=model_cache_dir)
        print(f"✓ Modelo cargado correctamente")

        # Leer credenciales de Azure desde variables de entorno
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        azure_key = os.getenv("AZURE_OPENAI_KEY", "")
        azure_deployment = os.getenv("AZURE_DEPLOYMENT", "gpt-4.1-mini")
        azure_model = os.getenv("AZURE_MODEL", "gpt-4.1-mini")
        
        # Rate limiter para Azure API
        self.azure_rate_limiter = RateLimiter(max_calls=30, period=60)

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
        
        self.cache_data()
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key='embedding_dim'")
        self.embedding_dim = int(cursor.fetchone()[0])
        
        cursor.execute("SELECT value FROM metadata WHERE key='total_pairs'")
        self.total_pairs = int(cursor.fetchone()[0])
        
        print(f"✓ Base de datos cargada: {self.total_pairs} pares")
    
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
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def cache_data(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, espanol, yoremnokki, esp_embedding, yor_embedding FROM traducciones")
        
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
                matches.append({**item, 'edit_distance': 0, 'match_type_original': 'original_acento'})
        
        if matches:
            return matches
        
        for item in self.data_cache:
            if item[source_norm_key] == query_norm:
                matches.append({**item, 'edit_distance': 0, 'match_type_original': 'normalized'})
            elif allow_typos:
                len_diff = abs(len(item[source_norm_key]) - len(query_norm))
                if len_diff <= max_edits:
                    dist = self.levenshtein_distance(item[source_norm_key], query_norm)
                    if dist <= max_edits:
                        matches.append({**item, 'edit_distance': dist, 'match_type_original': 'typo'})
        
        matches.sort(key=lambda x: x.get('edit_distance', 0))
        return matches
    
    def morphological_split_search(self, word, direction='es2yor', min_ratio=0.45, fuzzy_threshold=0.90):
        if direction != 'yor2es':
            return {'found': False, 'splits': []}
        
        word_norm = self.normalize_text(word)
        word_len = len(word_norm)
        
        if word_len < 4:
            return {'found': False, 'splits': []}
        
        valid_splits = []
        source_norm_key = 'yoremnokki_norm'
        target_key = 'espanol'
        
        for split_pos in range(int(word_len * min_ratio), int(word_len * 0.95) + 1):
            part1_norm = word_norm[:split_pos]
            part2_norm = word_norm[split_pos:]
            
            if len(part1_norm) < 1 or len(part2_norm) < 1:
                continue
            
            part2_matches = []
            for item in self.data_cache:
                if item[source_norm_key] == part2_norm:
                    part2_matches.append(item)
            
            if not part2_matches:
                continue
            
            part1_candidates = []
            for item in self.data_cache:
                item_norm = item[source_norm_key]
                
                if item_norm == part1_norm:
                    part1_candidates.append({
                        'item': item,
                        'similarity': 1.0,
                        'is_exact': True
                    })
                else:
                    max_len = max(len(item_norm), len(part1_norm))
                    if max_len == 0:
                        continue
                    
                    lev_dist = self.levenshtein_distance(item_norm, part1_norm)
                    similarity = 1.0 - (lev_dist / max_len)
                    
                    if similarity >= fuzzy_threshold:
                        part1_candidates.append({
                            'item': item,
                            'similarity': similarity,
                            'is_exact': False
                        })
            
            for p1_cand in part1_candidates:
                for p2_match in part2_matches:
                    combined_translation = f"{p1_cand['item'][target_key]} {p2_match[target_key]}"
                
                    score = p1_cand['similarity'] * 0.7 + 0.3
                    
                    valid_splits.append({
                        'part1': p1_cand['item'][source_norm_key],
                        'part2': p2_match[source_norm_key],
                        'part1_translation': p1_cand['item'][target_key],
                        'part2_translation': p2_match[target_key],
                        'combined_translation': combined_translation,
                        'part1_similarity': p1_cand['similarity'],
                        'part1_exact': p1_cand['is_exact'],
                        'part2_exact': True,
                        'score': score,
                        'split_position': split_pos
                    })
        
        valid_splits.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'found': len(valid_splits) > 0,
            'splits': valid_splits[:3]
        }
    
    def fuzzy_token_match(self, query, direction='es2yor', threshold=0.3):
        query_norm = self.normalize_text(query)
        query_tokens = set(query_norm.split())
        source_norm_key = 'espanol_norm' if direction == 'es2yor' else 'yoremnokki_norm'
        
        results = []
        for item in self.data_cache:
            item_tokens = set(item[source_norm_key].split())
            intersection = query_tokens & item_tokens
            union = query_tokens | item_tokens
            jaccard = len(intersection) / len(union) if union else 0
            
            if jaccard >= threshold:
                results.append({
                    **item,
                    'jaccard_score': jaccard
                })
        
        results.sort(key=lambda x: x['jaccard_score'], reverse=True)
        return results
    
    def embedding_search(self, query, direction='es2yor', top_k=10):
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        embedding_key = 'esp_embedding' if direction == 'es2yor' else 'yor_embedding'
        
        similarities = []
        for item in self.data_cache:
            db_embedding = item[embedding_key]
            similarity = np.dot(query_embedding, db_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding)
            )
            similarities.append({
                **item,
                'embedding_score': float(similarity)
            })
        
        similarities.sort(key=lambda x: x['embedding_score'], reverse=True)
        return similarities[:top_k]
    
    def ngram_windows(self, tokens, max_n=3):
        windows = []
        n_tokens = len(tokens)
        
        for n in range(min(max_n, n_tokens), 0, -1):
            for i in range(n_tokens - n + 1):
                chunk_tokens = tokens[i:i+n]
                chunk_text = ' '.join(chunk_tokens)
                windows.append((i, i+n, chunk_text))
        
        return windows
    
    def compositional_translate(self, query, direction='es2yor', max_window=3):
        original_tokens = query.split()
        tokens_norm = self.normalize_text(query).split()
        n = len(tokens_norm)
        
        if n == 0:
            return {'success': False, 'chunks': []}
        
        covered = [False] * n
        chunks = []
        
        windows = self.ngram_windows(tokens_norm, max_n=max_window)
        
        for start_idx, end_idx, chunk_text in windows:
            if any(covered[start_idx:end_idx]):
                continue
            
            chunk_original = ' '.join(original_tokens[start_idx:end_idx])
            
            matches = self.exact_match(chunk_original, direction, allow_typos=True, max_edits=1)
            
            if matches:
                best = matches[0]
                target_key = 'yoremnokki' if direction == 'es2yor' else 'espanol'
                
                chunks.append({
                    'start': start_idx,
                    'end': end_idx,
                    'source': chunk_original,
                    'translation': best[target_key],
                    'match_type': best.get('match_type_original', 'exact'),
                    'confidence': 1.0 - (best.get('edit_distance', 0) * 0.1)
                })
                
                for i in range(start_idx, end_idx):
                    covered[i] = True
        
        for i in range(n):
            if not covered[i]:
                token_original = original_tokens[i]
                token_norm = tokens_norm[i]
                
                if direction == 'yor2es':
                    morph_result = self.morphological_split_search(
                        token_original, 
                        direction='yor2es', 
                        min_ratio=0.45, 
                        fuzzy_threshold=0.90
                    )
                    
                    if morph_result['found']:
                        best_split = morph_result['splits'][0]
                        chunks.append({
                            'start': i,
                            'end': i + 1,
                            'source': token_original,
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
                    'start': i,
                    'end': i + 1,
                    'source': token_original,
                    'translation': f"[{token_original}]",
                    'match_type': 'unknown',
                    'confidence': 0.0
                })
        
        chunks.sort(key=lambda x: x['start'])
        
        return {
            'success': any(c['match_type'] != 'unknown' for c in chunks),
            'chunks': chunks
        }
    
    @lru_cache(maxsize=200)
    def corregir_gramatica_azure(self, texto_desordenado, client_id='default'):
        """Corrección gramatical con caché y rate limiting"""
        if not self.azure_client:
            return {
                'corregida': None,
                'original': texto_desordenado,
                'error': 'Azure OpenAI no configurado'
            }
        
        # Rate limiting
        if not self.azure_rate_limiter.is_allowed(client_id):
            return {
                'corregida': None,
                'original': texto_desordenado,
                'error': 'Rate limit excedido. Intenta en un minuto.'
            }
        
        try:
            texto_clean = str(texto_desordenado).strip()
            
            if not texto_clean:
                return {
                    'corregida': None,
                    'original': texto_desordenado,
                    'error': 'Texto vacío'
                }

            user_prompt = f"""Eres un corrector gramatical especializado en español. 
Tu tarea es reorganizar palabras desordenadas en frases gramaticalmente correctas, preservando el significado exacto.

Reglas:
- si dice él mujer se refiere a ella, si dice ella hombre se refiere a él.
- reorganiza y agrega artículos/preposiciones necesarios (a, de, el, la, etc.)
- Si hay símbolos como '+' interprétalos como separadores
- Responde SOLO con la frase corregida, sin explicaciones

Corrige esta frase: {texto_clean}"""
            
            response = self.azure_client.chat.completions.create(
                model=str(self.azure_deployment),
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=100,
                timeout=10
            )
            
            corregida = response.choices[0].message.content.strip()
            
            return {
                'corregida': corregida,
                'original': texto_desordenado,
                'error': None
            }
            
        except Exception as e:
            error_msg = str(e)
            
            if "404" in error_msg:
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
        
        alternatives = sorted(combined.values(), key=lambda x: x['score'], reverse=True)[:top_k]
        
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

# Rate limiter global para requests HTTP
request_limiter = RateLimiter(max_calls=100, period=60)

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

possible_paths = [
    'traductor_assets/traductor_yoremnokki.db',
    'traductor_yoremnokki.db',
    './traductor_yoremnokki.db',
    os.getenv("DATABASE_PATH", ""),   # opcional: permitir ruta por variable de entorno
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
# ============================================================

# Mantén el bloque if __name__ solo para desarrollo local
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)