# gunicorn_config.py - Configuración optimizada para Railway
import multiprocessing
import os

# ============================================================
# CRITICAL: Solo 1 worker para evitar duplicar modelo en memoria
# ============================================================
workers = 1  # NUNCA más de 1 con ML models grandes

# Usar threads en lugar de workers adicionales
# Railway tiene 1 vCPU, así que 2-4 threads es óptimo
threads = 4

# Worker class: gevent/eventlet para I/O async (mejor para requests concurrentes)
# Requiere: pip install gevent
worker_class = 'gevent'  # Alternativa: 'gthread' (no requiere instalación)

# Memory limits
max_requests = 1000  # Reciclar worker después de 1000 requests (previene leaks)
max_requests_jitter = 50  # Añade aleatoriedad al reciclaje

# Timeouts
timeout = 120  # 2 minutos para requests lentos (Azure OpenAI puede tardar)
graceful_timeout = 60  # Tiempo para terminar requests en curso antes de kill

# Preload para cargar modelo ANTES de fork (no aplica con 1 worker, pero por si acaso)
preload_app = True

# Logging
accesslog = '-'  # stdout
errorlog = '-'   # stderr
loglevel = 'info'

# Bind
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"

# Worker lifecycle hooks para debugging
def on_starting(server):
    """Llamado al iniciar Gunicorn master"""
    print("=" * 60)
    print("🚀 Gunicorn Master iniciando")
    print(f"   Workers: {workers}")
    print(f"   Threads: {threads}")
    print(f"   Worker class: {worker_class}")
    print("=" * 60)

def post_fork(server, worker):
    """Llamado DESPUÉS de fork (en cada worker)"""
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"🔧 Worker {worker.pid} forked - RSS: {mem_info.rss / 1024 / 1024:.1f}MB")

def worker_int(worker):
    """Llamado cuando worker recibe SIGINT/SIGTERM"""
    print(f"⚠️  Worker {worker.pid} recibió señal de terminación")

def worker_abort(worker):
    """Llamado cuando worker es abortado (OOM, crash)"""
    print(f"💀 Worker {worker.pid} ABORTADO (posible OOM)")
