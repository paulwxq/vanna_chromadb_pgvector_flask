# Qwen模型配置
QWEN_CONFIG = {
    "api_key": "sk-db68e37f00974031935395315bfe07f0",
    "model": "qwen-plus",
    "allow_llm_to_see_data": True,
    "temperature": 0.6,
    "n_results_sql": 5,
    "n_results_documentation": 5,
    "n_results_ddl": 5,
    "language": "Chinese",
    "use_ollama_embedding": True,
}
#qwen3-30b-a3b
#qwen3-235b-a22b

# Ollama配置（用于本地Embedding生成）
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"

# 应用数据库连接配置 (业务数据库)
DB_HOST = "127.0.0.1"
DB_PORT = 5432
DB_NAME = "retail_dw"
DB_USER = "postgres"
DB_PASSWORD = "postgres"


# 批处理配置
BATCH_PROCESSING_ENABLED = True
BATCH_SIZE = 10
MAX_WORKERS = 4
EMBEDDING_CACHE_SIZE = 100