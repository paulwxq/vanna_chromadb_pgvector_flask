
# 使用的模型类型（"qwen" 或 "deepseek"）
# 通过修改这个值来切换使用的模型
MODEL_TYPE = "deepseek"

# DeepSeek模型配置
DEEPSEEK_CONFIG = {
    "api_key": "xxx",  # 需要替换为实际的API密钥
    "model": "deepseek-reasoner",  # deepseek-chat, deepseek-reasoner
    "allow_llm_to_see_data": True,
    "temperature": 0.6,
    "n_results_sql": 5,
    "n_results_documentation": 5,
    "n_results_ddl": 5,
    "language": "Chinese",
    "use_ollama_embedding": True,  # 自定义，如果是false，则使用chromadb自带embedding
    "enable_thinking": False  # 自定义，是否支持流模式
}


# Qwen模型配置
QWEN_CONFIG = {
    "api_key": "xxx",
    "model": "qwen-plus-latest",
    "allow_llm_to_see_data": True,
    "temperature": 0.6,
    "n_results_sql": 5,
    "n_results_documentation": 5,
    "n_results_ddl": 5,
    "language": "Chinese",
    "use_ollama_embedding": True, #自定义，如果是false，则使用chromadb自带embedding。
    "enable_thinking": True #自定义，是否支持流模式，仅qwen3模型。
}
#qwen3-30b-a3b
#qwen3-235b-a22b
#qwen-plus-latest

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