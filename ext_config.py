# 导入dotenv库读取.env文件
from dotenv import load_dotenv
import os

# 加载.env文件中的环境变量
load_dotenv()

# 使用的模型类型（"qwen" 或 "deepseek"）
# 通过修改这个值来切换使用的模型
MODEL_TYPE = "qwen"

# 使用的向量数据库类型 ("chromadb" 或 "pgvector")
# 通过修改这个值来切换使用的向量数据库
VECTOR_DB_TYPE = "pgvector"

# DeepSeek模型配置
DEEPSEEK_CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取API密钥
    "model": "deepseek-reasoner",  # deepseek-chat, deepseek-reasoner
    "allow_llm_to_see_data": True,
    "temperature": 0.6,
    "n_results_sql": 6,
    "n_results_documentation": 6,
    "n_results_ddl": 6,
    "language": "Chinese",
    "use_ollama_embedding": True,  # 自定义，如果是false，则使用chromadb自带embedding
    "enable_thinking": False  # 自定义，是否支持流模式
}


# Qwen模型配置
QWEN_CONFIG = {
    "api_key": os.getenv("QWEN_API_KEY"),  # 从环境变量读取API密钥
    "model": "qwen-plus",
    "allow_llm_to_see_data": True,
    "temperature": 0.6,
    "n_results_sql": 6,
    "n_results_documentation": 6,
    "n_results_ddl": 6,
    "language": "Chinese",
    "use_ollama_embedding": True, #自定义，如果是false，则使用chromadb自带embedding。
    "enable_thinking": False #自定义，是否支持流模式，仅qwen3模型。
}
#qwen3-30b-a3b
#qwen3-235b-a22b
#qwen-plus-latest
#qwen-plus

# Ollama配置（用于本地Embedding生成）
OLLAMA_BASE_URL = "http://localhost:11434"
#OLLAMA_EMBEDDING_MODEL = "bge-m3:567m"
OLLAMA_EMBEDDING_MODEL = "quentinz/bge-large-zh-v1.5"
#OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"
#OLLAMA_EMBEDDING_MODEL = "turingdance/m3e-base:latest"
#OLLAMA_EMBEDDING_MODEL = "chevalblanc/acge_text_embedding:latest"
#OLLAMA_EMBEDDING_DIMENSION = 1024  # 嵌入向量维度，可以根据不同模型进行调整

OLLAMA_EMBEDDING_DIMENSION = 1024  

# 应用数据库连接配置 (业务数据库)
DB_HOST = "192.168.67.10"
DB_PORT = 5432
DB_NAME = "retail_dw"
DB_USER = "postgres"
DB_PASSWORD = "postgres"

# PgVector数据库连接配置 (向量数据库，独立于业务数据库)
PGVECTOR_HOST = "192.168.67.10"
PGVECTOR_PORT = 5432
PGVECTOR_DB = "vector_db"
PGVECTOR_USER = "postgres"
PGVECTOR_PASSWORD = "postgres"
PGVECTOR_TABLE = "langchain_pg_embedding"  # PgVector表名

# ChromaDB配置
CHROMADB_PATH = "."  # ChromaDB文件存储路径

# 批处理配置
BATCH_PROCESSING_ENABLED = True
BATCH_SIZE = 10
MAX_WORKERS = 4
EMBEDDING_CACHE_SIZE = 100