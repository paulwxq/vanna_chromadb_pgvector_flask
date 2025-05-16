from mychromadb import My_ChromaDB_VectorStore
from myqianwen import QianWenAI_Chat
from mypgvector import PG_VectorStore
from mydeepseek import DeepSeekChat
import ext_config

class Myvanna_Qwen_ChromaDB(My_ChromaDB_VectorStore, QianWenAI_Chat):
    def __init__(self, config=None):
        My_ChromaDB_VectorStore.__init__(self, config=config)
        QianWenAI_Chat.__init__(self, config=config)

class Myvanna_DeepSeek_ChromaDB(My_ChromaDB_VectorStore, DeepSeekChat):
    def __init__(self, config=None):
        My_ChromaDB_VectorStore.__init__(self, config=config)
        DeepSeekChat.__init__(self, config=config)

class Myvanna_Qwen_PgVector(PG_VectorStore, QianWenAI_Chat):
    def __init__(self, config=None):
        PG_VectorStore.__init__(self, config=config)
        QianWenAI_Chat.__init__(self, config=config)

class Myvanna_DeepSeek_PgVector(PG_VectorStore, DeepSeekChat):
    def __init__(self, config=None):
        PG_VectorStore.__init__(self, config=config)
        DeepSeekChat.__init__(self, config=config)


def create_vanna_instance(config_module=None):
    """
    工厂函数：创建并初始化一个Myvanna实例
    
    Args:
        config_module: 配置模块，默认为None时使用ext_config
        
    Returns:
        初始化后的Myvanna实例
    """
    # 如果没有提供配置模块，使用默认的ext_config
    if config_module is None:
        config_module = ext_config

    # 确定使用的模型类型
    model_type = config_module.MODEL_TYPE.lower()
    
    # 确定使用的向量数据库类型
    vector_db_type = config_module.VECTOR_DB_TYPE.lower()
    
    # 根据模型类型选择对应的配置
    if model_type == "deepseek":
        config = config_module.DEEPSEEK_CONFIG.copy()
        print(f"创建DeepSeek模型实例，使用模型: {config.get('model', 'deepseek-chat')}")
    elif model_type == "qwen":
        config = config_module.QWEN_CONFIG.copy()
        print(f"创建Qwen模型实例，使用模型: {config.get('model', 'qwen-plus-latest')}")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}") 
    
    # 如果配置指定了使用Ollama的embedding
    if config.get("use_ollama_embedding", False):
        # 导入OllamaEmbeddingFunction
        from ollama_embedding import OllamaEmbeddingFunction
        
        # 创建OllamaEmbeddingFunction实例
        embedding_function = OllamaEmbeddingFunction(
            model_name=config_module.OLLAMA_EMBEDDING_MODEL,
            base_url=config_module.OLLAMA_BASE_URL
        )
        
        # 将embedding_function添加到配置中
        config["embedding_function"] = embedding_function
        print(f"已配置使用Ollama ({config_module.OLLAMA_EMBEDDING_MODEL})作为嵌入向量模型，维度: {config_module.OLLAMA_EMBEDDING_DIMENSION}")
    
    # 根据向量数据库类型添加特定的配置
    if vector_db_type == "pgvector":
        # 添加PgVector所需的连接字符串
        connection_string = f"postgresql://{config_module.PGVECTOR_USER}:{config_module.PGVECTOR_PASSWORD}@{config_module.PGVECTOR_HOST}:{config_module.PGVECTOR_PORT}/{config_module.PGVECTOR_DB}"
        config["connection_string"] = connection_string
        print(f"已配置使用PgVector作为向量数据库：{config_module.PGVECTOR_HOST}:{config_module.PGVECTOR_PORT}/{config_module.PGVECTOR_DB}")
    elif vector_db_type == "chromadb":
        # 添加ChromaDB所需的路径配置
        config["path"] = config_module.CHROMADB_PATH
        print(f"已配置使用ChromaDB作为向量数据库：{config_module.CHROMADB_PATH}")
    else:
        raise ValueError(f"不支持的向量数据库类型: {vector_db_type}")
    
    # 根据模型类型和向量数据库类型创建实例
    if model_type == "deepseek" and vector_db_type == "chromadb":
        vn = Myvanna_DeepSeek_ChromaDB(config=config)
    elif model_type == "deepseek" and vector_db_type == "pgvector":
        vn = Myvanna_DeepSeek_PgVector(config=config)
    elif model_type == "qwen" and vector_db_type == "chromadb":
        vn = Myvanna_Qwen_ChromaDB(config=config)
    elif model_type == "qwen" and vector_db_type == "pgvector":
        vn = Myvanna_Qwen_PgVector(config=config)
    else:
        raise ValueError(f"不支持的组合: 模型类型={model_type}, 向量数据库类型={vector_db_type}")
    
    # 使用配置模块中的数据库参数连接到业务数据库
    vn.connect_to_postgres(
        host=config_module.DB_HOST,
        port=config_module.DB_PORT,
        dbname=config_module.DB_NAME,
        user=config_module.DB_USER,
        password=config_module.DB_PASSWORD
    )    
    return vn