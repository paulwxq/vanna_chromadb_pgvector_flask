from mychromadb import My_ChromaDB_VectorStore
from qianwen import QianWenAI_Chat
import ext_config

class Myvanna(My_ChromaDB_VectorStore, QianWenAI_Chat):
    def __init__(self, config=None):
        My_ChromaDB_VectorStore.__init__(self, config=config)
        QianWenAI_Chat.__init__(self, config=config)

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
    
    # 复制配置以避免修改原配置
    config = config_module.QWEN_CONFIG.copy()
    
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
        print(f"已配置使用Ollama ({config_module.OLLAMA_EMBEDDING_MODEL})作为嵌入向量模型")
    
    # 使用配置初始化Myvanna
    vn = Myvanna(config=config)
    
    # 使用配置模块中的数据库参数
    vn.connect_to_postgres(
        host=config_module.DB_HOST,
        port=config_module.DB_PORT,
        dbname=config_module.DB_NAME,
        user=config_module.DB_USER,
        password=config_module.DB_PASSWORD
    )    
    return vn 