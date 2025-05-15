from mychromadb import My_ChromaDB_VectorStore
from myqianwen import QianWenAI_Chat
from mydeepseek import DeepSeekChat
import ext_config

class Myvanna_Qwen(My_ChromaDB_VectorStore, QianWenAI_Chat):
    def __init__(self, config=None):
        My_ChromaDB_VectorStore.__init__(self, config=config)
        QianWenAI_Chat.__init__(self, config=config)

class Myvanna_DeepSeek(My_ChromaDB_VectorStore, DeepSeekChat):
    def __init__(self, config=None):
        My_ChromaDB_VectorStore.__init__(self, config=config)
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
        print(f"已配置使用Ollama ({config_module.OLLAMA_EMBEDDING_MODEL})作为嵌入向量模型")
    
    # 根据模型类型创建实例
    if model_type == "deepseek":
        vn = Myvanna_DeepSeek(config=config)
    elif model_type == "qwen":
        vn = Myvanna_Qwen(config=config)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 使用配置模块中的数据库参数
    vn.connect_to_postgres(
        host=config_module.DB_HOST,
        port=config_module.DB_PORT,
        dbname=config_module.DB_NAME,
        user=config_module.DB_USER,
        password=config_module.DB_PASSWORD
    )    
    return vn 