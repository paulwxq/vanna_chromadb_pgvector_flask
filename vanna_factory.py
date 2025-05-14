from vanna.chromadb import ChromaDB_VectorStore
from qianwen import QianWenAI_Chat
import ext_config

class Myvanna(ChromaDB_VectorStore, QianWenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
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
    
    # 使用配置模块中的QWEN_CONFIG初始化
    vn = Myvanna(config=config_module.QWEN_CONFIG)
    
    # 使用配置模块中的数据库参数
    vn.connect_to_postgres(
        host=config_module.DB_HOST,
        port=config_module.DB_PORT,
        dbname=config_module.DB_NAME,
        user=config_module.DB_USER,
        password=config_module.DB_PASSWORD
    )    
    return vn 