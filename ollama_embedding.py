import requests
from typing import List, Union

class OllamaEmbeddingFunction:
    """Ollama嵌入向量生成类，符合ChromaDB的embedding_function接口"""
    
    def __init__(self, model_name="bge-m3:latest", base_url="http://localhost:11434", verbose=False):
        """
        初始化Ollama Embedding Function
        
        Args:
            model_name: Ollama模型名称，默认为"bge-m3:latest"
            base_url: Ollama API的基础URL，默认为"http://localhost:11434"
            verbose: 是否打印详细日志，默认为False (已废弃，现在总是打印详细日志)
        """
        self.embedding_model_name = model_name
        self.ollama_base_url = base_url
        print(f"已初始化Ollama嵌入向量生成器 (模型: {model_name})")
    
    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        """
        ChromaDB调用embedding_function的接口方法
        
        Args:
            input: 单个文本或文本列表
            
        Returns:
            嵌入向量列表
        """
        # 确保输入是列表
        if isinstance(input, str):
            input = [input]
        
        # 为每个文本生成嵌入向量
        embeddings = []
        for text in input:
            embeddings.append(self.generate_embedding(text))
        
        return embeddings
    
    def generate_embedding(self, data: str) -> List[float]:
        """
        使用本地Ollama生成文本向量
        """
        print(f"\n===调试: 开始生成嵌入向量===")
        print(f"输入文本长度: {len(data)} 字符")
        
        # 处理空字符串输入
        if not data or len(data.strip()) == 0:
            print("[WARNING] 输入文本为空，返回零向量")
            # 返回1024维的零向量
            return [0.0] * 1024
        
        try:
            # 直接调用Ollama API
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={"model": self.embedding_model_name, "prompt": data}
            )
            
            if response.status_code != 200:
                error_msg = f"API请求错误: {response.status_code}, {response.text}"
                print(f"错误: {error_msg}")
                raise Exception(error_msg)
                
            result = response.json()
            vector = result.get("embedding")
            
            if not vector:
                error_msg = "API返回中没有embedding字段"
                print(f"错误: {error_msg}")
                raise Exception(error_msg)
                
            print(f"向量长度: {len(vector)}")
            print(f"向量前5个元素: {vector[:5]}")
            print("===调试: 嵌入向量生成成功===\n")
                
            return vector
            
        except Exception as e:
            print(f"[ERROR] Ollama嵌入向量生成异常: {str(e)}")
            print("===调试: 嵌入向量生成失败===\n")
            raise 