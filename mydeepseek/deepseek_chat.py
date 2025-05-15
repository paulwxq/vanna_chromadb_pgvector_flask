import os

from openai import OpenAI
from vanna.base import VannaBase
#from base import VannaBase


# from vanna.chromadb import ChromaDB_VectorStore

# class DeepSeekVanna(ChromaDB_VectorStore, DeepSeekChat):
#     def __init__(self, config=None):
#         ChromaDB_VectorStore.__init__(self, config=config)
#         DeepSeekChat.__init__(self, config=config)

# vn = DeepSeekVanna(config={"api_key": "sk-************", "model": "deepseek-chat"})


class DeepSeekChat(VannaBase):
    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)
        
        print("...DeepSeekChat init...")
        if config is None:
            raise ValueError(
                "For DeepSeek, config must be provided with an api_key and model"
            )
        if "api_key" not in config:
            raise ValueError("config must contain a DeepSeek api_key")

        if "model" not in config:
            config["model"] = "deepseek-chat"  # 默认模型
            print(f"未指定模型，使用默认模型: {config['model']}")
        
        # 设置默认值
        self.temperature = config.get("temperature", 0.7)
        self.model = config["model"]
        
        print("传入的 config 参数如下：")
        for key, value in config.items():
            if key != "api_key":  # 不打印API密钥
                print(f"  {key}: {value}")
        
        # 使用标准的OpenAI客户端，但更改基础URL
        self.client = OpenAI(
            api_key=config["api_key"], 
            base_url="https://api.deepseek.com/v1"
        )
    
    def system_message(self, message: str) -> any:
        print(f"system_content: {message}")
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        print(f"\nuser_content: {message}")
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        print(f"assistant_content: {message}")
        return {"role": "assistant", "content": message}

    def submit_prompt(self, prompt, **kwargs) -> str:
        if prompt is None:
            raise Exception("Prompt is None")

        if len(prompt) == 0:
            raise Exception("Prompt is empty")

        # Count the number of tokens in the message log
        # Use 4 as an approximation for the number of characters per token
        num_tokens = 0
        for message in prompt:
            num_tokens += len(message["content"]) / 4
        
        # 从配置和参数中获取model设置，kwargs优先
        model = kwargs.get("model", self.model)
        
        print(f"\nUsing model {model} for {num_tokens} tokens (approx)")
        
        # 创建请求参数
        chat_params = {
            "model": model,
            "messages": prompt,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        try:
            chat_response = self.client.chat.completions.create(**chat_params)
            # 返回生成的文本
            return chat_response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API调用失败: {e}")
            raise

    def generate_sql(self, question: str, **kwargs) -> str:
        # 使用父类的 generate_sql
        sql = super().generate_sql(question, **kwargs)
        
        # 替换 "\_" 为 "_"，解决特殊字符转义问题
        sql = sql.replace("\\_", "_")
        
        return sql