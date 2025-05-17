"""
中文千问AI实现
基于对源码的正确理解，实现正确的方法
"""
import os
from openai import OpenAI
from vanna.base import VannaBase
from typing import List, Dict, Any, Optional


class QianWenAI_Chat_CN(VannaBase):
    """
    中文千问AI聊天类，直接继承VannaBase
    实现正确的方法名(get_sql_prompt而不是generate_sql_prompt)
    """
    def __init__(self, client=None, config=None):
        """
        初始化中文千问AI实例
        
        Args:
            client: 可选，OpenAI兼容的客户端
            config: 配置字典，包含API密钥等配置
        """
        print("初始化QianWenAI_Chat_CN...")
        VannaBase.__init__(self, config=config)

        print("传入的 config 参数如下：")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

        # 设置语言为中文
        self.language = "Chinese"
        
        # 默认参数 - 可通过config覆盖
        self.temperature = 0.7

        if "temperature" in config:
            print(f"temperature is changed to: {config['temperature']}")
            self.temperature = config["temperature"]

        if "api_type" in config:
            raise Exception(
                "Passing api_type is now deprecated. Please pass an OpenAI client instead."
            )

        if "api_base" in config:
            raise Exception(
                "Passing api_base is now deprecated. Please pass an OpenAI client instead."
            )

        if "api_version" in config:
            raise Exception(
                "Passing api_version is now deprecated. Please pass an OpenAI client instead."
            )

        if client is not None:
            self.client = client
            return

        if config is None and client is None:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            return

        if "api_key" in config:
            if "base_url" not in config:
                self.client = OpenAI(api_key=config["api_key"],
                                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
            else:
                self.client = OpenAI(api_key=config["api_key"],
                                    base_url=config["base_url"])
        
        print("中文千问AI初始化完成")
    
    def _response_language(self) -> str:
        """
        返回响应语言指示
        """
        return "请用中文回答。"
    
    def system_message(self, message: str) -> any:
        """
        创建系统消息
        """
        print(f"[DEBUG] 系统消息: {message}")
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        """
        创建用户消息
        """
        print(f"[DEBUG] 用户消息: {message}")
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        """
        创建助手消息
        """
        print(f"[DEBUG] 助手消息: {message}")
        return {"role": "assistant", "content": message}

    def submit_prompt(self, prompt, **kwargs) -> str:
        """
        提交提示词到LLM
        """
        if prompt is None:
            raise Exception("Prompt is None")

        if len(prompt) == 0:
            raise Exception("Prompt is empty")

        # Count the number of tokens in the message log
        # Use 4 as an approximation for the number of characters per token
        num_tokens = 0
        for message in prompt:
            num_tokens += len(message["content"]) / 4

        # 从配置和参数中获取enable_thinking设置
        # 优先使用参数中传入的值，如果没有则从配置中读取，默认为False
        enable_thinking = kwargs.get("enable_thinking", self.config.get("enable_thinking", False))
        
        # 公共参数
        common_params = {
            "messages": prompt,
            "stop": None,
            "temperature": self.temperature,
        }
        
        # 如果启用了thinking，则使用流式处理，但不直接传递enable_thinking参数
        if enable_thinking:
            common_params["stream"] = True
            # 千问API不接受enable_thinking作为参数，可能需要通过header或其他方式传递
            # 也可能它只是默认启用stream=True时的thinking功能
        
        model = None
        # 确定使用的模型
        if kwargs.get("model", None) is not None:
            model = kwargs.get("model", None)
            common_params["model"] = model
        elif kwargs.get("engine", None) is not None:
            engine = kwargs.get("engine", None)
            common_params["engine"] = engine
            model = engine
        elif self.config is not None and "engine" in self.config:
            common_params["engine"] = self.config["engine"]
            model = self.config["engine"]
        elif self.config is not None and "model" in self.config:
            common_params["model"] = self.config["model"]
            model = self.config["model"]
        else:
            if num_tokens > 3500:
                model = "qwen-long"
            else:
                model = "qwen-plus"
            common_params["model"] = model
        
        print(f"\nUsing model {model} for {num_tokens} tokens (approx)")
        
        if enable_thinking:
            # 流式处理模式
            print("使用流式处理模式，启用thinking功能")
            
            # 检查是否需要通过headers传递enable_thinking参数
            response_stream = self.client.chat.completions.create(**common_params)
            
            # 收集流式响应
            collected_thinking = []
            collected_content = []
            
            for chunk in response_stream:
                # 处理thinking部分
                if hasattr(chunk, 'thinking') and chunk.thinking:
                    collected_thinking.append(chunk.thinking)
                
                # 处理content部分
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    collected_content.append(chunk.choices[0].delta.content)
            
            # 可以在这里处理thinking的展示逻辑，如保存到日志等
            if collected_thinking:
                print("Model thinking process:", "".join(collected_thinking))
            
            # 返回完整的内容
            return "".join(collected_content)
        else:
            # 非流式处理模式
            print("使用非流式处理模式")
            response = self.client.chat.completions.create(**common_params)
            
            # Find the first response from the chatbot that has text in it (some responses may not have text)
            for choice in response.choices:
                if "text" in choice:
                    return choice.text

            # If no response with text is found, return the first response's content (which may be empty)
            return response.choices[0].message.content

    # 核心方法：get_sql_prompt
    def get_sql_prompt(self, question: str, 
                      question_sql_list: list, 
                      ddl_list: list, 
                      doc_list: list, 
                      **kwargs) -> List[Dict[str, str]]:
        """
        生成SQL查询的中文提示词
        """
        print("[DEBUG] 正在生成中文SQL提示词...")
        print(f"[DEBUG] 问题: {question}")
        print(f"[DEBUG] 相关SQL数量: {len(question_sql_list) if question_sql_list else 0}")
        print(f"[DEBUG] 相关DDL数量: {len(ddl_list) if ddl_list else 0}")
        print(f"[DEBUG] 相关文档数量: {len(doc_list) if doc_list else 0}")
        
        # 获取dialect
        dialect = getattr(self, 'dialect', 'SQL')
        
        # 创建基础提示词
        messages = [
            self.system_message(
                f"""你是一个专业的SQL助手，根据用户的问题生成正确的{dialect}查询语句。
                你只需生成SQL语句，不需要任何解释或评论。
                用户问题: {question}
                """
            )
        ]

        # 添加相关的DDL（如果有）
        if ddl_list and len(ddl_list) > 0:
            ddl_text = "\n\n".join([f"-- DDL项 {i+1}:\n{ddl}" for i, ddl in enumerate(ddl_list)])
            messages.append(
                self.user_message(
                    f"""
                    以下是可能相关的数据库表结构定义，请基于这些信息生成SQL:
                    
                    {ddl_text}
                    
                    记住，这些只是参考信息，可能并不包含所有需要的表和字段。
                    """
                )
            )

        # 添加相关的文档（如果有）
        if doc_list and len(doc_list) > 0:
            doc_text = "\n\n".join([f"-- 文档项 {i+1}:\n{doc}" for i, doc in enumerate(doc_list)])
            messages.append(
                self.user_message(
                    f"""
                    以下是可能有用的业务逻辑文档:
                    
                    {doc_text}
                    """
                )
            )

        # 添加相关的问题和SQL（如果有）
        if question_sql_list and len(question_sql_list) > 0:
            qs_text = ""
            for i, qs_item in enumerate(question_sql_list):
                qs_text += f"问题 {i+1}: {qs_item.get('question', '')}\n"
                qs_text += f"SQL:\n```sql\n{qs_item.get('sql', '')}\n```\n\n"
                
            messages.append(
                self.user_message(
                    f"""
                    以下是与当前问题相似的问题及其对应的SQL查询:
                    
                    {qs_text}
                    
                    请参考这些样例来生成当前问题的SQL查询。
                    """
                )
            )

        # 添加最终的用户请求和限制
        messages.append(
            self.user_message(
                f"""
                根据以上信息，为以下问题生成一个{dialect}查询语句:
                
                问题: {question}
                
                要求:
                1. 仅输出SQL语句，不要有任何解释或说明
                2. 确保语法正确，符合{dialect}标准
                3. 不要使用不存在的表或字段
                4. 查询应尽可能高效
                """
            )
        )

        return messages
        
    def get_followup_questions_prompt(self, 
                                     question: str, 
                                     sql: str, 
                                     df_metadata: str, 
                                     **kwargs) -> List[Dict[str, str]]:
        """
        生成后续问题的中文提示词
        """
        print("[DEBUG] 正在生成中文后续问题提示词...")
        
        messages = [
            self.system_message(
                f"""你是一个专业的数据分析师，能够根据已有问题提出相关的后续问题。
                {self._response_language()}
                """
            ),
            self.user_message(
                f"""
                原始问题: {question}
                
                已执行的SQL查询:
                ```sql
                {sql}
                ```
                
                数据结构:
                {df_metadata}
                
                请基于上述信息，生成3-5个相关的后续问题，这些问题应该：
                1. 与原始问题和数据相关，是自然的延续
                2. 提供更深入的分析视角或维度拓展
                3. 探索可能的业务洞见和价值发现
                4. 简洁明了，便于用户理解
                5. 确保问题可以通过SQL查询解答，与现有数据结构相关
                
                只需列出问题，不要提供任何解释或SQL。每个问题应该是完整的句子，以问号结尾。
                """
            )
        ]
        
        return messages
        
    def get_summary_prompt(self, question: str, df_markdown: str, **kwargs) -> List[Dict[str, str]]:
        """
        生成摘要的中文提示词
        """
        print("[DEBUG] 正在生成中文摘要提示词...")
        
        messages = [
            self.system_message(
                f"""你是一个专业的数据分析师，能够清晰解释SQL查询的含义和结果。
                {self._response_language()}
                """
            ),
            self.user_message(
                f"""
                你是一个有帮助的数据助手。用户问了这个问题: '{question}'

                以下是一个pandas DataFrame，包含查询的结果: 
                {df_markdown}
                
                请用中文简明扼要地总结这些数据，回答用户的问题。不要提供任何额外的解释，只需提供摘要。
                """
            )
        ]
        
        return messages
        
    def get_plotly_prompt(self, question: str, sql: str, df_metadata: str, 
                        chart_instructions: Optional[str] = None, **kwargs) -> List[Dict[str, str]]:
        """
        生成Python可视化代码的中文提示词
        """
        print("[DEBUG] 正在生成中文Python可视化提示词...")
        
        instructions = chart_instructions if chart_instructions else "生成一个适合展示数据的图表"
        
        messages = [
            self.system_message(
                f"""你是一个专业的Python数据可视化专家，擅长使用Plotly创建数据可视化图表。
                {self._response_language()}
                """
            ),
            self.user_message(
                f"""
                问题: {question}
                
                SQL查询:
                ```sql
                {sql}
                ```
                
                数据结构:
                {df_metadata}
                
                请生成一个Python函数，使用Plotly库为上述数据创建一个可视化图表。要求：
                1. {instructions}
                2. 确保代码语法正确，可直接运行
                3. 图表应直观展示数据中的关键信息和关系
                4. 只需提供Python代码，不要有任何解释
                5. 使用中文作为图表标题、轴标签和图例
                6. 添加合适的颜色方案，保证图表美观
                7. 针对数据类型选择最合适的图表类型
                
                输出格式必须是可以直接运行的Python代码。
                """
            )
        ]
        
        return messages 