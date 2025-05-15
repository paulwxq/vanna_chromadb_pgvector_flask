import os
from openai import OpenAI
from vanna.base import VannaBase


class QianWenAI_Chat(VannaBase):
  def __init__(self, client=None, config=None):
    print("...QianWenAI_Chat init...")
    VannaBase.__init__(self, config=config)

    print("传入的 config 参数如下：")
    for key, value in self.config.items():
        print(f"  {key}: {value}")

    # default parameters - can be overrided using config
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
