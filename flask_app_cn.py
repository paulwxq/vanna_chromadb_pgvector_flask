"""
Flask应用入口文件（中文版），使用直接中文提示词实现
"""
from vanna.flask import VannaFlaskApp
import ext_config
from vanna_factory import create_vanna_instance
from pathlib import Path
import logging

# 配置日志记录
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

root_dir = Path(__file__).parent.resolve()

# 检查中文提示词设置
if not hasattr(ext_config, 'USE_CHINESE_PROMPTS'):
    logger.info("添加USE_CHINESE_PROMPTS配置到ext_config...")
    ext_config.USE_CHINESE_PROMPTS = True
else:
    # 尊重ext_config.py中的设置，不强制覆盖
    logger.info(f"使用ext_config.py中的USE_CHINESE_PROMPTS设置: {ext_config.USE_CHINESE_PROMPTS}")

print(f"中文提示词状态: {'已启用' if ext_config.USE_CHINESE_PROMPTS else '未启用'}")

# 使用直接实现版工厂函数创建Vanna实例
vn = create_vanna_instance()

app = VannaFlaskApp(
    vn,
    title="智能数据问答平台",
    subtitle="让 AI 为你写 SQL",
    chart=False,
    allow_llm_to_see_data=True
)

# 运行Flask应用
print("正在启动Flask应用...")
app.run(host="0.0.0.0", port=8084) 