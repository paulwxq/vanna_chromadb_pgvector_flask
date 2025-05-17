from vanna.flask import VannaFlaskApp
import ext_config,os
from vanna_factory import create_vanna_instance

# 获取当前脚本所在目录（无论你在哪启动它都不影响）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 拼接静态资源和 HTML 的绝对路径
index_html_path = os.path.join(BASE_DIR, "static", "templates", "index.html")
assets_folder = os.path.join(BASE_DIR, "static", "assets")
vn = create_vanna_instance()

# 实例化 VannaFlaskApp
app = VannaFlaskApp(
    vn,
    title="智能数据问答平台",
    subtitle="让 AI 为你写 SQL",
    index_html_path=index_html_path,
    assets_folder=assets_folder,
    allow_llm_to_see_data=True,
    language="Chinese"
)
print("正在启动Flask应用...")
app.run(host="0.0.0.0", port=8084)