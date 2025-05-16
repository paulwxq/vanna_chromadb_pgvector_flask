from vanna.flask import VannaFlaskApp
import ext_config
from vanna_factory import create_vanna_instance

vn = create_vanna_instance()

app = VannaFlaskApp(
    vn,
    title="智能数据问答平台",
    subtitle="让 AI 为你写 SQL",
    chart=False,
    allow_llm_to_see_data=True
)

app.run()