from vanna.flask import VannaFlaskApp
import ext_config
from vanna_factory import create_vanna_instance

vn = create_vanna_instance()

app = VannaFlaskApp(vn,chart=False,allow_llm_to_see_data=True)
app.run()