from vanna.flask import VannaFlaskApp
import ext_config
from vanna_factory import create_vanna_instance

vn = create_vanna_instance()
vn.list_config_parameters(print_output=True)

app = VannaFlaskApp(vn,chart=False)
app.run()