from vanna.flask import VannaFlaskApp
import ext_config
from vanna_factory import create_vanna_instance

vn = create_vanna_instance()

vn.train(ddl="""
    CREATE TABLE dim_campaigns (
    campaign_sk INTEGER PRIMARY KEY,
    campaign_id VARCHAR(10) NOT NULL,
    campaign_name VARCHAR(100) NOT NULL,
    start_date_sk INTEGER NOT NULL,
    end_date_sk INTEGER NOT NULL,
    campaign_budget BIGINT
    );
         """)

app = VannaFlaskApp(vn)
app.run()