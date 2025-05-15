# vanna-flask
Web server for chatting with your database



https://github.com/vanna-ai/vanna-flask/assets/7146154/5794c523-0c99-4a53-a558-509fa72885b9



# Setup

## Set your environment variables
```
VANNA_MODEL=
VANNA_API_KEY=
SNOWFLAKE_ACCOUNT=
SNOWFLAKE_USERNAME=
SNOWFLAKE_PASSWORD=
SNOWFLAKE_DATABASE=
SNOWFLAKE_WAREHOUSE=
```

## Install dependencies
```
pip install -r requirements.txt
```

## Run the server
```
python app.py
```


以下是修改PostgreSQL表字段类型的SQL语句，将langchain_pg_embedding表的collection_id字段从TEXT类型改为UUID类型：
ALTER TABLE langchain_pg_embedding
ALTER COLUMN collection_id TYPE uuid
USING collection_id::uuid;