import ast
import json
import logging
import uuid
import hashlib

import pandas as pd
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from sqlalchemy import create_engine, text

from vanna.exceptions import ValidationError
from vanna.base import VannaBase
from vanna.types import TrainingPlan, TrainingPlanItem


class PG_VectorStore(VannaBase):
    def __init__(self, config=None):
        if not config or "connection_string" not in config:
            raise ValueError(
                "A valid 'config' dictionary with a 'connection_string' is required.")

        VannaBase.__init__(self, config=config)

        print("...PG_VectorStore init...")
        
        if config and "connection_string" in config:
            self.connection_string = config.get("connection_string")
            self.n_results_sql = config.get("n_results_sql", 10)
            self.n_results_documentation = config.get("n_results_documentation", 10)
            self.n_results_ddl = config.get("n_results_ddl", 10)
            print(f"向量搜索默认结果数: SQL={self.n_results_sql}, DDL={self.n_results_ddl}, Documentation={self.n_results_documentation}")

        if config and "embedding_function" in config:
            self.embedding_function = config.get("embedding_function")
            print(f"使用自定义embedding函数: {type(self.embedding_function).__name__}")
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print(f"使用默认embedding函数: HuggingFaceEmbeddings (all-MiniLM-L6-v2)")

        try:
            # 初始化数据库引擎
            self.engine = create_engine(self.connection_string)
            
            # 初始化集合
            self.sql_collection = PGVector(
                embeddings=self.embedding_function,
                collection_name="sql",
                connection=self.connection_string,
            )
            self.ddl_collection = PGVector(
                embeddings=self.embedding_function,
                collection_name="ddl",
                connection=self.connection_string,
            )
            self.documentation_collection = PGVector(
                embeddings=self.embedding_function,
                collection_name="documentation",
                connection=self.connection_string,
            )
            print("PgVector集合初始化成功")
        except Exception as e:
            print(f"PgVector集合初始化失败: {e}")
            raise

    def _generate_int_id(self, content, prefix=0):
        """生成整数ID
        
        Args:
            content: 用于生成ID的内容
            prefix: ID前缀，用于区分不同类型的ID
                   0: 文档
                   1: DDL
                   2: SQL
            
        Returns:
            整数ID
        """
        # 使用内容的哈希值生成一个大整数，但要确保在PostgreSQL INTEGER范围内
        # PostgreSQL INTEGER范围: -2,147,483,648 到 +2,147,483,647
        # 我们使用较小的范围来确保安全: 0 到 999,999
        content_hash = int(hashlib.md5(content.encode()).hexdigest(), 16) % 1000000
        
        # 添加较小的前缀以区分不同类型 (1-3位数字)
        # 0-99: 文档
        # 100-199: DDL
        # 200-299: SQL
        prefix_map = {0: 0, 1: 100000, 2: 200000}
        prefix_value = prefix_map.get(prefix, 0)
        
        return prefix_value + content_hash

    def add_question_sql(self, question: str, sql: str, **kwargs) -> int:
        """添加问题-SQL对到向量数据库
        
        Args:
            question: 问题文本
            sql: SQL语句
            
        Returns:
            添加的记录ID (整数)
        """
        question_sql_json = json.dumps(
            {
                "question": question,
                "sql": sql,
            },
            ensure_ascii=False,
        )
        # 生成整数ID
        id = self._generate_int_id(question_sql_json, prefix=2)
        createdat = kwargs.get("createdat")
        doc = Document(
            page_content=question_sql_json,
            metadata={"id": id, "createdat": createdat},
        )
        
        try:
            self.sql_collection.add_documents([doc], ids=[str(id)])
            print(f"添加问题-SQL对成功，ID: {id}")
            return id
        except Exception as e:
            print(f"添加问题-SQL对失败: {e}")
            raise

    def add_ddl(self, ddl: str, **kwargs) -> int:
        """添加DDL语句到向量数据库
        
        Args:
            ddl: DDL语句
            
        Returns:
            添加的记录ID (整数)
        """
        # 生成整数ID
        _id = self._generate_int_id(ddl, prefix=1)
        doc = Document(
            page_content=ddl,
            metadata={"id": _id},
        )
        
        try:
            self.ddl_collection.add_documents([doc], ids=[str(_id)])
            print(f"添加DDL成功，ID: {_id}")
            return _id
        except Exception as e:
            print(f"添加DDL失败: {e}")
            raise

    def add_documentation(self, documentation: str, **kwargs) -> int:
        """添加文档到向量数据库
        
        Args:
            documentation: 文档内容
            
        Returns:
            添加的记录ID (整数)
        """
        # 生成整数ID
        _id = self._generate_int_id(documentation, prefix=0)
        doc = Document(
            page_content=documentation,
            metadata={"id": _id},
        )
        
        try:
            self.documentation_collection.add_documents([doc], ids=[str(_id)])
            print(f"添加文档成功，ID: {_id}")
            return _id
        except Exception as e:
            print(f"添加文档失败: {e}")
            raise

    def get_collection(self, collection_name):
        """获取指定的集合
        
        Args:
            collection_name: 集合名称
            
        Returns:
            集合对象
        """
        match collection_name:
            case "sql":
                return self.sql_collection
            case "ddl":
                return self.ddl_collection
            case "documentation":
                return self.documentation_collection
            case _:
                raise ValueError("指定的集合不存在.")

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """查询相似问题并返回文档列表，同时打印相似度信息
        
        Args:
            question: 问题文本
            
        Returns:
            相似问题的SQL对列表
        """
        try:
            documents = self.sql_collection.similarity_search(query=question, k=self.n_results_sql)
            print(f"查询问题: {question}")
            print(f"找到 {len(documents)} 个相似问题")
            return [ast.literal_eval(document.page_content) for document in documents]
        except Exception as e:
            print(f"查询相似问题失败: {e}")
            return []

    def get_related_ddl(self, question: str, **kwargs) -> list:
        """查询相关DDL并返回文档列表，同时打印相似度信息
        
        Args:
            question: 问题文本
            
        Returns:
            相关DDL列表
        """
        try:
            documents = self.ddl_collection.similarity_search(query=question, k=self.n_results_ddl)
            print(f"DDL查询: {question}")
            print(f"找到 {len(documents)} 个相关DDL")
            return [document.page_content for document in documents]
        except Exception as e:
            print(f"查询相关DDL失败: {e}")
            return []

    def get_related_documentation(self, question: str, **kwargs) -> list:
        """查询相关文档并返回文档列表，同时打印相似度信息
        
        Args:
            question: 问题文本
            
        Returns:
            相关文档列表
        """
        try:
            documents = self.documentation_collection.similarity_search(query=question, k=self.n_results_documentation)
            print(f"文档查询: {question}")
            print(f"找到 {len(documents)} 个相关文档")
            return [document.page_content for document in documents]
        except Exception as e:
            print(f"查询相关文档失败: {e}")
            return []

    def train(
        self,
        question: str | None = None,
        sql: str | None = None,
        ddl: str | None = None,
        documentation: str | None = None,
        plan: TrainingPlan | None = None,
        createdat: str | None = None,
    ):
        """训练函数：添加各类数据到向量数据库
        
        Args:
            question: 问题文本
            sql: SQL语句
            ddl: DDL语句
            documentation: 文档内容
            plan: 训练计划
            createdat: 创建时间
            
        Returns:
            添加的记录ID
        """
        if question and not sql:
            raise ValidationError("请提供SQL查询。")

        if documentation:
            logging.info(f"添加文档: {documentation}")
            return self.add_documentation(documentation)

        if sql and question:
            return self.add_question_sql(question=question, sql=sql, createdat=createdat)

        if ddl:
            logging.info(f"添加DDL: {ddl}")
            return self.add_ddl(ddl)

        if plan:
            for item in plan._plan:
                if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
                    self.add_ddl(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
                    self.add_documentation(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL and item.item_name:
                    self.add_question_sql(question=item.item_name, sql=item.item_value)

    def get_training_data(self, **kwargs) -> pd.DataFrame:
        """获取训练数据
        
        从langchain_pg_embedding和langchain_pg_collection表联合查询，
        根据collection表确定训练数据类型，不再从ID格式推断
        
        Returns:
            包含所有训练数据的DataFrame
        """
        # 建立数据库连接
        try:
            engine = create_engine(self.connection_string)

            # 联合查询，通过collection表确定类型
            query = """
            SELECT e.cmetadata, e.document, c.name as training_data_type
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            """
            df_result = pd.read_sql(query, engine)
            
            # 用于累积处理后的行的列表
            processed_rows = []

            # 处理DataFrame中的每一行
            for _, row in df_result.iterrows():
                custom_id = row["cmetadata"]["id"]
                document = row["document"]
                training_data_type = row["training_data_type"].lower()  # 转为小写以统一处理
                
                # 根据类型处理数据
                if training_data_type == "sql":
                    # 将文档字符串转换为字典
                    try:
                        doc_dict = ast.literal_eval(document)
                        question = doc_dict.get("question")
                        content = doc_dict.get("sql")
                    except (ValueError, SyntaxError):
                        print(f"警告: SQL解析错误，ID={custom_id}，将尝试更宽松的解析")
                        # 尝试更宽松的解析
                        if isinstance(document, str) and "question" in document and "sql" in document:
                            try:
                                # 尝试使用JSON解析
                                doc_dict = json.loads(document)
                                question = doc_dict.get("question")
                                content = doc_dict.get("sql")
                            except:
                                # 如果解析失败，使用原始文档
                                question = None
                                content = document
                        else:
                            question = None
                            content = document
                elif training_data_type in ["documentation", "ddl"]:
                    question = None  # 问题的默认值
                    content = document
                else:
                    # 未知类型，使用原始文档
                    print(f"警告: 未知训练数据类型 '{training_data_type}'，ID={custom_id}")
                    question = None
                    content = document

                # 将处理后的数据添加到列表中
                processed_rows.append(
                    {"id": custom_id, "question": question, "content": content, "training_data_type": training_data_type}
                )

            # 从处理后的行列表创建DataFrame
            df_processed = pd.DataFrame(processed_rows)
            
            # 输出详细统计信息
            total = len(df_processed)
            if not df_processed.empty:
                by_type = df_processed["training_data_type"].value_counts().to_dict()
                print(f"获取到 {total} 条训练数据:")
                for type_name, count in by_type.items():
                    print(f" - {type_name}: {count}条")
            else:
                print(f"获取到 {total} 条训练数据")
            
            return df_processed
        except Exception as e:
            print(f"获取训练数据失败: {e}")
            # 返回空DataFrame
            return pd.DataFrame(columns=["id", "question", "content", "training_data_type"])

    def remove_training_data(self, id: str, **kwargs) -> bool:
        """删除训练数据
        
        Args:
            id: 记录ID
            
        Returns:
            是否删除成功
        """
        # 创建数据库引擎
        try:
            engine = create_engine(self.connection_string)

            # SQL DELETE 语句
            delete_statement = text(
                """
                DELETE FROM langchain_pg_embedding
                WHERE cmetadata ->> 'id' = :id
            """
            )

            # 连接到数据库并执行删除语句
            with engine.connect() as connection:
                # 开始事务
                with connection.begin() as transaction:
                    try:
                        result = connection.execute(delete_statement, {"id": id})
                        # 如果删除成功则提交事务
                        transaction.commit()
                        # 检查是否有行被删除，并相应地返回True或False
                        success = result.rowcount > 0
                        print(f"删除训练数据 {id}: {'成功' if success else '失败'}")
                        return success
                    except Exception as e:
                        # 错误时回滚事务
                        logging.error(f"发生错误: {e}")
                        transaction.rollback()
                        print(f"删除训练数据失败: {e}")
                        return False
        except Exception as e:
            print(f"删除训练数据失败: {e}")
            return False

    def remove_collection(self, collection_name: str) -> bool:
        """删除集合中的所有数据
        
        Args:
            collection_name: 集合名称 ('ddl', 'sql', 或 'documentation')
            
        Returns:
            是否删除成功
        """
        try:
            engine = create_engine(self.connection_string)

            # 根据集合名称确定ID范围
            id_range = {
                "documentation": "(cmetadata->>'id')::int < 100000",
                "ddl": "(cmetadata->>'id')::int >= 100000 AND (cmetadata->>'id')::int < 200000",
                "sql": "(cmetadata->>'id')::int >= 200000"
            }
            
            condition = id_range.get(collection_name)
            if not condition:
                # 尝试使用旧格式的后缀
                suffix_map = {"ddl": "ddl", "sql": "sql", "documentation": "doc"}
                suffix = suffix_map.get(collection_name)
                
                if not suffix:
                    logging.info("无效的集合名称。请从 'ddl', 'sql', 或 'documentation' 中选择。")
                    return False
                    
                # 使用旧的ID格式条件
                condition = f"cmetadata->>'id' LIKE '%{suffix}'"

            # SQL 查询，根据条件删除行
            query = text(
                f"""
                DELETE FROM langchain_pg_embedding
                WHERE {condition}
            """
            )

            # 在事务块内执行删除操作
            with engine.connect() as connection:
                with connection.begin() as transaction:
                    try:
                        result = connection.execute(query)
                        transaction.commit()  # 显式提交事务
                        if result.rowcount > 0:
                            logging.info(
                                f"从 langchain_pg_embedding 表中删除了 {result.rowcount} 行，集合为 {collection_name}。"
                            )
                            print(f"删除集合 {collection_name} 成功，删除了 {result.rowcount} 行数据")
                            return True
                        else:
                            logging.info(f"集合 {collection_name} 没有删除任何行。")
                            print(f"集合 {collection_name} 没有数据可删除")
                            return False
                    except Exception as e:
                        logging.error(f"发生错误: {e}")
                        transaction.rollback()  # 错误时回滚
                        print(f"删除集合 {collection_name} 失败: {e}")
                        return False
        except Exception as e:
            print(f"删除集合失败: {e}")
            return False

    def generate_embedding(self, data: str, **kwargs):
        """生成嵌入向量
        
        Args:
            data: 要嵌入的文本
            
        Returns:
            嵌入向量
        """
        try:
            return self.embedding_function.encode([data])[0]
        except Exception as e:
            print(f"生成嵌入向量失败: {e}")
            raise