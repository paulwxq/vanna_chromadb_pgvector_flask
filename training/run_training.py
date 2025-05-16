# run_training.py
import os
import time
import re
import json
import sys
import requests
import pandas as pd
from sqlalchemy import create_engine


from vanna_trainer import (
    train_ddl,
    train_documentation,
    train_sql_example,
    train_question_sql_pair,
    flush_training,
    shutdown_trainer
)

def check_embedding_model_connection():
    """检查嵌入模型连接是否可用
    
    如果无法连接到嵌入模型，则终止程序执行
    
    Returns:
        bool: 连接成功返回True，否则终止程序
    """
    import ext_config

    print("正在检查嵌入模型连接...")
    try:
        # 检查配置是否使用Ollama嵌入
        if ext_config.DEEPSEEK_CONFIG.get("use_ollama_embedding") or ext_config.QWEN_CONFIG.get("use_ollama_embedding"):
            # 直接检查Ollama服务是否可用
            response = requests.get(f"{ext_config.OLLAMA_BASE_URL}/api/tags")
            
            if response.status_code != 200:
                raise Exception(f"Ollama服务返回错误状态码: {response.status_code}")
                
            # 检查嵌入模型是否可用
            embedding_model = ext_config.OLLAMA_EMBEDDING_MODEL
            embedding_dimension = ext_config.OLLAMA_EMBEDDING_DIMENSION
            models = [model["name"] for model in response.json().get("models", [])]
            
            if not any(embedding_model.split(":")[0] in model for model in models):
                print(f"警告: 未找到指定的嵌入模型 {embedding_model}，但Ollama服务正常运行")
                print(f"可用模型: {models}")
            
            # 测试生成向量，验证维度
            print(f"正在验证嵌入模型的向量维度...")
            try:
                # 导入嵌入功能类
                from ollama_embedding import OllamaEmbeddingFunction
                
                # 创建实例并测试
                embedding_function = OllamaEmbeddingFunction(
                    model_name=embedding_model,
                    base_url=ext_config.OLLAMA_BASE_URL
                )
                
                # 测试生成向量
                test_vector = embedding_function.generate_embedding("测试文本")
                actual_dimension = len(test_vector)
                
                if actual_dimension != embedding_dimension:
                    print(f"注意: 模型实际生成的向量维度({actual_dimension})与配置维度({embedding_dimension})不一致")
                    print(f"建议将ext_config.py中的OLLAMA_EMBEDDING_DIMENSION修改为{actual_dimension}")
                else:
                    print(f"向量维度验证成功: {embedding_dimension}")
                
            except Exception as e:
                print(f"向量维度验证失败: {e}")
                
            print(f"Ollama服务连接成功! 使用模型: {embedding_model}, 向量维度: {embedding_dimension}")
            print(f"可以继续训练过程。")
            return True
        else:
            # 使用vanna实例测试
            from vanna_factory import create_vanna_instance
            vn = create_vanna_instance()
            
            # 简单测试vanna实例
            test_sql = "SELECT 1"
            _ = vn.run_sql(test_sql)
            
            print("向量数据库连接成功! 可以继续训练过程。")
            return True
    except Exception as e:
        print(f"\n错误: 无法连接到嵌入模型或向量数据库: {e}")
        print("训练过程终止。请确保Ollama服务正在运行且配置正确，或检查向量数据库连接。")
        sys.exit(1)  # 终止程序执行

def read_file_by_delimiter(filepath, delimiter="---"):
    """通用读取：将文件按分隔符切片为多个段落"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = [block.strip() for block in content.split(delimiter) if block.strip()]
    return blocks

def read_markdown_file_by_sections(filepath):
    """专门用于Markdown文件：按标题(#、##、###)分割文档
    
    Args:
        filepath (str): Markdown文件路径
        
    Returns:
        list: 分割后的Markdown章节列表
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 确定文件是否为Markdown
    is_markdown = filepath.lower().endswith('.md') or filepath.lower().endswith('.markdown')
    
    if not is_markdown:
        # 非Markdown文件使用默认的---分隔
        return read_file_by_delimiter(filepath, "---")
    
    # 直接按照标题级别分割内容，处理#、##和###
    sections = []
    
    # 匹配所有级别的标题（#、##或###开头）
    header_pattern = r'(?:^|\n)((?:#|##|###)[^#].*?)(?=\n(?:#|##|###)[^#]|\Z)'
    all_sections = re.findall(header_pattern, content, re.DOTALL)
    
    for section in all_sections:
        section = section.strip()
        if section:
            sections.append(section)
    
    # 处理没有匹配到标题的情况
    if not sections and content.strip():
        sections = [content.strip()]
        
    return sections

def train_ddl_statements(ddl_file):
    """训练DDL语句
    Args:
        ddl_file (str): DDL文件路径
    """
    print(f"开始训练 DDL: {ddl_file}")
    if not os.path.exists(ddl_file):
        print(f"DDL 文件不存在: {ddl_file}")
        return
    for idx, ddl in enumerate(read_file_by_delimiter(ddl_file, ";"), start=1):
        try:
            print(f"\n DDL 训练 {idx}")
            train_ddl(ddl)
        except Exception as e:
            print(f"错误：DDL #{idx} - {e}")

def train_documentation_blocks(doc_file):
    """训练文档块
    Args:
        doc_file (str): 文档文件路径
    """
    print(f"开始训练 文档: {doc_file}")
    if not os.path.exists(doc_file):
        print(f"文档文件不存在: {doc_file}")
        return
    
    # 检查是否为Markdown文件
    is_markdown = doc_file.lower().endswith('.md') or doc_file.lower().endswith('.markdown')
    
    if is_markdown:
        # 使用Markdown专用分割器
        sections = read_markdown_file_by_sections(doc_file)
        print(f" Markdown文档已分割为 {len(sections)} 个章节")
        
        for idx, section in enumerate(sections, start=1):
            try:
                section_title = section.split('\n', 1)[0].strip()
                print(f"\n Markdown章节训练 {idx}: {section_title}")
                
                # 检查部分长度并提供警告
                if len(section) > 2000:
                    print(f" 章节 {idx} 长度为 {len(section)} 字符，接近API限制(2048)")
                
                train_documentation(section)
            except Exception as e:
                print(f" 错误：章节 #{idx} - {e}")
    else:
        # 非Markdown文件使用传统的---分隔
        for idx, doc in enumerate(read_file_by_delimiter(doc_file, "---"), start=1):
            try:
                print(f"\n 文档训练 {idx}")
                train_documentation(doc)
            except Exception as e:
                print(f" 错误：文档 #{idx} - {e}")

def train_sql_examples(sql_file):
    """训练SQL示例
    Args:
        sql_file (str): SQL示例文件路径
    """
    print(f" 开始训练 SQL 示例: {sql_file}")
    if not os.path.exists(sql_file):
        print(f" SQL 示例文件不存在: {sql_file}")
        return
    for idx, sql in enumerate(read_file_by_delimiter(sql_file, ";"), start=1):
        try:
            print(f"\n SQL 示例训练 {idx}")
            train_sql_example(sql)
        except Exception as e:
            print(f" 错误：SQL #{idx} - {e}")

def train_question_sql_pairs(qs_file):
    """训练问答对
    Args:
        qs_file (str): 问答对文件路径
    """
    print(f" 开始训练 问答对: {qs_file}")
    if not os.path.exists(qs_file):
        print(f" 问答文件不存在: {qs_file}")
        return
    try:
        with open(qs_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(lines, start=1):
            if "::" not in line:
                continue
            question, sql = line.strip().split("::", 1)
            print(f"\n 问答训练 {idx}")
            train_question_sql_pair(question.strip(), sql.strip())
    except Exception as e:
        print(f" 错误：问答训练 - {e}")

def train_formatted_question_sql_pairs(formatted_file):
    """训练格式化的问答对文件
    支持两种格式：
    1. Question: xxx\nSQL: xxx (单行SQL)
    2. Question: xxx\nSQL:\nxxx\nxxx (多行SQL)
    
    Args:
        formatted_file (str): 格式化问答对文件路径
    """
    print(f" 开始训练 格式化问答对: {formatted_file}")
    if not os.path.exists(formatted_file):
        print(f" 格式化问答文件不存在: {formatted_file}")
        return
    
    # 读取整个文件内容
    with open(formatted_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 按双空行分割不同的问答对
    # 使用更精确的分隔符，避免误识别
    pairs = []
    blocks = content.split("\n\nQuestion:")
    
    # 处理第一块（可能没有前导的"\n\nQuestion:"）
    first_block = blocks[0]
    if first_block.strip().startswith("Question:"):
        pairs.append(first_block.strip())
    elif "Question:" in first_block:
        # 处理文件开头没有Question:的情况
        question_start = first_block.find("Question:")
        pairs.append(first_block[question_start:].strip())
    
    # 处理其余块
    for block in blocks[1:]:
        pairs.append("Question:" + block.strip())
    
    # 处理每个问答对
    successfully_processed = 0
    for idx, pair in enumerate(pairs, start=1):
        try:
            if "Question:" not in pair or "SQL:" not in pair:
                print(f" 跳过不符合格式的对 #{idx}")
                continue
                
            # 提取问题部分
            question_start = pair.find("Question:") + len("Question:")
            sql_start = pair.find("SQL:", question_start)
            
            if sql_start == -1:
                print(f" SQL部分未找到，跳过对 #{idx}")
                continue
                
            question = pair[question_start:sql_start].strip()
            
            # 提取SQL部分（支持多行）
            sql_part = pair[sql_start + len("SQL:"):].strip()
            
            # 检查是否存在下一个Question标记（防止解析错误）
            next_question = pair.find("Question:", sql_start)
            if next_question != -1:
                sql_part = pair[sql_start + len("SQL:"):next_question].strip()
            
            if not question or not sql_part:
                print(f" 问题或SQL为空，跳过对 #{idx}")
                continue
            
            # 训练问答对
            print(f"\n格式化问答训练 {idx}")
            print(f"问题: {question}")
            print(f"SQL: {sql_part}")
            train_question_sql_pair(question, sql_part)
            successfully_processed += 1
            
        except Exception as e:
            print(f" 错误：格式化问答训练对 #{idx} - {e}")
    
    print(f"格式化问答训练完成，共成功处理 {successfully_processed} 对问答（总计 {len(pairs)} 对）")

def train_json_question_sql_pairs(json_file):
    """训练JSON格式的问答对
    
    Args:
        json_file (str): JSON格式问答对文件路径
    """
    print(f" 开始训练 JSON格式问答对: {json_file}")
    if not os.path.exists(json_file):
        print(f" JSON问答文件不存在: {json_file}")
        return
    
    try:
        # 读取JSON文件
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 确保数据是列表格式
        if not isinstance(data, list):
            print(f" 错误: JSON文件格式不正确，应为问答对列表")
            return
            
        successfully_processed = 0
        for idx, pair in enumerate(data, start=1):
            try:
                # 检查问答对格式
                if not isinstance(pair, dict) or "question" not in pair or "sql" not in pair:
                    print(f" 跳过不符合格式的对 #{idx}")
                    continue
                
                question = pair["question"].strip()
                sql = pair["sql"].strip()
                
                if not question or not sql:
                    print(f" 问题或SQL为空，跳过对 #{idx}")
                    continue
                
                # 训练问答对
                print(f"\n JSON格式问答训练 {idx}")
                print(f"问题: {question}")
                print(f"SQL: {sql}")
                train_question_sql_pair(question, sql)
                successfully_processed += 1
                
            except Exception as e:
                print(f" 错误：JSON问答训练对 #{idx} - {e}")
        
        print(f"JSON格式问答训练完成，共成功处理 {successfully_processed} 对问答（总计 {len(data)} 对）")
        
    except json.JSONDecodeError as e:
        print(f" 错误：JSON解析失败 - {e}")
    except Exception as e:
        print(f" 错误：处理JSON问答训练 - {e}")

def main():
    """主函数：配置和运行训练流程"""
    
    # 导入os模块
    import os
    import ext_config

    # 检查嵌入模型连接
    check_embedding_model_connection()
    
    # 打印当前使用的向量数据库
    print(f"\n===== 当前使用的向量数据库: {ext_config.VECTOR_DB_TYPE} =====")
    
    # 如果使用PgVector，提供一些额外信息
    if ext_config.VECTOR_DB_TYPE == "pgvector":
        print(f"PgVector数据库: {ext_config.PGVECTOR_HOST}:{ext_config.PGVECTOR_PORT}/{ext_config.PGVECTOR_DB}")
        print(f"PgVector表: {ext_config.PGVECTOR_TABLE}")

    # 打印ChromaDB相关信息 - 仅当使用ChromaDB时才执行
    if ext_config.VECTOR_DB_TYPE.lower() == "chromadb":
        try:
            import mychromadb
            
            # 尝试查看当前使用的ChromaDB文件
            chroma_file = "chroma.sqlite3"  # 默认文件名
            if os.path.exists(chroma_file):
                file_size = os.path.getsize(chroma_file) / 1024  # KB
                print(f"\n===== ChromaDB数据库: {os.path.abspath(chroma_file)} (大小: {file_size:.2f} KB) =====")
            else:
                print("\n===== 未找到默认ChromaDB数据库文件 =====")
                
            # 尝试获取ChromaDB版本
            print(f"===== ChromaDB版本: {mychromadb.__version__ if hasattr(mychromadb, '__version__') else '未知'} =====\n")
        except Exception as e:
            print(f"\n===== 无法获取ChromaDB信息: {e} =====\n")
    
    # 配置基础路径
    BASE_PATH = r"D:\TechDoc\NL2SQL\RetailStoreStarSchemaDataset"  # Windows 路径格式

    # 配置训练文件路径
    TRAINING_FILES = {
        "ddl_1": os.path.join(BASE_PATH, "create_table.ddl"),
        "doc_1": os.path.join(BASE_PATH, "table_detail.md"),
        "json_qs_1": os.path.join(BASE_PATH, "question_sql_pair.json"),
        "sql_1": os.path.join(BASE_PATH, "sql_example.sql"),
    }

    # 添加DDL语句训练
    train_ddl_statements(TRAINING_FILES["ddl_1"])

    #添加文档结构训练
    train_documentation_blocks(TRAINING_FILES["doc_1"])

    # 添加SQL示例训练
    train_sql_examples(TRAINING_FILES["sql_1"])
    
    # 添加JSON格式问答对训练
    train_json_question_sql_pairs(TRAINING_FILES["json_qs_1"])
    
    # 训练结束，刷新和关闭批处理器
    print("\n===== 训练完成，处理剩余批次 =====")
    flush_training()
    shutdown_trainer()
    
    # 验证数据是否成功写入
    print("\n===== 验证训练数据 =====")
    from vanna_factory import create_vanna_instance
    vn = create_vanna_instance()
    
    # 根据向量数据库类型执行不同的验证逻辑
    if ext_config.VECTOR_DB_TYPE.lower() == "pgvector":
        try:
            # 使用PgVector的数据库连接直接查询
            connection_string = f"postgresql://{ext_config.PGVECTOR_USER}:{ext_config.PGVECTOR_PASSWORD}@{ext_config.PGVECTOR_HOST}:{ext_config.PGVECTOR_PORT}/{ext_config.PGVECTOR_DB}"
            engine = create_engine(connection_string)
            
            # 查询总记录数
            query_total = "SELECT COUNT(*) FROM langchain_pg_embedding"
            total_count = pd.read_sql(query_total, engine).iloc[0, 0]
            
            # 按类型查询记录数
            query_by_type = """
            SELECT c.name, COUNT(*) as count
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            GROUP BY c.name
            """
            type_counts = pd.read_sql(query_by_type, engine)
            
            print(f"成功写入 {total_count} 条训练数据:")
            for _, row in type_counts.iterrows():
                print(f" - {row['name']}: {row['count']}条")
                
        except Exception as e:
            print(f"统计训练数据失败: {e}")
            # 使用通用方法作为备选
            training_data = vn.get_training_data()
            if not training_data.empty:
                print(f"成功写入 {len(training_data)} 条训练数据")
            else:
                print("未找到任何训练数据，请检查数据库连接和表结构")
    else:
        # 使用通用方法获取训练数据（包括ChromaDB）
        training_data = vn.get_training_data()
        # get_training_data方法已经打印了详细信息，这里不需要重复
    
    # 输出embedding模型信息
    print("\n===== Embedding模型信息 =====")
    print(f"模型名称: {ext_config.OLLAMA_EMBEDDING_MODEL}")
    print(f"向量维度: {ext_config.OLLAMA_EMBEDDING_DIMENSION}")
    if ext_config.VECTOR_DB_TYPE.lower() == "pgvector":
        print(f"向量数据库: PgVector ({ext_config.PGVECTOR_HOST}:{ext_config.PGVECTOR_PORT}/{ext_config.PGVECTOR_DB})")
    else:
        print(f"向量数据库: ChromaDB ({ext_config.CHROMADB_PATH})")
    print("===== 训练流程完成 =====\n")

if __name__ == "__main__":
    main() 