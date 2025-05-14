# run_training.py
import os
import time
import re
import json
from vanna_trainer import (
    train_ddl,
    train_documentation,
    train_sql_example,
    train_question_sql_pair,
    flush_training,
    shutdown_trainer
)
from tools.chroma_cleaner import clear_chroma_database

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
    # 打印ChromaDB相关信息
    try:
        import os
        import chromadb
        
        # 尝试查看当前使用的ChromaDB文件
        chroma_file = "chroma.sqlite3"  # 默认文件名
        if os.path.exists(chroma_file):
            file_size = os.path.getsize(chroma_file) / 1024  # KB
            print(f"\n===== ChromaDB数据库: {os.path.abspath(chroma_file)} (大小: {file_size:.2f} KB) =====")
        else:
            print("\n===== 未找到默认ChromaDB数据库文件 =====")
            
        # 尝试获取ChromaDB版本
        print(f"===== ChromaDB版本: {chromadb.__version__} =====\n")
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


if __name__ == "__main__":
    main()
