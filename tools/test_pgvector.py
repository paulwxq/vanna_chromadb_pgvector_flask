# tools/test_pgvector.py
"""
测试PgVector向量数据库功能
"""

import sys
import os
import time
import argparse

# 添加父目录到路径，确保能正确导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置文件和工厂函数
import ext_config
from vanna_factory import create_vanna_instance

def test_pgvector_basic():
    """
    测试PgVector基本功能
    """
    print("===== 测试PgVector基本功能 =====")
    
    # 强制使用PgVector
    ext_config.VECTOR_DB_TYPE = "pgvector"
    
    # 创建Vanna实例
    print("创建Vanna实例...")
    vn = create_vanna_instance()
    
    # 测试添加文档
    print("\n测试添加文档...")
    doc_id = vn.train(documentation="这是一个测试文档，用于验证PgVector功能。")
    print(f"添加文档成功，ID: {doc_id}")
    
    # 测试添加DDL
    print("\n测试添加DDL...")
    ddl_id = vn.train(ddl="CREATE TABLE test (id INT, name VARCHAR(100));")
    print(f"添加DDL成功，ID: {ddl_id}")
    
    # 测试添加SQL问答对
    print("\n测试添加SQL问答对...")
    sql_id = vn.train(question="查询所有用户", sql="SELECT * FROM users;")
    print(f"添加SQL问答对成功，ID: {sql_id}")
    
    # 测试查询相似文档
    print("\n测试查询相似文档...")
    docs = vn.get_related_documentation("测试文档")
    print(f"查询到 {len(docs)} 个相似文档")
    
    # 测试查询相似DDL
    print("\n测试查询相似DDL...")
    ddls = vn.get_related_ddl("创建表")
    print(f"查询到 {len(ddls)} 个相似DDL")
    
    # 测试查询相似SQL
    print("\n测试查询相似SQL...")
    sqls = vn.get_similar_question_sql("查询用户")
    print(f"查询到 {len(sqls)} 个相似SQL")
    
    # 测试获取所有训练数据
    print("\n测试获取所有训练数据...")
    df = vn.get_training_data()
    print(f"获取到 {len(df)} 条训练数据")
    if not df.empty:
        print("数据示例:")
        print(df.head(3))
    
    print("\n===== PgVector基本功能测试完成 =====")

def test_pgvector_performance(n_docs=100):
    """
    测试PgVector性能
    
    Args:
        n_docs: 测试文档数量
    """
    print(f"===== 测试PgVector性能 (添加{n_docs}个文档) =====")
    
    # 强制使用PgVector
    ext_config.VECTOR_DB_TYPE = "pgvector"
    
    # 创建Vanna实例
    print("创建Vanna实例...")
    vn = create_vanna_instance()
    
    # 测试批量添加文档
    print(f"\n测试批量添加{n_docs}个文档...")
    start_time = time.time()
    
    for i in range(n_docs):
        doc_id = vn.train(documentation=f"这是测试文档 {i+1}，用于验证PgVector性能。")
        if (i+1) % 10 == 0:
            print(f"已添加 {i+1}/{n_docs} 个文档")
    
    elapsed = time.time() - start_time
    print(f"批量添加{n_docs}个文档耗时: {elapsed:.2f}秒，平均每个文档: {(elapsed/n_docs)*1000:.2f}毫秒")
    
    # 测试查询性能
    print("\n测试查询性能...")
    start_time = time.time()
    
    for i in range(10):
        docs = vn.get_related_documentation(f"测试文档 {i*10}")
    
    elapsed = time.time() - start_time
    print(f"执行10次查询耗时: {elapsed:.2f}秒，平均每次查询: {(elapsed/10)*1000:.2f}毫秒")
    
    print("\n===== PgVector性能测试完成 =====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试PgVector功能')
    parser.add_argument('--perf', action='store_true', help='运行性能测试')
    parser.add_argument('--ndocs', type=int, default=100, help='性能测试文档数量')
    
    args = parser.parse_args()
    
    if args.perf:
        test_pgvector_performance(args.ndocs)
    else:
        test_pgvector_basic()