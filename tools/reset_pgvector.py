# reset_pgvector.py
"""
用于重置pgvector数据库表的脚本
会删除现有的vanna_pgvector表并创建新表
"""

import sys
import os
import psycopg2
# from dotenv import load_dotenv # 不再使用dotenv

# 添加父目录到路径，确保能正确导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置文件
import config as app_config # 使用别名

# 导入正确的模块 (vn实例可能不需要了，因为我们直接用配置)
# from vanna_pgvector_qwen import vn # 如果vn实例在这里没有其他用途，可以注释掉

# 不再加载环境变量
# load_dotenv()

def reset_pgvector_database():
    """
    重置PgVector数据库，确保向量表可以存储1024维向量（针对BGE-M3模型）
    """
    # 从配置文件获取数据库连接信息
    pgvector_host = app_config.PGVECTOR_HOST
    pgvector_port = app_config.PGVECTOR_PORT
    pgvector_db = app_config.PGVECTOR_DB
    pgvector_user = app_config.PGVECTOR_USER
    pgvector_password = app_config.PGVECTOR_PASSWORD
    pgvector_table = app_config.PGVECTOR_TABLE
    
    # 连接到PostgreSQL
    try:
        print(f"正在连接到PgVector数据库: {pgvector_host}:{pgvector_port}/{pgvector_db}")
        conn = psycopg2.connect(
            host=pgvector_host,
            port=pgvector_port,
            dbname=pgvector_db,
            user=pgvector_user,
            password=pgvector_password
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # 检查vector扩展是否安装
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname='vector'")
        if cursor.fetchone() is None:
            print("安装vector扩展...")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            print("✅ vector扩展安装成功")
        else:
            print("✅ vector扩展已安装")
        
        # 删除现有的表（如果存在）
        cursor.execute(f"DROP TABLE IF EXISTS {pgvector_table}")
        print(f"✅ 已删除现有表（如果存在）: {pgvector_table}")
        
        # 创建新表，使用1024维向量（适用于BGE-M3模型）
        cursor.execute(f"""
        CREATE TABLE {pgvector_table} (
            id SERIAL PRIMARY KEY,
            type TEXT,
            content TEXT,
            embedding VECTOR(1024)
        )
        """)
        print(f"✅ 已创建新表 {pgvector_table}，支持1024维向量（BGE-M3模型）")
        
        # 关闭连接
        cursor.close()
        conn.close()
        print("✅ 数据库重置完成")
        
    except Exception as e:
        print(f"❌ 重置数据库时出错: {e}")
        raise

if __name__ == "__main__":
    reset_pgvector_database() 