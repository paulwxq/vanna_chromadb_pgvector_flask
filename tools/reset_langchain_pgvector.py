# reset_langchain_pgvector.py
"""
用于重置LangChain PGVector数据库表的脚本
会删除或清空langchain_pg_collection和langchain_pg_embedding表
如果表不存在，则创建表，并确保collection_id字段为UUID类型
"""

import sys
import os
import psycopg2
import argparse

# 添加父目录到路径，确保能正确导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置文件
import ext_config as app_config

def reset_langchain_pgvector(host=None, port=None, dbname=None, user=None, password=None, dimension=1536, confirm=False):
    """
    重置LangChain PGVector数据库表
    
    Args:
        host: 数据库主机
        port: 数据库端口
        dbname: 数据库名称
        user: 数据库用户
        password: 数据库密码
        dimension: 向量维度
        confirm: 是否已确认操作
    """
    # 使用参数或从配置文件获取数据库连接信息
    pgvector_host = host or app_config.PGVECTOR_HOST
    pgvector_port = port or app_config.PGVECTOR_PORT
    pgvector_db = dbname or app_config.PGVECTOR_DB
    pgvector_user = user or app_config.PGVECTOR_USER
    pgvector_password = password or app_config.PGVECTOR_PASSWORD
    
    # 如果未确认且不是交互式运行，提示确认
    if not confirm:
        print(f"警告: 此操作将删除或清空langchain_pg_collection和langchain_pg_embedding表！")
        print(f"数据库连接信息: {pgvector_host}:{pgvector_port}/{pgvector_db}")
        confirm_input = input("确认继续？(y/N): ")
        if confirm_input.lower() not in ['y', 'yes']:
            print("操作已取消")
            return False
    
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
            print("vector扩展安装成功")
        else:
            print("✅ vector扩展已安装")
        
        # 处理langchain_pg_collection表
        cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'langchain_pg_collection')")
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            print("表langchain_pg_collection已存在，正在清空数据...")
            cursor.execute("TRUNCATE TABLE langchain_pg_collection CASCADE")
            print("✅ 表langchain_pg_collection已清空")
        else:
            print("创建langchain_pg_collection表...")
            cursor.execute("""
            CREATE TABLE langchain_pg_collection (
                uuid UUID PRIMARY KEY,
                name VARCHAR(50),
                cmetadata JSONB
            )
            """)
            print("✅ 表langchain_pg_collection已创建")
        
        # 处理langchain_pg_embedding表
        cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'langchain_pg_embedding')")
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            print("表langchain_pg_embedding已存在，正在清空数据...")
            cursor.execute("TRUNCATE TABLE langchain_pg_embedding")
            
            # 检查collection_id字段类型并修正
            cursor.execute("""
            SELECT data_type FROM information_schema.columns 
            WHERE table_name = 'langchain_pg_embedding' AND column_name = 'collection_id'
            """)
            col_type = cursor.fetchone()[0]
            
            if col_type.lower() != 'uuid':
                print(f"检测到collection_id字段类型为{col_type}，正在修改为UUID类型...")
                # 先删除依赖此表的约束
                cursor.execute("ALTER TABLE langchain_pg_embedding DROP CONSTRAINT IF EXISTS langchain_pg_embedding_collection_id_fkey")
                # 修改字段类型
                cursor.execute("ALTER TABLE langchain_pg_embedding ALTER COLUMN collection_id TYPE UUID USING collection_id::uuid")
                print("✅ collection_id字段已修改为UUID类型")
            else:
                print("✅ collection_id字段已是UUID类型")
                
            print("✅ 表langchain_pg_embedding已清空")
        else:
            print("创建langchain_pg_embedding表...")
            cursor.execute(f"""
            CREATE TABLE langchain_pg_embedding (
                uuid UUID PRIMARY KEY,
                collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
                document TEXT,
                embedding VECTOR({dimension}),
                cmetadata JSONB
            )
            """)
            print(f"✅ 表langchain_pg_embedding已创建，支持{dimension}维向量")
        
        # 创建索引
        print("创建向量索引...")
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS langchain_pg_embedding_embedding_idx 
        ON langchain_pg_embedding 
        USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100)
        """)
        print("✅ 已创建向量索引")
        
        # 关闭连接
        cursor.close()
        conn.close()
        print("✅ LangChain PGVector表重置完成")
        return True
        
    except Exception as e:
        print(f"❌ 重置数据库时出错: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='重置LangChain PGVector数据库表')
    parser.add_argument('--host', type=str, help='数据库主机')
    parser.add_argument('--port', type=int, help='数据库端口')
    parser.add_argument('--dbname', type=str, help='数据库名称')
    parser.add_argument('--user', type=str, help='数据库用户')
    parser.add_argument('--password', type=str, help='数据库密码')
    parser.add_argument('--dimension', type=int, default=1536, help='向量维度 (默认: 1536)')
    parser.add_argument('--force', action='store_true', help='强制执行，不提示确认')
    
    args = parser.parse_args()
    
    reset_langchain_pgvector(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
        dimension=args.dimension,
        confirm=args.force
    ) 