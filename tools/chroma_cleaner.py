# chroma_cleaner.py
"""
ChromaDB数据库清理工具
用于清空或重置ChromaDB的数据
"""

import os
import shutil
import time
import argparse
from mychromadb import Client, Settings

def clear_chroma_database(db_path="chroma.sqlite3"):
    """清理Chroma数据库数据
    
    Args:
        db_path (str, optional): 数据库文件路径. 默认为 "chroma.sqlite3".
    
    Returns:
        bool: 清理是否成功
    """
    print("🧹 开始清理Chroma数据库...")
    success = False
    
    # 检查数据库文件是否存在
    if not os.path.exists(db_path):
        print(f"⚠️ 数据库文件不存在: {db_path}")
        return False
    
    # 备份数据库文件
    try:
        backup_path = f"{db_path}.backup"
        if os.path.exists(backup_path):
            os.remove(backup_path)
        shutil.copy2(db_path, backup_path)
        print(f"✅ 数据库已备份到: {backup_path}")
    except Exception as e:
        print(f"⚠️ 备份数据库失败: {e}")
    
    try:
        # 方法1: 使用API重置数据库
        print("\n尝试使用ChromaDB API重置数据库...")
        try:
            settings = Settings(allow_reset=True)
            client = Client(settings=settings)
            client.reset()
            print("✅ 通过API重置成功")
            success = True
        except Exception as e:
            print(f"⚠️ API重置失败: {e}")
        
        # 方法2: 如果API失败，尝试删除数据库文件
        if not success:
            print("\n尝试删除数据库文件...")
            # 尝试删除数据库文件
            max_retries = 3
            for i in range(max_retries):
                try:
                    os.remove(db_path)
                    print(f"✅ 成功删除数据库文件: {db_path}")
                    success = True
                    break
                except Exception as e:
                    if i < max_retries - 1:
                        print(f"⚠️ 删除失败，等待重试... ({i+1}/{max_retries})")
                        time.sleep(2)
                    else:
                        print(f"❌ 无法删除数据库文件: {e}")
                        print("💡 提示: 请确保没有应用程序正在使用此文件")
        
        # 检查清理结果
        if success:
            if os.path.exists(db_path):
                print("⚠️ 数据库文件仍然存在，但内容可能已被清空")
            else:
                print("✅ 数据库文件已被成功删除")
        
    except Exception as e:
        print(f"❌ 清理过程出错: {e}")
        success = False
    
    print("✨ 清理操作完成")
    return success

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='清理ChromaDB数据库')
    parser.add_argument('--path', type=str, default="chroma.sqlite3",
                        help='ChromaDB数据库文件路径 (默认: chroma.sqlite3)')
    parser.add_argument('--force', action='store_true',
                        help='强制清理，不进行确认')
    
    args = parser.parse_args()
    
    # 确认操作
    if not args.force:
        confirm = input(f"确定要清理ChromaDB数据库 '{args.path}'? 此操作不可逆! [y/N]: ")
        if confirm.lower() not in ('y', 'yes'):
            print("已取消操作")
            return
    
    success = clear_chroma_database(args.path)
    if success:
        print("👍 数据库清理成功")
    else:
        print("❌ 数据库清理可能不完全，请手动检查")

if __name__ == "__main__":
    # 直接运行此文件可以清理默认位置的Chroma数据库
    main() 