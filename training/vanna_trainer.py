# vanna_trainer.py
import os
import time
import threading
import queue
import concurrent.futures
from functools import lru_cache
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ext_config
from vanna_factory import create_vanna_instance

vn = create_vanna_instance()

def get_default_embedding_model():
    """获取系统默认的Embedding模型名称"""
    # 从Vanna对象获取
    if hasattr(vn, 'get_default_embedding_model'):
        return vn.get_default_embedding_model()
    # 从配置获取
    elif hasattr(ext_config, 'DEFAULT_EMBEDDING_MODEL'):
        return ext_config.DEFAULT_EMBEDDING_MODEL
    # 最后的备选值
    return "bge-large-zh"

# 使用函数获取默认值
embedding_model = get_default_embedding_model()
print(f"\n===== 当前使用的Embedding模型: {embedding_model} =====")

# 从ext_config获取其他配置
BATCH_PROCESSING_ENABLED = ext_config.BATCH_PROCESSING_ENABLED
BATCH_SIZE = ext_config.BATCH_SIZE
MAX_WORKERS = ext_config.MAX_WORKERS


# 数据批处理器
class BatchProcessor:
    def __init__(self, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.batches = defaultdict(list)
        self.lock = threading.Lock()  # 线程安全锁
        
        # 初始化工作线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # 是否启用批处理
        self.batch_enabled = BATCH_PROCESSING_ENABLED       

        print(f"[DEBUG] 批处理器初始化: 启用={self.batch_enabled}, 批大小={self.batch_size}, 最大工作线程={self.max_workers}")
    
    def add_item(self, batch_type: str, item: Dict[str, Any]):
        """添加一个项目到批处理队列"""
        if not self.batch_enabled:
            # 如果未启用批处理，直接处理
            self._process_single_item(batch_type, item)
            return
        
        with self.lock:
            self.batches[batch_type].append(item)
            
            if len(self.batches[batch_type]) >= self.batch_size:
                batch_items = self.batches[batch_type]
                self.batches[batch_type] = []
                # 提交批处理任务到线程池
                self.executor.submit(self._process_batch, batch_type, batch_items)
    
    def _process_single_item(self, batch_type: str, item: Dict[str, Any]):
        """处理单个项目"""
        try:
            if batch_type == 'ddl':
                vn.train(ddl=item['ddl'])
            elif batch_type == 'documentation':
                vn.train(documentation=item['documentation'])
            elif batch_type == 'question_sql':
                vn.train(question=item['question'], sql=item['sql'])
            
            print(f"[DEBUG] 单项处理成功: {batch_type}")
                
        except Exception as e:
            print(f"[ERROR] 处理 {batch_type} 项目失败: {e}")
    
    def _process_batch(self, batch_type: str, items: List[Dict[str, Any]]):
        """处理一批项目"""
        print(f"[INFO] 开始批量处理 {len(items)} 个 {batch_type} 项")
        start_time = time.time()
        
        try:
            # 准备批处理数据
            batch_data = []
            
            if batch_type == 'ddl':
                for item in items:
                    batch_data.append({
                        'type': 'ddl',
                        'content': item['ddl']
                    })
            
            elif batch_type == 'documentation':
                for item in items:
                    batch_data.append({
                        'type': 'documentation',
                        'content': item['documentation']
                    })
            
            elif batch_type == 'question_sql':
                for item in items:
                    batch_data.append({
                        'type': 'question_sql',
                        'question': item['question'],
                        'sql': item['sql']
                    })
            
            # 使用批量添加方法
            if hasattr(vn, 'add_batch') and callable(getattr(vn, 'add_batch')):
                success = vn.add_batch(batch_data)
                if success:
                    print(f"[INFO] 批量处理成功: {len(items)} 个 {batch_type} 项")
                else:
                    print(f"[WARNING] 批量处理部分失败: {batch_type}")
            else:
                # 如果没有批处理方法，退回到逐条处理
                print(f"[WARNING] 批处理不可用，使用逐条处理: {batch_type}")
                for item in items:
                    self._process_single_item(batch_type, item)
                
        except Exception as e:
            print(f"[ERROR] 批处理 {batch_type} 失败: {e}")
            # 如果批处理失败，尝试逐条处理
            print(f"[INFO] 尝试逐条处理...")
            for item in items:
                try:
                    self._process_single_item(batch_type, item)
                except Exception as item_e:
                    print(f"[ERROR] 处理项目失败: {item_e}")
        
        elapsed = time.time() - start_time
        print(f"[INFO] 批处理完成 {len(items)} 个 {batch_type} 项，耗时 {elapsed:.2f} 秒")
    
    def flush_all(self):
        """强制处理所有剩余项目"""
        with self.lock:
            for batch_type, items in self.batches.items():
                if items:
                    print(f"[INFO] 正在处理剩余的 {len(items)} 个 {batch_type} 项")
                    self._process_batch(batch_type, items)
            
            # 清空队列
            self.batches = defaultdict(list)
        
        print("[INFO] 所有批处理项目已完成")
    
    def shutdown(self):
        """关闭处理器和线程池"""
        self.flush_all()
        self.executor.shutdown(wait=True)
        print("[INFO] 批处理器已关闭")

# 创建全局批处理器实例
batch_processor = BatchProcessor()

# 原始训练函数的批处理增强版本
def train_ddl(ddl_sql: str):
    print(f"[DDL] Training on DDL:\n{ddl_sql}")
    batch_processor.add_item('ddl', {'ddl': ddl_sql})

def train_documentation(doc: str):
    print(f"[DOC] Training on documentation:\n{doc}")
    batch_processor.add_item('documentation', {'documentation': doc})

def train_sql_example(sql: str):
    """训练单个SQL示例，通过SQL生成相应的问题"""
    print(f"[SQL] Training on SQL:\n{sql}")
    
    # 从SQL提取注释信息
    comment_info = None
    try:
        if "--" in sql:
            comment_parts = sql.split("--")
            comment_info = comment_parts[1].split("\n")[0].strip()
    except Exception as e:
        # 如果提取注释失败，不报错，只记录日志
        print(f"[INFO] 提取SQL注释信息时出现问题: {e}")
    
    # 使用大模型生成问题
    try:
        # 准备提示词
        if comment_info:
            prompt = f"""
根据以下SQL及其注释，生成一个简洁、明确的中文问题，问题应该能够反映SQL的功能或目的。
注释信息: {comment_info}
SQL: {sql}
生成的问题需要是一个问句，以问号结尾。
"""
        else:
            prompt = f"""
根据以下SQL，生成一个简洁、明确的中文问题，问题应该能够反映SQL的功能或目的。
SQL: {sql}
生成的问题需要是一个问句，以问号结尾。
"""
        
        # 使用vn对象调用大模型生成问题
        if hasattr(vn, 'generate_question_for_sql') and callable(getattr(vn, 'generate_question_for_sql')):
            # 如果有专门的方法，使用它
            question = vn.generate_question_for_sql(sql=sql, comment=comment_info)
        elif hasattr(vn, 'llm') and hasattr(vn.llm, 'generate'):
            # 尝试通过llm属性调用生成方法
            question = vn.llm.generate(prompt)
        elif hasattr(vn, 'generate_text'):
            # 尝试使用generate_text方法
            question = vn.generate_text(prompt)
        else:
            # 如果无法调用大模型，使用基于规则的方法生成问题
            print("[INFO] 无法调用大模型生成问题，使用规则方法生成")
            
            # 使用注释作为基本问题
            if comment_info:
                question = comment_info
                if not question.endswith("?"):
                    question += "?"
            else:
                # 使用基于SQL结构的规则生成问题
                if "SELECT" in sql.upper():
                    if "COUNT" in sql.upper():
                        question = "如何统计指定条件的记录数量？"
                    elif "SUM" in sql.upper() or "AVG" in sql.upper():
                        question = "如何计算字段的汇总值？"
                    elif "GROUP BY" in sql.upper():
                        question = "如何按分组统计数据？"
                    elif "JOIN" in sql.upper():
                        question = "如何连接多个表查询数据？"
                    elif "ORDER BY" in sql.upper():
                        question = "如何对查询结果进行排序？"
                    else:
                        question = "如何查询指定条件的数据？"
                elif "INSERT" in sql.upper():
                    question = "如何插入新数据？"
                elif "UPDATE" in sql.upper():
                    question = "如何更新现有数据？"
                elif "DELETE" in sql.upper():
                    question = "如何删除满足条件的数据？"
                elif "CREATE" in sql.upper():
                    question = "如何创建数据库对象？"
                elif "ALTER" in sql.upper():
                    question = "如何修改数据库对象结构？"
                else:
                    question = "如何执行这个SQL语句？"
                    
        # 处理问题格式
        question = question.strip()
        if not question.endswith("?"):
            question += "?"
            
    except Exception as e:
        print(f"[WARNING] 生成问题时出错: {e}")
        # 如果有注释，使用注释作为问题
        if comment_info:
            question = comment_info + "?"
        else:
            question = "如何执行这个SQL语句？"
        
    print(f"[SQL] 生成问题: {question}")
    # 使用标准方式存储问题-SQL对
    batch_processor.add_item('question_sql', {'question': question, 'sql': sql})

def train_question_sql_pair(question: str, sql: str):
    print(f"[Q-S] Training on:\nquestion: {question}\nsql: {sql}")
    batch_processor.add_item('question_sql', {'question': question, 'sql': sql})

# 完成训练后刷新所有待处理项
def flush_training():
    """强制处理所有待处理的训练项目"""
    batch_processor.flush_all()

# 关闭训练器
def shutdown_trainer():
    """关闭训练器和相关资源"""
    batch_processor.shutdown() 