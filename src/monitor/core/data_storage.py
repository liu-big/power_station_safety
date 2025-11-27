import sqlite3
import threading
import os
import time
from datetime import datetime, timedelta
from PyQt5.QtCore import QObject, pyqtSignal


class SqliteStorage(QObject):
    """SQLite数据存储类"""
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.db_path = config['database']['path']
        self.lock = threading.Lock()
        self.init_db()
        
    def init_db(self):
        """初始化数据库"""
        try:
            # 确保数据库目录存在
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 创建识别记录表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS recognition_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        input_type TEXT NOT NULL,
                        target_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        risk_level TEXT NOT NULL,
                        image_path TEXT
                    )
                ''')
                
                # 创建告警日志表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alarm_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        target_info TEXT NOT NULL,
                        handle_status TEXT DEFAULT '未处理'
                    )
                ''')
                
                conn.commit()
                conn.close()
                print("数据库初始化成功")
                
        except Exception as e:
            self.error_occurred.emit(f"数据库初始化失败: {str(e)}")
    
    def insert_recognition_record(self, input_type, detections, image_path=None):
        """插入识别记录"""
        try:
            # 确保表存在
            self._ensure_tables_exist()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                for detection in detections:
                    cursor.execute('''
                        INSERT INTO recognition_records 
                        (timestamp, input_type, target_type, confidence, risk_level, image_path)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp,
                        input_type,
                        detection['class_name'],
                        detection['confidence'],
                        detection['risk_level'],
                        image_path
                    ))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            self.error_occurred.emit(f"插入识别记录失败: {str(e)}")
    
    def insert_alarm_log(self, risk_level, target_info):
        """插入告警日志"""
        try:
            # 确保表存在
            self._ensure_tables_exist()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO alarm_logs 
                    (timestamp, risk_level, target_info, handle_status)
                    VALUES (?, ?, ?, ?)
                ''', (
                    timestamp,
                    risk_level,
                    target_info,
                    '未处理'
                ))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            self.error_occurred.emit(f"插入告警日志失败: {str(e)}")
    
    def _ensure_tables_exist(self):
        """确保数据库表存在"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 检查并创建识别记录表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS recognition_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        input_type TEXT NOT NULL,
                        target_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        risk_level TEXT NOT NULL,
                        image_path TEXT
                    )
                ''')
                
                # 检查并创建告警日志表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alarm_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        target_info TEXT NOT NULL,
                        handle_status TEXT DEFAULT '未处理'
                    )
                ''')
                
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"确保表存在时出错: {str(e)}")
    
    def clean_old_records(self):
        """清理过期记录"""
        try:
            retention_days = self.config['database'].get('retention_days', 30)
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
            
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 删除过期的识别记录
                cursor.execute('''
                    DELETE FROM recognition_records WHERE timestamp < ?
                ''', (cutoff_str,))
                records_deleted = cursor.rowcount
                
                # 删除过期的告警日志
                cursor.execute('''
                    DELETE FROM alarm_logs WHERE timestamp < ?
                ''', (cutoff_str,))
                logs_deleted = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                print(f"清理了 {records_deleted} 条识别记录和 {logs_deleted} 条告警日志")
                
        except Exception as e:
            self.error_occurred.emit(f"清理过期记录失败: {str(e)}")
    
    def query_recognition_records(self, limit=100):
        """查询识别记录"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM recognition_records 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                records = cursor.fetchall()
                conn.close()
                
                return records
                
        except Exception as e:
            self.error_occurred.emit(f"查询识别记录失败: {str(e)}")
            return []
    
    def query_alarm_logs(self, limit=100):
        """查询告警日志"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM alarm_logs 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                logs = cursor.fetchall()
                conn.close()
                
                return logs
                
        except Exception as e:
            self.error_occurred.emit(f"查询告警日志失败: {str(e)}")
            return []