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
    
    def query_records(self, start_time=None, end_time=None, risk_level=None, limit=100):
        """查询识别记录"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 构建查询语句
                query = "SELECT * FROM recognition_records WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                    
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                    
                if risk_level:
                    query += " AND risk_level = ?"
                    params.append(risk_level)
                    
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                records = cursor.fetchall()
                
                conn.close()
                return records
                
        except Exception as e:
            self.error_occurred.emit(f"查询识别记录失败: {str(e)}")
            return []
    
    def query_alarms(self, start_time=None, end_time=None, risk_level=None, handle_status=None, limit=100):
        """查询告警日志"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 构建查询语句
                query = "SELECT * FROM alarm_logs WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                    
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                    
                if risk_level:
                    query += " AND risk_level = ?"
                    params.append(risk_level)
                    
                if handle_status:
                    query += " AND handle_status = ?"
                    params.append(handle_status)
                    
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                alarms = cursor.fetchall()
                
                conn.close()
                return alarms
                
        except Exception as e:
            self.error_occurred.emit(f"查询告警日志失败: {str(e)}")
            return []
    
    def clean_old_records(self):
        """清理旧记录"""
        try:
            # 计算保留日期
            retention_days = self.config['database']['retention_days']
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            cutoff_date_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
            
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 删除旧的识别记录
                cursor.execute(
                    "DELETE FROM recognition_records WHERE timestamp < ?", 
                    (cutoff_date_str,)
                )
                deleted_records = cursor.rowcount
                
                # 删除旧的告警日志
                cursor.execute(
                    "DELETE FROM alarm_logs WHERE timestamp < ?", 
                    (cutoff_date_str,)
                )
                deleted_alarms = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                print(f"清理了 {deleted_records} 条识别记录和 {deleted_alarms} 条告警日志")
                
        except Exception as e:
            self.error_occurred.emit(f"清理旧记录失败: {str(e)}")