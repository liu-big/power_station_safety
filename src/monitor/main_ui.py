import sys
import os
import yaml
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.uic import loadUi
import numpy as np
import time

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.data_input import ImageInput, CameraInput, VideoInput
from core.model_infer import YoloInfer
from core.result_display import ResultDisplay
from core.data_storage import SqliteStorage


class SafetyMonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 加载配置
        self.config = self.load_config()
        
        # 加载UI
        ui_path = os.path.join(os.path.dirname(__file__), 'ui', 'main_window.ui')
        loadUi(ui_path, self)
        
        # 初始化各模块
        self.init_modules()
        
        # 连接信号槽
        self.connect_signals()
        
        # 设置初始状态
        self.set_initial_state()
        
        # 清理旧数据库记录
        self.storage.clean_old_records()
        
        # 添加性能监控定时器
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self.update_performance_info)
        self.performance_timer.start(1000)  # 每秒更新一次
        
        # 性能统计变量
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0
        self.avg_inference_time = 0
        self.inference_times = []
        
    def load_config(self):
        """加载配置文件"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
            # 返回默认配置
            return {
                'model': {
                    'path': 'powerplant_safety_detection/yolov8n_experiment/weights/best.pt',
                    'confidence_threshold': 0.6,
                    'input_size': [640, 640]
                },
                'classes': {
                    0: "fire",
                    1: "hardhat",
                    2: "no-hardhat",
                    3: "safety-vest",
                    4: "no-safety-vest"
                },
                'chinese_classes': {
                    "fire": "明火",
                    "hardhat": "戴安全帽",
                    "no-hardhat": "未戴安全帽",
                    "safety-vest": "合规工作服",
                    "no-safety-vest": "不合规工作服"
                },
                'risk_levels': {
                    "fire": "紧急",
                    "no-hardhat": "高风险",
                    "no-safety-vest": "中风险",
                    "hardhat": "安全",
                    "safety-vest": "安全"
                },
                'database': {
                    'path': 'safety_monitor.db',
                    'retention_days': 30
                },
                'ui': {
                    'queue_maxsize': 10,
                    'fps': 30
                }
            }
    
    def init_modules(self):
        """初始化各功能模块"""
        # 数据输入模块
        self.image_input = ImageInput(self.config)
        self.camera_input = CameraInput(self.config)
        self.video_input = VideoInput(self.config)
        
        # 模型推理模块
        self.model_infer = YoloInfer(self.config)
        if not self.model_infer.load_model():
            QMessageBox.critical(self, "错误", "模型加载失败，请检查模型路径配置")
        
        # 结果展示模块
        self.result_display = ResultDisplay(self.config)
        
        # 数据存储模块
        self.storage = SqliteStorage(self.config)
        
        # 当前输入源
        self.current_input = None
        
    def connect_signals(self):
        """连接信号槽"""
        # 输入源选择按钮
        self.btn_image.clicked.connect(self.on_image_selected)
        self.btn_camera.clicked.connect(self.on_camera_selected)
        self.btn_video.clicked.connect(self.on_video_selected)
        
        # 控制按钮
        self.btn_start.clicked.connect(self.on_start_clicked)
        self.btn_pause.clicked.connect(self.on_pause_clicked)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        self.btn_history.clicked.connect(self.on_history_clicked)
        
        # 置信度滑块
        self.slider_confidence.valueChanged.connect(self.on_confidence_changed)
        
        # 数据输入信号
        self.image_input.frame_ready.connect(self.on_frame_ready)
        self.camera_input.frame_ready.connect(self.on_frame_ready)
        self.video_input.frame_ready.connect(self.on_frame_ready)
        self.image_input.error_occurred.connect(self.on_input_error)
        self.camera_input.error_occurred.connect(self.on_input_error)
        self.video_input.error_occurred.connect(self.on_input_error)
        
        # 模型推理信号
        self.model_infer.inference_finished.connect(self.on_inference_finished)
        self.model_infer.error_occurred.connect(self.on_inference_error)
        
        # 结果展示信号
        self.result_display.alert_triggered.connect(self.on_alert_triggered)
        
        # 数据存储信号
        self.storage.error_occurred.connect(self.on_storage_error)
        
    def set_initial_state(self):
        """设置初始状态"""
        # 默认选中本地图片
        self.btn_image.setChecked(True)
        
        # 设置置信度滑块初始值
        initial_threshold = int(self.config['model']['confidence_threshold'] * 100)
        self.slider_confidence.setValue(initial_threshold)
        self.label_confidence_value.setText(f"{initial_threshold/100:.2f}")
        
        # 设置按钮初始状态
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        
    @pyqtSlot()
    def on_image_selected(self):
        """选择本地图片"""
        if self.current_input:
            self.current_input.stop()
            
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", "", 
            "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            self.image_input.set_image_path(file_path)
            self.current_input = self.image_input
            self.label_alert.setText("状态：已选择图片文件")
        else:
            self.btn_image.setChecked(False)
    
    @pyqtSlot()
    def on_camera_selected(self):
        """选择摄像头"""
        if self.current_input:
            self.current_input.stop()
            
        # 默认使用摄像头0
        self.camera_input.set_camera_id(0)
        self.current_input = self.camera_input
        self.label_alert.setText("状态：已选择摄像头")
    
    @pyqtSlot()
    def on_video_selected(self):
        """选择本地视频"""
        if self.current_input:
            self.current_input.stop()
            
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", 
            "视频文件 (*.mp4 *.avi *.mkv)"
        )
        
        if file_path:
            self.video_input.set_video_path(file_path)
            self.current_input = self.video_input
            self.label_alert.setText("状态：已选择视频文件")
        else:
            self.btn_video.setChecked(False)
    
    @pyqtSlot()
    def on_start_clicked(self):
        """开始识别"""
        if not self.current_input:
            QMessageBox.warning(self, "警告", "请先选择输入源")
            return
            
        # 更新按钮状态
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        
        # 禁用输入源选择
        self.btn_image.setEnabled(False)
        self.btn_camera.setEnabled(False)
        self.btn_video.setEnabled(False)
        
        # 重置性能统计
        self.frame_count = 0
        self.last_time = time.time()
        self.inference_times.clear()
        
        # 开始数据输入
        self.current_input.start()
        
        self.label_alert.setText("状态：正在识别中...")
    
    @pyqtSlot()
    def on_pause_clicked(self):
        """暂停识别"""
        if self.current_input:
            self.current_input.pause()
            if self.current_input.paused:
                self.btn_pause.setText("继续识别")
                self.label_alert.setText("状态：已暂停")
            else:
                self.btn_pause.setText("暂停识别")
                self.label_alert.setText("状态：正在识别中...")
    
    @pyqtSlot()
    def on_stop_clicked(self):
        """停止识别"""
        if self.current_input:
            self.current_input.stop()
            
        # 更新按钮状态
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("暂停识别")
        
        # 启用输入源选择
        self.btn_image.setEnabled(True)
        self.btn_camera.setEnabled(True)
        self.btn_video.setEnabled(True)
        
        self.label_alert.setText("状态：已停止")
    
    @pyqtSlot(int)
    def on_confidence_changed(self, value):
        """置信度阈值改变"""
        threshold = value / 100.0
        self.label_confidence_value.setText(f"{threshold:.2f}")
        self.model_infer.set_confidence_threshold(threshold)
    
    @pyqtSlot(object, object)
    def on_frame_ready(self, original_frame, processed_frame):
        """数据帧准备就绪"""
        # 增加帧计数
        self.frame_count += 1
        
        # 显示原始帧
        self.result_display.display_frame(self.display_original, original_frame)
        
        # 执行模型推理
        self.model_infer.infer_single_frame(processed_frame)
    
    @pyqtSlot(dict)
    def on_inference_finished(self, result_data):
        """推理完成"""
        # 记录推理时间用于统计
        self.inference_times.append(result_data['inference_time'])
        if len(self.inference_times) > 100:  # 限制列表长度
            self.inference_times.pop(0)
        
        # 显示标注后的帧
        self.result_display.display_frame(self.display_annotated, result_data['annotated_frame'])
        
        # 更新风险列表
        self.result_display.update_risk_list(self.list_risk, result_data['detections'])
        
        # 触发告警
        sound_enabled = self.checkbox_alarm_sound.isChecked()
        self.result_display.trigger_alert(result_data['detections'], sound_enabled)
        
        # 存储数据
        input_type = self.get_current_input_type()
        self.storage.insert_recognition_record(input_type, result_data['detections'])
        
        # 检查是否有高风险目标需要记录到告警日志
        high_risk_detections = [
            d for d in result_data['detections'] 
            if d['risk_level'] in ["紧急", "高风险", "中风险"]
        ]
        
        for detection in high_risk_detections:
            target_info = f"{detection['chinese_name']} (置信度: {detection['confidence']:.2f})"
            self.storage.insert_alarm_log(detection['risk_level'], target_info)
    
    @pyqtSlot(str)
    def on_input_error(self, error_msg):
        """数据输入错误"""
        QMessageBox.critical(self, "数据输入错误", error_msg)
        self.on_stop_clicked()
    
    @pyqtSlot(str)
    def on_inference_error(self, error_msg):
        """推理错误"""
        QMessageBox.critical(self, "推理错误", error_msg)
    
    @pyqtSlot(str)
    def on_storage_error(self, error_msg):
        """数据存储错误"""
        print(f"数据存储错误: {error_msg}")
    
    @pyqtSlot(str, str)
    def on_alert_triggered(self, risk_level, message):
        """告警触发"""
        self.result_display.update_alert_display(self.label_alert, risk_level, message)
    
    def get_current_input_type(self):
        """获取当前输入源类型"""
        if self.btn_image.isChecked():
            return "图片"
        elif self.btn_camera.isChecked():
            return "摄像头"
        elif self.btn_video.isChecked():
            return "视频"
        else:
            return "未知"
    
    @pyqtSlot()
    def on_history_clicked(self):
        """查询历史记录"""
        QMessageBox.information(self, "功能提示", "历史记录查询功能将在后续版本中实现")
    
    def update_performance_info(self):
        """更新性能信息"""
        current_time = time.time()
        elapsed_time = current_time - self.last_time
        
        if elapsed_time >= 1.0:  # 每秒计算一次FPS
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.last_time = current_time
            
            # 计算平均推理时间
            if self.inference_times:
                self.avg_inference_time = sum(self.inference_times) / len(self.inference_times)
            
            # 更新状态栏显示FPS和推理时间
            status_text = f"FPS: {self.fps:.1f}"
            if self.avg_inference_time > 0:
                status_text += f" | 平均推理时间: {self.avg_inference_time*1000:.1f}ms"
            
            self.statusBar().showMessage(status_text)
    
    def closeEvent(self, event):
        """关闭事件"""
        # 停止所有输入源
        if self.current_input:
            self.current_input.stop()
            
        # 停止性能监控定时器
        if self.performance_timer.isActive():
            self.performance_timer.stop()
            
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = SafetyMonitorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()