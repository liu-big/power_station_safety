import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import winsound
import time


class ResultDisplay(QObject):
    """结果展示类"""
    alert_triggered = pyqtSignal(str, str)  # 风险等级, 描述
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.enable_sound_alarm = True
        self.alert_timer = None
        self.blink_timers = []  # 用于跟踪所有闪烁定时器
        # 添加声音告警冷却时间，避免过于频繁的告警声
        self.last_sound_time = 0
        self.sound_cooldown = 1.0  # 1秒冷却时间
        
    def display_frame(self, label, frame):
        """在 QLabel 上显示图像帧"""
        if frame is None:
            return
            
        try:
            # 转换颜色空间 BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 创建 QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 缩放图像以适应 QLabel
            pixmap = QPixmap.fromImage(q_img)
            label.setPixmap(pixmap.scaled(
                label.width(), 
                label.height(), 
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
        except Exception as e:
            print(f"显示帧错误: {str(e)}")
    
    def update_risk_list(self, list_widget, detections):
        """更新风险列表"""
        try:
            current_time = time.strftime("%H:%M:%S")
            
            for detection in detections:
                class_name = detection['chinese_name']
                confidence = detection['confidence']
                risk_level = detection['risk_level']
                
                # 创建列表项文本
                item_text = f"[{current_time}] {class_name} (置信度: {confidence:.2f}) - {risk_level}"
                
                # 添加到列表顶部
                list_widget.insertItem(0, item_text)
                
                # 限制列表项数量，防止过多占用内存
                if list_widget.count() > 50:  # 减少列表项数量以提高性能
                    list_widget.takeItem(list_widget.count() - 1)
                
                # 设置风险级别样式
                item = list_widget.item(0)
                if risk_level == "紧急":
                    item.setBackground(Qt.red)
                    item.setForeground(Qt.white)
                elif risk_level == "高风险":
                    item.setBackground(Qt.yellow)
                elif risk_level == "中风险":
                    item.setBackground(Qt.darkYellow)
                    item.setForeground(Qt.white)
                    
        except Exception as e:
            print(f"更新风险列表错误: {str(e)}")
    
    def trigger_alert(self, detections, sound_enabled):
        """触发告警"""
        try:
            # 检查是否有需要告警的目标
            high_risk_detections = [
                d for d in detections 
                if d['risk_level'] in ["紧急", "高风险", "中风险"]
            ]
            
            if not high_risk_detections:
                return
                
            # 找到最高风险等级
            risk_order = {"紧急": 3, "高风险": 2, "中风险": 1}
            highest_risk = max(high_risk_detections, 
                             key=lambda x: risk_order.get(x['risk_level'], 0))
            risk_level = highest_risk['risk_level']
            class_name = highest_risk['chinese_name']
            
            # 发送告警信号
            alert_msg = f"检测到 {class_name}，风险等级: {risk_level}"
            self.alert_triggered.emit(risk_level, alert_msg)
            
            # 声音告警（带冷却时间限制）
            current_time = time.time()
            if sound_enabled and (current_time - self.last_sound_time) >= self.sound_cooldown:
                self._play_sound_alert(risk_level)
                self.last_sound_time = current_time
                
        except Exception as e:
            print(f"触发告警错误: {str(e)}")
    
    def _play_sound_alert(self, risk_level):
        """播放声音告警"""
        try:
            if risk_level == "紧急":
                # 紧急告警 - 高频蜂鸣
                winsound.Beep(1000, 300)  # 缩短持续时间以提高响应性
            elif risk_level == "高风险":
                # 高风险告警 - 中频蜂鸣
                winsound.Beep(800, 200)
            elif risk_level == "中风险":
                # 中风险告警 - 低频蜂鸣
                winsound.Beep(600, 100)
        except Exception as e:
            print(f"播放声音告警错误: {str(e)}. 这可能是由于系统不支持或没有音频设备导致的，但不影响主要功能。")
    
    def update_alert_display(self, label, risk_level, message):
        """更新告警显示"""
        try:
            # 设置文本
            label.setText(message)
            
            # 根据风险等级设置背景色
            if risk_level == "紧急":
                # 红色闪烁效果通过定时器实现
                label.setStyleSheet("background-color: red; color: white; padding: 5px;")
                self._start_blinking_effect(label, "red")
            elif risk_level == "高风险":
                # 黄色背景
                label.setStyleSheet("background-color: yellow; color: black; padding: 5px;")
            elif risk_level == "中风险":
                # 橙色背景
                label.setStyleSheet("background-color: orange; color: black; padding: 5px;")
            else:
                # 正常状态
                label.setStyleSheet("background-color: lightgray; color: black; padding: 5px;")
                
        except Exception as e:
            print(f"更新告警显示错误: {str(e)}")
    
    def _start_blinking_effect(self, label, color):
        """开始闪烁效果"""
        # 停止之前的闪烁定时器
        self._stop_all_blinking()
        
        # 创建新的定时器
        blink_timer = QTimer()
        blink_timer.timeout.connect(lambda: self._toggle_blink(label, color))
        blink_timer.start(500)  # 500ms切换一次
        
        # 保存定时器引用
        self.blink_timers.append(blink_timer)
        
        # 3秒后停止闪烁
        stop_timer = QTimer()
        stop_timer.timeout.connect(lambda: self._stop_blinking(label, color))
        stop_timer.setSingleShot(True)
        stop_timer.start(3000)
        self.blink_timers.append(stop_timer)
    
    def _toggle_blink(self, label, color):
        """切换闪烁状态"""
        current_style = label.styleSheet()
        if "white" in current_style or color not in current_style:
            # 显示彩色背景
            label.setStyleSheet(f"background-color: {color}; color: white; padding: 5px;")
        else:
            # 显示白色背景
            label.setStyleSheet("background-color: white; color: black; padding: 5px;")
    
    def _stop_blinking(self, label, color):
        """停止闪烁并恢复原始颜色"""
        # 停止所有闪烁定时器
        self._stop_all_blinking()
        # 恢复红色背景
        label.setStyleSheet("background-color: red; color: white; padding: 5px;")
    
    def _stop_all_blinking(self):
        """停止所有闪烁效果"""
        for timer in self.blink_timers:
            if timer and timer.isActive():
                timer.stop()
        self.blink_timers.clear()