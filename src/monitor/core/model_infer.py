import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtCore import QObject, pyqtSignal
import time


class YoloInfer(QObject):
    """YOLO模型推理类"""
    inference_finished = pyqtSignal(dict)  # 推理完成信号
    error_occurred = pyqtSignal(str)       # 错误信号

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = None
        self.confidence_threshold = config['model']['confidence_threshold']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classes = config['classes']
        self.chinese_classes = config['chinese_classes']
        self.risk_levels = config['risk_levels']
        # 添加推理超时设置（秒）
        self.inference_timeout = 5.0

    def load_model(self):
        """加载模型"""
        try:
            model_path = self.config['model']['path']
            self.model = YOLO(model_path)
            print(f"模型加载成功，使用设备: {self.device}")
            return True
        except Exception as e:
            self.error_occurred.emit(f"模型加载失败: {str(e)}")
            return False

    def set_confidence_threshold(self, threshold):
        """设置置信度阈值"""
        self.confidence_threshold = threshold

    def infer_single_frame(self, frame):
        """单帧推理"""
        try:
            if self.model is None:
                self.error_occurred.emit("模型未加载")
                return None

            # 记录开始时间
            start_time = time.time()
            
            # 执行推理，添加超时机制
            results = self.model(
                frame, 
                conf=self.confidence_threshold, 
                device=self.device,
                verbose=False  # 减少日志输出
            )
            
            # 检查是否超时
            inference_time = time.time() - start_time
            if inference_time > self.inference_timeout:
                print(f"警告: 推理时间过长 {inference_time:.2f}秒")
            
            # 解析结果
            result_data = self._parse_results(frame, results, inference_time)
            
            # 发送结果信号
            self.inference_finished.emit(result_data)
            
            return result_data
            
        except Exception as e:
            self.error_occurred.emit(f"推理错误: {str(e)}")
            return None

    def _parse_results(self, frame, results, inference_time):
        """解析推理结果"""
        # 获取第一个结果（我们只处理单帧）
        result = results[0]
        
        # 提取边界框信息
        boxes = result.boxes
        if boxes is None:
            # 没有检测到目标
            return {
                'frame': frame,
                'annotated_frame': frame.copy(),
                'detections': [],
                'inference_time': inference_time
            }
        
        # 提取边界框坐标、置信度和类别
        box_data = boxes.data.cpu().numpy()
        detections = []
        
        # 创建标注图像副本
        annotated_frame = frame.copy()
        
        # 颜色定义（BGR格式）
        colors = {
            '紧急': (0, 0, 255),    # 红色
            '高风险': (0, 255, 255), # 黄色
            '中风险': (0, 165, 255), # 橙色
            '安全': (0, 255, 0)     # 绿色
        }
        
        for box in box_data:
            x1, y1, x2, y2, conf, cls_id = box
            cls_id = int(cls_id)
            
            # 获取类别名称
            if cls_id in self.classes:
                class_name = self.classes[cls_id]
                chinese_name = self.chinese_classes.get(class_name, class_name)
                risk_level = self.risk_levels.get(class_name, "未知")
            else:
                class_name = "未知"
                chinese_name = "未知"
                risk_level = "未知"
            
            # 转换坐标为整数
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 添加到检测结果列表
            detection = {
                'bbox': (x1, y1, x2, y2),
                'confidence': float(conf),
                'class_id': cls_id,
                'class_name': class_name,
                'chinese_name': chinese_name,
                'risk_level': risk_level
            }
            detections.append(detection)
            
            # 在图像上绘制边界框
            color = colors.get(risk_level, (255, 255, 255))  # 默认白色
            
            # 绘制边界框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{chinese_name} {conf:.2f}"
            
            # 使用 Hershey fonts 字体系列中的 FONT_HERSHEY_SIMPLEX 字体
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            
            # 获取文本大小
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 确保标签不会超出图像边界
            label_x = x1
            label_y = max(y1 - 10, text_height + 10)
            
            # 绘制标签背景，增加一点填充使文本更清晰
            cv2.rectangle(annotated_frame, 
                         (label_x, label_y - text_height - 5), 
                         (label_x + text_width, label_y + 5), 
                         color, -1)
            
            # 绘制标签文字，使用 LINE_AA 抗锯齿
            cv2.putText(annotated_frame, label, 
                       (label_x, label_y), 
                       font, 
                       font_scale, 
                       (255, 255, 255), 
                       thickness, 
                       cv2.LINE_AA)
        
        return {
            'frame': frame,
            'annotated_frame': annotated_frame,
            'detections': detections,
            'inference_time': inference_time
        }