import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtCore import QObject, pyqtSignal
import time
from PIL import Image, ImageDraw, ImageFont


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
        # 添加推理缓存以提高重复帧的处理速度
        self.last_frame_hash = None
        self.last_result = None

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

            # 计算帧的哈希值，用于缓存优化
            frame_hash = hash(frame.tobytes())
            
            # 如果是同一帧，直接返回缓存结果
            if frame_hash == self.last_frame_hash and self.last_result is not None:
                self.inference_finished.emit(self.last_result)
                return self.last_result

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
            
            # 缓存结果
            self.last_frame_hash = frame_hash
            self.last_result = result_data
            
            # 发送结果信号
            self.inference_finished.emit(result_data)
            
            return result_data
            
        except Exception as e:
            self.error_occurred.emit(f"推理错误: {str(e)}")
            return None

    def _put_chinese_text(self, img, text, pos, font_size=20, color=(255, 255, 255)):
        """在图像上绘制中文文本"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 尝试使用几种常见的中文字体
        font_paths = [
            "simhei.ttf",  # 黑体
            "simsun.ttc",  # 宋体
            "msyh.ttc",    # 微软雅黑
            "arialuni.ttf" # Arial Unicode
        ]
        
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        
        # 如果找不到字体文件，使用默认字体
        if font is None:
            font = ImageFont.load_default()
        
        # 绘制文本
        draw.text(pos, text, font=font, fill=color)
        
        # 转换回OpenCV格式
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_cv

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
            
            # 使用支持中文的函数绘制标签
            try:
                # 绘制标签背景
                ((text_width, text_height), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - 30), (x1 + text_width, y1), color, -1)
                
                # 绘制中文标签
                annotated_frame = self._put_chinese_text(
                    annotated_frame, 
                    label, 
                    (x1, y1 - 25), 
                    font_size=18, 
                    color=(255, 255, 255)  # 白色文字
                )
            except Exception as e:
                # 如果中文绘制失败，使用备用方案
                cv2.putText(annotated_frame, label, 
                           (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, 
                           (255, 255, 255), 
                           1, 
                           cv2.LINE_AA)
        
        return {
            'frame': frame,
            'annotated_frame': annotated_frame,
            'detections': detections,
            'inference_time': inference_time
        }