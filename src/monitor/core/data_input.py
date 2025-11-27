import cv2
import threading
import queue
import time
from PyQt5.QtCore import QObject, pyqtSignal
import os


class DataInput(QObject):
    """数据输入基类"""
    frame_ready = pyqtSignal(object, object)  # 原始帧, 处理后帧
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.running = False
        self.paused = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=config['ui']['queue_maxsize'])
        # 添加最大帧率控制
        self.max_fps = config['ui'].get('fps', 30)
        self.frame_time = 1.0 / self.max_fps if self.max_fps > 0 else 0

    def start(self):
        """开始数据输入"""
        if not self.running:
            self.running = True
            self.paused = False
            self.thread = threading.Thread(target=self._run)
            self.thread.daemon = True
            self.thread.start()

    def pause(self):
        """暂停数据输入"""
        self.paused = not self.paused

    def stop(self):
        """停止数据输入"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)  # 设置超时避免无限等待

    def _run(self):
        """运行数据输入线程"""
        pass

    def _put_frame(self, original_frame, processed_frame):
        """将帧放入队列"""
        try:
            if not self.paused and self.running:
                # 如果队列满了，移除最旧的帧
                while not self.frame_queue.empty() and self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                self.frame_queue.put((original_frame, processed_frame))
                self.frame_ready.emit(original_frame, processed_frame)
        except Exception as e:
            self.error_occurred.emit(f"数据输入错误: {str(e)}")


class ImageInput(DataInput):
    """图片输入类"""

    def __init__(self, config):
        super().__init__(config)
        self.image_path = None

    def set_image_path(self, path):
        """设置图片路径"""
        self.image_path = path

    def _run(self):
        """运行图片输入"""
        try:
            if not self.image_path or not os.path.exists(self.image_path):
                self.error_occurred.emit("图片文件不存在")
                return

            # 读取图片
            frame = cv2.imread(self.image_path)
            if frame is None:
                self.error_occurred.emit("无法读取图片文件")
                return

            # 预处理
            processed_frame = self._preprocess_frame(frame)
            
            # 发送帧
            self._put_frame(frame, processed_frame)
            
            # 图片只需要处理一次
            self.running = False
            self.finished.emit()
            
        except Exception as e:
            self.error_occurred.emit(f"图片输入错误: {str(e)}")

    def _preprocess_frame(self, frame):
        """预处理帧"""
        # 调整尺寸
        input_size = self.config['model']['input_size']
        processed = cv2.resize(frame, (input_size[0], input_size[1]))
        return processed


class CameraInput(DataInput):
    """摄像头输入类"""

    def __init__(self, config):
        super().__init__(config)
        self.camera_id = 0
        self.cap = None

    def set_camera_id(self, camera_id):
        """设置摄像头ID"""
        self.camera_id = camera_id

    def _run(self):
        """运行摄像头输入"""
        try:
            # 打开摄像头
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.error_occurred.emit("无法打开摄像头")
                return

            # 设置摄像头参数以提高稳定性
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区大小
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率为30fps
            
            last_frame_time = time.time()
            
            while self.running:
                if not self.paused:
                    current_time = time.time()
                    # 控制帧率
                    if (current_time - last_frame_time) >= self.frame_time:
                        ret, frame = self.cap.read()
                        if not ret:
                            self.error_occurred.emit("无法读取摄像头帧")
                            break

                        # 预处理
                        processed_frame = self._preprocess_frame(frame)
                        
                        # 发送帧
                        self._put_frame(frame.copy(), processed_frame)
                        last_frame_time = current_time

                    # 短暂休眠以减少CPU使用
                    time.sleep(0.001)
                else:
                    # 暂停时短暂休眠以减少CPU使用
                    time.sleep(0.01)

        except Exception as e:
            self.error_occurred.emit(f"摄像头输入错误: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
            self.finished.emit()

    def _preprocess_frame(self, frame):
        """预处理帧"""
        # 调整尺寸
        input_size = self.config['model']['input_size']
        processed = cv2.resize(frame, (input_size[0], input_size[1]))
        return processed


class VideoInput(DataInput):
    """视频输入类"""

    def __init__(self, config):
        super().__init__(config)
        self.video_path = None
        self.cap = None

    def set_video_path(self, path):
        """设置视频路径"""
        self.video_path = path

    def _run(self):
        """运行视频输入"""
        try:
            if not self.video_path or not os.path.exists(self.video_path):
                self.error_occurred.emit("视频文件不存在")
                return

            # 打开视频
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.error_occurred.emit("无法打开视频文件")
                return

            # 获取视频帧率
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = self.config['ui']['fps']
            delay = 1.0 / fps

            last_frame_time = time.time()
            
            while self.running:
                if not self.paused:
                    current_time = time.time()
                    # 控制帧率
                    if (current_time - last_frame_time) >= delay:
                        ret, frame = self.cap.read()
                        if not ret:
                            # 视频结束
                            break

                        # 预处理
                        processed_frame = self._preprocess_frame(frame)
                        
                        # 发送帧
                        self._put_frame(frame.copy(), processed_frame)
                        last_frame_time = current_time

                    # 短暂休眠以减少CPU使用
                    time.sleep(0.001)
                else:
                    # 暂停时短暂休眠以减少CPU使用
                    time.sleep(0.01)

        except Exception as e:
            self.error_occurred.emit(f"视频输入错误: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
            self.finished.emit()

    def _preprocess_frame(self, frame):
        """预处理帧"""
        # 调整尺寸
        input_size = self.config['model']['input_size']
        processed = cv2.resize(frame, (input_size[0], input_size[1]))
        return processed