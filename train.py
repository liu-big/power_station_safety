import os
import yaml
from ultralytics import YOLO
import torch

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def train_model():
    """训练模型主函数"""
    print("开始训练电力设施安全检测模型...")
    
    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载训练配置
    config = load_config('train_config.yaml')
    print("配置文件加载成功")
    
    # 更新设备配置
    config['device'] = device
    
    try:
        # 创建模型
        model = YOLO('yolov8n_powerplant.yaml')
        print("模型创建成功")
        
        # 或者可以加载预训练模型
        # model = YOLO('yolov8n.pt')
        
        # 开始训练
        print("开始训练...")
        model.train(**config)
        
        print("训练完成!")
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        raise e

def resume_training():
    """恢复训练"""
    print("恢复训练...")
    
    # 加载训练配置
    config = load_config('train_config.yaml')
    
    # 设置恢复训练参数
    config['resume'] = True
    
    try:
        # 加载已有模型
        model = YOLO('runs/detect/train/weights/last.pt')
        print("模型加载成功")
        
        # 恢复训练
        print("恢复训练...")
        model.train(**config)
        
        print("训练完成!")
        
    except Exception as e:
        print(f"恢复训练过程中出现错误: {str(e)}")
        raise e

if __name__ == "__main__":
    # 检查是否存在之前的训练权重来决定是全新训练还是恢复训练
    if os.path.exists('runs/detect/train/weights/last.pt'):
        choice = input("检测到之前训练的权重，是否恢复训练？(y/n): ")
        if choice.lower() == 'y':
            resume_training()
        else:
            train_model()
    else:
        train_model()