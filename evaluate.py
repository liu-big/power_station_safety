import os
import yaml
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import numpy as np

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def evaluate_model():
    """评估模型性能"""
    print("开始评估电力设施安全检测模型...")
    
    # 检查最佳模型权重是否存在
    best_model_path = 'runs/detect/train/weights/best.pt'
    last_model_path = 'runs/detect/train/weights/last.pt'
    
    if os.path.exists(best_model_path):
        model_path = best_model_path
        print(f"使用最佳模型权重: {model_path}")
    elif os.path.exists(last_model_path):
        model_path = last_model_path
        print(f"使用最后模型权重: {model_path}")
    else:
        print("未找到训练好的模型权重文件")
        return
    
    try:
        # 加载模型
        model = YOLO(model_path)
        print("模型加载成功")
        
        # 加载数据集配置
        data_config = 'dataset/powerplant_safety/data.yaml'
        print(f"使用数据集配置: {data_config}")
        
        # 在验证集上评估
        print("在验证集上评估模型...")
        val_metrics = model.val(data=data_config, split='val')
        print_results("验证集", val_metrics)
        
        # 在测试集上评估
        print("在测试集上评估模型...")
        test_metrics = model.val(data=data_config, split='test')
        print_results("测试集", test_metrics)
        
        # 打印详细指标
        print("\n详细评估指标:")
        print("-" * 50)
        print(f"mAP50 (验证集): {val_metrics.box.map50:.4f}")
        print(f"mAP50-95 (验证集): {val_metrics.box.map:.4f}")
        print(f"mAP50 (测试集): {test_metrics.box.map50:.4f}")
        print(f"mAP50-95 (测试集): {test_metrics.box.map:.4f}")
        
        # 各类别指标
        if hasattr(val_metrics.box, 'maps') and val_metrics.box.maps is not None:
            print("\n各类别在验证集上的mAP50:")
            class_names = ['fire', 'hardhat', 'no-hardhat', 'safety-vest', 'no-safety-vest']
            for i, (name, map_val) in enumerate(zip(class_names, val_metrics.box.maps)):
                print(f"  {name}: {map_val:.4f}")
        
        # 速度评估
        print("\n推理速度评估:")
        speed = val_metrics.speed['inference']
        print(f"平均推理时间: {speed:.2f}ms per image")
        
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")
        raise e

def print_results(dataset_name, metrics):
    """打印评估结果"""
    print(f"\n{dataset_name}评估结果:")
    print("-" * 30)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"精确率: {metrics.box.p:.4f}")
    print(f"召回率: {metrics.box.r:.4f}")
    print(f"F1分数: {metrics.box.f1:.4f}")

def visualize_predictions():
    """可视化预测结果"""
    print("生成预测结果可视化...")
    
    # 检查模型权重
    best_model_path = 'runs/detect/train/weights/best.pt'
    if not os.path.exists(best_model_path):
        print("未找到最佳模型权重")
        return
    
    try:
        # 加载模型
        model = YOLO(best_model_path)
        
        # 预测几张测试图像
        test_images_path = 'dataset/powerplant_safety/test/images'
        if os.path.exists(test_images_path):
            test_images = os.listdir(test_images_path)[:5]  # 只取前5张图像
            
            for img_name in test_images:
                img_path = os.path.join(test_images_path, img_name)
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # 进行预测
                    results = model(img_path)
                    
                    # 保存预测结果
                    save_dir = 'predictions'
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # 保存带标注的图像
                    results[0].save(filename=os.path.join(save_dir, f"pred_{img_name}"))
                    
            print(f"预测结果已保存到 {save_dir} 目录")
        else:
            print("未找到测试图像目录")
            
    except Exception as e:
        print(f"可视化过程中出现错误: {str(e)}")
        raise e

if __name__ == "__main__":
    evaluate_model()
    visualize_predictions()