import os
from ultralytics import YOLO

def export_model():
    """导出训练好的模型为不同格式"""
    print("开始导出模型...")
    
    # 检查最佳模型权重是否存在
    best_model_path = 'runs/detect/train/weights/best.pt'
    
    if not os.path.exists(best_model_path):
        print("未找到训练好的模型权重文件")
        return
    
    try:
        # 加载模型
        model = YOLO(best_model_path)
        print("模型加载成功")
        
        # 创建导出目录
        export_dir = 'exported_models'
        os.makedirs(export_dir, exist_ok=True)
        print(f"导出目录: {export_dir}")
        
        # 导出为ONNX格式
        print("正在导出为ONNX格式...")
        onnx_path = model.export(format='onnx', 
                                opset=12,
                                dynamic=False,
                                simplify=True,
                                export_dir=export_dir)
        print(f"ONNX模型已导出到: {onnx_path}")
        
        # 导出为TensorRT格式 (如果系统支持)
        try:
            print("正在导出为TensorRT格式...")
            trt_path = model.export(format='engine',
                                   device=0,  # 使用第一个GPU
                                   half=False,  # 不使用半精度
                                   export_dir=export_dir)
            print(f"TensorRT模型已导出到: {trt_path}")
        except Exception as e:
            print(f"TensorRT导出失败: {str(e)}")
            print("注意: TensorRT导出需要安装TensorRT并配置相关环境")
        
        # 导出为TorchScript格式
        print("正在导出为TorchScript格式...")
        ts_path = model.export(format='torchscript',
                              optimize=False,
                              export_dir=export_dir)
        print(f"TorchScript模型已导出到: {ts_path}")
        
        # 导出为OpenVINO格式
        try:
            print("正在导出为OpenVINO格式...")
            openvino_path = model.export(format='openvino',
                                        export_dir=export_dir)
            print(f"OpenVINO模型已导出到: {openvino_path}")
        except Exception as e:
            print(f"OpenVINO导出失败: {str(e)}")
            print("注意: OpenVINO导出需要安装openvino库")
        
        print("\n模型导出完成!")
        print(f"所有导出的模型都保存在: {export_dir}")
        
    except Exception as e:
        print(f"导出过程中出现错误: {str(e)}")
        raise e

if __name__ == "__main__":
    export_model()