import os
import shutil
import yaml

def merge_datasets():
    """
    合并Personal_Protective和fire两个数据集，创建适用于电力设施安全检测的新数据集
    """
    # 定义路径
    base_path = "dataset"
    pp_path = os.path.join(base_path, "Personal_Protective")
    fire_path = os.path.join(base_path, "fire")
    output_path = os.path.join(base_path, "powerplant_safety")
    
    # 创建输出目录结构
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)
    
    # 定义新的类别映射
    # 我们需要的类别: 0=明火, 1=戴安全帽, 2=未戴安全帽, 3=合规工作服, 4=不合规工作服
    new_class_mapping = {
        # 从fire数据集映射
        'fire_fire': 0,  # 明火
        
        # 从Personal_Protective数据集映射
        'Personal_Protective_Hardhat': 1,      # 戴安全帽
        'Personal_Protective_NO-Hardhat': 2,   # 未戴安全帽
        'Personal_Protective_Safety Vest': 3,  # 合规工作服
        'Personal_Protective_NO-Safety Vest': 4  # 不合规工作服
    }
    
    # 处理每个分割集
    for split in ['train', 'valid', 'test']:
        print(f"处理 {split} 数据集...")
        
        # 处理fire数据集
        fire_images_path = os.path.join(fire_path, split, 'images')
        if os.path.exists(fire_images_path):
            process_dataset_split(fire_images_path, os.path.join(fire_path, split, 'labels'), 
                                os.path.join(output_path, split), new_class_mapping, 'fire')
        
        # 处理Personal_Protective数据集
        pp_images_path = os.path.join(pp_path, split, 'images')
        if os.path.exists(pp_images_path):
            process_dataset_split(pp_images_path, os.path.join(pp_path, split, 'labels'), 
                                os.path.join(output_path, split), new_class_mapping, 'Personal_Protective')
    
    # 创建新的数据集配置文件
    create_dataset_yaml(output_path)
    
    print("数据集合并完成！")
    print("新数据集位置:", output_path)

def process_dataset_split(images_path, labels_path, output_path, class_mapping, dataset_type):
    """
    处理单个数据集分割
    """
    if not os.path.exists(labels_path):
        print(f"警告: 标签路径不存在 {labels_path}")
        return
    
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    
    for i, label_file in enumerate(label_files):
        try:
            # 读取标签文件
            with open(os.path.join(labels_path, label_file), 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                old_class_id = int(parts[0])
                
                # 获取原数据集中的类别名
                if dataset_type == 'fire':
                    old_classes = ['fire', 'light', 'nonfire', 'smoke']
                else:  # Personal_Protective
                    old_classes = ['Fall-Detected', 'Gloves', 'Goggles', 'Hardhat', 'Ladder', 'Mask', 
                                  'NO-Gloves', 'NO-Goggles', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
                                  'Person', 'Safety Cone', 'Safety Vest']
                
                if old_class_id >= len(old_classes):
                    continue
                    
                old_class_name = old_classes[old_class_id]
                key = f"{dataset_type}_{old_class_name}"
                
                # 如果这个类别在我们的映射中，则进行转换
                if key in class_mapping:
                    new_class_id = class_mapping[key]
                    # 保持边界框坐标不变，只改变类别ID
                    parts[0] = str(new_class_id)
                    new_lines.append(' '.join(parts) + '\n')
            
            # 只有当新标签文件中有内容时才复制
            if new_lines:
                # 复制标签文件
                new_label_file = os.path.join(output_path, 'labels', label_file)
                with open(new_label_file, 'w') as f:
                    f.writelines(new_lines)
                
                # 复制对应的图像文件
                # 尝试.jpg扩展名
                image_file = label_file.replace('.txt', '.jpg')
                src_image = os.path.join(images_path, image_file)
                dst_image = os.path.join(output_path, 'images', image_file)
                
                if not os.path.exists(dst_image) and os.path.exists(src_image):
                    shutil.copy2(src_image, dst_image)
                
                # 也尝试.png扩展名
                if not os.path.exists(dst_image):
                    image_file = label_file.replace('.txt', '.png')
                    src_image = os.path.join(images_path, image_file)
                    dst_image = os.path.join(output_path, 'images', image_file)
                    if os.path.exists(src_image):
                        shutil.copy2(src_image, dst_image)
                        
            # 显示进度
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1} 个文件")
                
        except Exception as e:
            print(f"处理文件 {label_file} 时出错: {str(e)}")
            continue

def create_dataset_yaml(output_path):
    """
    创建新的数据集配置文件
    """
    yaml_content = {
        'path': os.path.abspath(output_path),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 5,
        'names': ['fire', 'hardhat', 'no-hardhat', 'safety-vest', 'no-safety-vest']
    }
    
    yaml_path = os.path.join(output_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, encoding='utf-8', allow_unicode=True)
    
    print(f"数据集配置文件已创建: {yaml_path}")

if __name__ == "__main__":
    merge_datasets()