"""
电力设施安全检测模型训练主程序
基于YOLOv8实现明火、安全帽和工作服检测
"""

import os
import sys
import argparse
from train import train_model
from evaluate import evaluate_model
from export_model import export_model

def show_menu():
    """显示操作菜单"""
    print("\n" + "="*60)
    print("电力设施安全检测模型训练系统")
    print("="*60)
    print("1. 数据预处理 (已通过data_preprocess.py完成)")
    print("2. 开始训练模型")
    print("3. 评估模型性能")
    print("4. 导出模型")
    print("5. 全流程执行 (训练->评估->导出)")
    print("0. 退出")
    print("="*60)

def main():
    """主函数"""
    print("欢迎使用电力设施安全检测模型训练系统!")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='电力设施安全检测模型训练系统')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'export', 'all'], 
                       help='运行模式: train(训练), eval(评估), export(导出), all(全流程)')
    args = parser.parse_args()
    
    # 如果提供了命令行参数，直接执行相应功能
    if args.mode:
        if args.mode == 'train':
            train_model()
        elif args.mode == 'eval':
            evaluate_model()
        elif args.mode == 'export':
            export_model()
        elif args.mode == 'all':
            print("开始全流程执行...")
            train_model()
            evaluate_model()
            export_model()
            print("全流程执行完成!")
        return
    
    # 否则显示交互式菜单
    while True:
        show_menu()
        choice = input("请选择操作 (0-5): ").strip()
        
        if choice == '0':
            print("感谢使用，再见!")
            break
        elif choice == '1':
            print("数据预处理已完成，请查看data_preprocess.py脚本")
        elif choice == '2':
            train_model()
        elif choice == '3':
            evaluate_model()
        elif choice == '4':
            export_model()
        elif choice == '5':
            print("开始全流程执行...")
            train_model()
            evaluate_model()
            export_model()
            print("全流程执行完成!")
        else:
            print("无效选择，请重新输入!")

if __name__ == "__main__":
    main()