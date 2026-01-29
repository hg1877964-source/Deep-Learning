# Deep-Learning
深度学习代码
# 深度学习课程设计：基于 Transformer 的稳健深度估计研究

**作者：** [黄钢]  
**硬件环境：** 单块 NVIDIA GeForce RTX 5070 (12GB/16GB VRAM)

## 1. 摘要 (Abstract)
本文针对单目深度估计的泛化性及 Transformer 的特征伪影问题，解读了 Depth Anything (CVPR 2024) 与 ViT Registers (ICLR 2024)。针对硬件限制，提出了基于 RTX 5070 的混合精度训练与梯度累加优化方案。实验表明，改进后的模型在实时性与特征纯净度上均取得显著提升。

## 2. 快速运行
```bash
# 安装依赖
pip install torch torchvision opencv-python

# 运行单卡优化训练模拟
python train_opt.py

# 启动摄像头实时演示
python demo_rtx5070.py
