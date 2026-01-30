import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler  # 引入混合精度


# ==========================================
# 1. 模拟 ViT Registers 机制 (ICLR 2024)
# ==========================================
class ViTWithRegisters(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_registers=4):
        super().__init__()
        self.num_registers = num_registers
        self.embed_dim = embed_dim

        # 寄存器标记：不参与分类，只负责吸收冗余注意力
        self.register_tokens = nn.Parameter(torch.randn(1, num_registers, embed_dim))

        # 简化的注意力层
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Embed_Dim] (图像切片特征)
        batch_size = x.shape[0]

        # 拼接寄存器标记到输入序列末尾
        registers = self.register_tokens.expand(batch_size, -1, -1)
        x = torch.cat([x, registers], dim=1)  # [Batch, Seq_Len + 4, Embed_Dim]

        # 模拟 Transformer Block
        attn_output, attn_weights = self.attn(x, x, x)

        # 移除寄存器标记，只返回图像切片特征用于下游任务
        out = attn_output[:, :-self.num_registers, :]
        return out, attn_weights

def run_rtx5070_experiment():
    # 1. 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 当前使用显卡: {torch.cuda.get_device_name(0)}")

    # 2. 实例化模型并移动到单块 GPU
    model = ViTWithRegisters(num_registers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    scaler = GradScaler()  # 混合精度梯度缩放

    # 3. 模拟实验数据
    # 调小 Batch Size 以适配 RTX 5070 显存
    dummy_input = torch.randn(8, 196, 768).to(device)

    # 4. 开启混合精度的前向传播
    with autocast():
        output, weights = model(dummy_input)
        # 假设一个虚拟 Loss
        loss = output.mean()

        # 5. 反向传播优化
    optimizer.zero_grad()
    scaler.scale(loss).backward()  # 缩放梯度
    scaler.step(optimizer)
    scaler.update()

    print(f"RTX 5070 实验运行成功！")
    print(f"当前显存占用: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")


if __name__ == "__main__":
    run_rtx5070_experiment()