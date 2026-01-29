import torch
import torch.nn as nn


class ViTWithRegisters(nn.Module):
    """
    实现 Vision Transformers Need Registers (ICLR 2024) 的核心思想
    通过引入寄存器标记来吸收高范数特征伪影
    """

    def __init__(self, embed_dim=768, num_heads=12, num_registers=4):
        super().__init__()
        self.num_registers = num_registers
        # 定义可学习的寄存器标记
        self.register_tokens = nn.Parameter(torch.randn(1, num_registers, embed_dim))
        # 简化版多头自注意力
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        # x: [Batch, Seq_Len, Embed_Dim]
        b = x.shape[0]
        # 拼接寄存器
        regs = self.register_tokens.expand(b, -1, -1)
        x = torch.cat([x, regs], dim=1)

        # 经过注意力层
        attn_out, _ = self.attn(x, x, x)

        # 移除寄存器，仅保留图像特征输出
        return attn_out[:, :-self.num_registers, :]