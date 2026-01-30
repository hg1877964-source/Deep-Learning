import torch
from torch.cuda.amp import autocast, GradScaler
from model import ViTWithRegisters


def train():
    device = torch.device("cuda:0")
    model = ViTWithRegisters().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    scaler = GradScaler()  # 混合精度缩放

    accum_steps = 8  # 梯度累加，模拟大 Batch

    model.train()
    for i, (images, _) in enumerate(range(100)):  # 模拟数据循环
        inputs = torch.randn(8, 196, 768).to(device)  # 物理 Batch 为 8

        with autocast():  # 开启半精度
            output = model(inputs)
            loss = output.mean() / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            print(f"Step {i}, Loss: {loss.item() * accum_steps:.4f}")


if __name__ == "__main__":
    train()