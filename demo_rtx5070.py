import cv2
import torch
import numpy as np


def run_live_demo():
    device = torch.device("cuda")
    # 启用 cuDNN 性能自动优化
    torch.backends.cudnn.benchmark = True

    cap = cv2.VideoCapture(0)
    print("RTX 5070 实时深度估计 Demo 启动...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 模拟模型推理过程 (实际应加载模型权重)
        start = cv2.getTickCount()

        # 预处理与伪彩色映射 (模拟结果)
        # 备注：在你的报告中展示此处使用了 Inferno 映射
        depth_map = cv2.applyColorMap(cv2.convertScaleAbs(frame, alpha=0.5), cv2.COLORMAP_INFERNO)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - start)

        cv2.putText(depth_map, f"FPS: {fps:.1f} (RTX 5070)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Real-time Analysis', np.hstack((frame, depth_map)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_demo()