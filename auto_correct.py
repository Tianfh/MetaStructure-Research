import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 0. 自动选择设备（有 CUDA 用 CUDA，没有就用 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 准备“教学数据”
# 我们模拟 100 个点，规律是 y = 3x + 1
x = torch.linspace(-1, 1, 100).reshape(-1, 1).to(device)
y_true = 3 * x + 1 + torch.randn(x.size()).to(device) * 0.1  # 加一点噪声模拟真实情况

# 2. 建立一个“白纸”模型
# nn.Linear(1, 1) 代表输入 1 个数，输出 1 个数。刚开始它的参数是随机的错误数值。
model = nn.Linear(1, 1).to(device)

# 3. 准备工具
# 损失函数：计算“预测值”和“真实值”差了多少
criterion = nn.MSELoss()
# 优化器：负责根据误差去旋转“参数旋钮”
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("训练开始前，模型的预测简直是乱猜...")

# 4. 自动纠错循环
loss_history = []
for epoch in range(201):
    # 第一步：模型试着猜一下结果
    prediction = model(x)
    
    # 第二步：计算猜得有多离谱（误差）
    loss = criterion(prediction, y_true)
    loss_history.append(loss.item())
    
    # 第三步：【核心】自动求导，算出每个参数该往哪调
    optimizer.zero_grad() # 先清空旧的记忆
    loss.backward()       # 像剥洋葱一样反向算出梯度
    
    # 第四步：真正动手旋转旋钮（更新参数）
    optimizer.step()
    
    if epoch % 50 == 0:
        # 提取模型现在的参数（权重 w 和 偏置 b）
        w = model.weight.item()
        b = model.bias.item()
        print(f"迭代 {epoch:3d} 次 | 误差: {loss.item():.4f} | AI 认为公式是: y = {w:.2f}x + {b:.2f}")

print("\n纠错完成！")

# 5. 绘制 Loss 随迭代次数变化曲线（训练结束后弹窗显示）
plt.figure(figsize=(8, 4.5))
plt.plot(loss_history, linewidth=2)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()