import torch

def check():
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前显卡: {torch.cuda.get_device_name(0)}")
        # 做一个简单的矩阵运算测试
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("显卡矩阵运算测试通过！")
    else:
        print("目前正在使用 CPU 模式。")

if __name__ == "__main__":
    check()