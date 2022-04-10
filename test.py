from model import MNIST_MLP
import numpy as np

MNIST_DIR = "./mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"


# 构建网络
def build_mnist_mlp(param_dir='weight.npy'):
    mlp = MNIST_MLP(hidden1=128, max_epoch=30)
    mlp.load_data(MNIST_DIR, TRAIN_DATA, TRAIN_LABEL, TEST_DATA, TEST_LABEL)  # 加载数据
    mlp.build_model()
    mlp.init_model()  # 初始化参数
    mlp.load_model('./mlp.npy')
    return mlp


if __name__ == '__main__':
    mlp = build_mnist_mlp()  # 构建网络
    mlp.evaluate()  # 模型测试

    # 可视化网络参数
    print("可视化网络参数：")
    weights = np.load("./mlp.npy", allow_pickle=True)
    print(weights)
