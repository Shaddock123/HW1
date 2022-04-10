from model import MNIST_MLP
from matplotlib import pyplot as plt

MNIST_DIR = "./mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"


# 构建网络
def build_mnist_mlp(param_dir='weight.npy'):
    mlp = MNIST_MLP(hidden1=128, max_epoch=5, lr=0.001, lambd=0.1)
    mlp.load_data(MNIST_DIR, TRAIN_DATA, TRAIN_LABEL, TEST_DATA, TEST_LABEL)  # 加载数据
    mlp.build_model()
    mlp.init_model()  # 初始化参数
    return mlp


if __name__ == '__main__':
    # 构建网络
    mlp = build_mnist_mlp()
    train_loss, test_loss, acc = mlp.train()  # 模型训练
    mlp.save_model('mlp.npy')  # 保存模型
    plt.figure()
    plt.xlabel("iter")
    plt.ylabel("train loss")
    plt.plot(train_loss)
    plt.savefig('./results/train_loss.jpg')

    plt.figure()
    plt.xlabel("iter")
    plt.ylabel("test loss")
    plt.plot(test_loss)
    plt.savefig('./results/test_loss.jpg')

    plt.figure()
    plt.xlabel("iter")
    plt.ylabel("acc")
    plt.plot(acc)
    plt.savefig('./results/accuracy.jpg')

