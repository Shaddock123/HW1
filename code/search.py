from model import MNIST_MLP

MNIST_DIR = "./mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"

lr = [0.1, 0.01, 0.001]  # 学习率大小
lambd = [0.1, 0.2, 0.3]  # 正则化强度
hidden = [512, 256, 128]  # 隐藏层个数
epoch = 5


def build_mnist_mlp(hidden, lr, lambd, epoch=5):
    mlp = MNIST_MLP(hidden1=hidden, max_epoch=epoch, lr=lr, lambd=lambd)
    mlp.load_data(MNIST_DIR, TRAIN_DATA, TRAIN_LABEL, TEST_DATA, TEST_LABEL)  # 加载数据
    mlp.build_model()
    mlp.init_model()  # 初始化参数
    return mlp


if __name__ == '__main__':
    best_para = {"hidden": None, "lr": None, "lambd": None}
    best_acc = 0.0
    for h in hidden:
        for learning_rate in lr:
            for l in lambd:
                mlp = build_mnist_mlp(h, learning_rate, l)
                mlp.train()  # 模型训练
                acc,_ = mlp.evaluate()
                if acc > best_acc:
                    print("hidden: {},  lr: {},  lambd: {}".format(h, learning_rate, l))
                    print("acc: {}".format(acc))
                    best_acc = acc
                    best_para["hidden"] = h
                    best_para["lr"] = learning_rate
                    best_para["lambd"] = l
                    # mlp.save_model('mlp_{}_{}_{}_{}.npy'.format())  # 保存模型

    # 使用最优的参数
    mlp = build_mnist_mlp(best_para["hidden"], best_para["lr"], best_para["lambd"])
    mlp.train()  # 模型训练
    mlp.save_model('mlp_{}_{}_{}.npy'.format(best_para["hidden"], best_para["lr"], best_para["lambd"]))