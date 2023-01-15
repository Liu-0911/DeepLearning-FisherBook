import sys, os
sys.path.append(os.pardir) # 为了导入父目录中的文件而进行的设定
import numpy as np
from SourceCode.dataset.mnist import  load_mnist
from PIL import Image
import pickle
from ActivationFunction import sigmoid, softmax
import time
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 第一次调用会花费几分钟 ……
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False,normalize=False,one_hot_label=True)
#
# # 输出各个数据的形状
# print(x_train.shape) # (60000, 784)
# print(t_train.shape) # (60000,)
# print(x_test.shape) # (10000, 784)
# print(t_test.shape) # (10000,)


# img = x_train[0]
# label = t_train[0]
# print(label) # 5
#
# print(img.shape) # (784,)
# img = img.reshape(28, 28) # 把图像的形状变成原来的尺寸
# print(img.shape) # (28, 28)
# img_show(img)

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("../SourceCode/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


start_time = time.perf_counter()   # 程序开始时间

x, t = get_data()
# print("x.shape: {}, t.shape: {}".format(x.shape, t.shape))

network = init_network()
# print("netowrk: ", ["参数名称：{}， 参数尺寸：{}".format(k, v.shape) for k, v in network.items()])
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)    # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

end_time = time.perf_counter()   # 程序结束时间
run_time = end_time - start_time    # 程序的运行时间，单位为秒
print(run_time)