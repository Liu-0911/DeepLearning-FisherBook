import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# X = np.array([1.0,0.5])
#
# W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
# B1 = np.array([0.1, 0.2, 0.3])
# print(X.shape) # (2,)
# print(W1.shape) # (2, 3)
# print(B1.shape) # (3,)
#
# A1 = np.dot(X, W1) + B1
# Z1 = sigmoid(A1)
#
# W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
# B2 = np.array([0.1, 0.2])
# print(Z1.shape) # (3,)
# print(W2.shape) # (3, 2)
# print(B2.shape) # (2,)
#
# A2 = np.dot(Z1, W2) + B2
# Z2 = sigmoid(A2)
#
# W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
# B3 = np.array([0.1, 0.2])
#
# A3 = np.dot(Z2, W3) + B3
# Y = identity_function(A3)

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)