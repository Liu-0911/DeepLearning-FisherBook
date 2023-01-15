import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def gendata(fun):
    x1 = np.array([0,1,0,1])
    x2 = np.array([0,0,1,1])
    y = np.array([fun(i,j) for i,j in zip(x1, x2)])
    data = np.array([x1,x2,y]).T
    return data

def drawfrom():
    columns = ['x1','x2','y']
    FramOfAND = pd.DataFrame(gendata(AND),columns=columns)
    print('----AND----')
    print(FramOfAND.to_string(index=False))
    FramOfAND = pd.DataFrame(gendata(NAND), columns=columns)
    print('----NAND----')
    print(FramOfAND.to_string(index=False))
    FramOfAND = pd.DataFrame(gendata(OR), columns=columns)
    print('----OR----')
    print(FramOfAND.to_string(index=False))


drawfrom()
