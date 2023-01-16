# coding: utf-8
import numpy as np

def _numerical_gradient_1d(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        
    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])    #高效的多维迭代器 flags的参数['multi_index']表示同时跟踪多个索引  op_flags=['readwrite']表示同时进行读写操作
    while not it.finished:
        idx = it.multi_index   # it.multi_index 为当前元素的位置，为元组形式
        # print(f"idx = {idx}")
        tmp_val = x[idx]
        # print(f"tmp_val = {tmp_val}")
        x[idx] = float(tmp_val) + h
        # print(f"x[idx] = {x[idx]}")
        fxh1 = f(x) # f(x+h)
        # print(f"fxh1 = {fxh1}")
        # print(f"x = {x}")
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad