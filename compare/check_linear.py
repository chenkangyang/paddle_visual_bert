from importlib.machinery import FrozenImporter
import numpy as np
from ipdb import set_trace as st

x = np.load("torch_before_hs.npy")
w = np.load("torch_linear_weight.npy")
b = np.load("torch_linear_bias.npy")
y = np.load("torch_after_hs.npy")
np_y = np.matmul(x, w.T) + b
print("y=wx+b: numpy VS. torch y", np.amax(abs(y-np_y)))

x1 = np.load("paddle_before_hs.npy")
w1 = np.load("paddle_linear_weight.npy")
b1 = np.load("paddle_linear_bias.npy")
y1 = np.load("paddle_after_hs.npy")
np_y1 = np.matmul(x1, w1.T) + b1
print("y=wx+b: numpy VS. paddle", np.amax(abs(y1-np_y1)))

print("pytorch paddle 768x30522 Linear层精度对比")
print("x_diff", np.amax(abs(x-x1)))
print("w_diff", np.amax(abs(w-w1)))
print("b_diff", np.amax(abs(b-b1)))
print("y_diff", np.amax(abs(y-y1)))
print("np_y_diff", np.amax(abs(np_y-np_y1)))