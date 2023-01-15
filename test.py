import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

# -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch


# x = np.arange(0,10000).reshape(100,100)
# print(x)
# print(x[np.arange(5),[4,3,2,1,0]])
# print(x[[0,1,2,3,4],[4,3,2,1,0]])
# print(x[0,4],x[1,3],x[2,2],x[3,1],x[4,0])

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
Z = X ** 2 + Y ** 2
print(Z,Z.shape)
X, Y = np.meshgrid(X, Y)
Z = X ** 2 + Y ** 2
print(Z,Z.shape)
#
# plt.ion()
fig = plt.figure() # 生成画布
ax = Axes3D(fig,auto_add_to_figure=False) #实例化Axes3D对象，创建3D图像（注意：见下方注释）
fig.add_axes(ax) # 手动将3D图像添加到画布对象上
#在matplotlib库3.4版本之后，AXes3D自动添加到Figure画布对象中这一过程被弃用了，要想免除该警告，需要在实例化Axes3D时将其auto_add_to_figure参数设置为False，然后使用fig.add_axes(ax)手动将实例化的Axes3D对象添加到Figure画布中
surf=ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
# for i in range(30):
#     x = np.linspace(1, 20, 9)
#     y = np.arange(10,19,1)
#     z = np.random.randint(20, 50, 9)  # numpy分别生成三个维度数据
#
#     ax.plot(x, y, z, 'gx--')
#     plt.show()
#     plt.pause(0.3)
# plt.ioff()
# plt.show()

# import matplotlib as mpl
#
# from mpl_toolkits.mplot3d import Axes3D
#
# import numpy as np
#
# import matplotlib.pyplot as plt
#
# mpl.rcParams['legend.fontsize'] = 10
#
# fig = plt.figure()
#
# ax = fig.add_subplot(projection='3d')
#
# theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
#
# z = np.linspace(-2, 2, 100)
#
# r = z**2 + 1
#
# x = r * np.sin(theta)
#
# y = r * np.cos(theta)
#
# ax.plot(x, y, z, label='parametric curve')
# ax.scatter(x, y, z, label='parametric curve')
# # ax.plot_wireframe(x, y, z, label='parametric curve')
# ax.plot_surface(x, y, z, label='parametric curve')
# ax.legend()
#
# plt.show()

