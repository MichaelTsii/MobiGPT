import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import seaborn as sns
from scipy.interpolate import interp1d


def expand_array(arr, target_size):
    """
    将一个形状为 (12,8,8) 的数组在第一个维度上通过线性插值拓展到 (64,8,8)

    参数:
    arr (np.ndarray): 原始数组，形状为 (12,8,8)
    target_size (int): 目标尺寸，第一个维度的目标大小，如 64

    返回:
    np.ndarray: 形状为 (64,8,8) 的新数组
    """
    original_size = arr.shape[0]

    # 创建一个新的数组，用于插值后的结果
    new_shape = (target_size, arr.shape[1], arr.shape[2])
    new_arr = np.zeros(new_shape)

    # 原始数组的第一个维度的索引
    x = np.arange(original_size)

    # 新数组的第一个维度的索引
    new_x = np.linspace(0, original_size - 1, target_size)

    # 对每一个 (8,8) 的子数组进行插值
    for i in range(arr.shape[1]):
        for j in range(arr.shape[2]):
            f = interp1d(x, arr[:, i, j], kind='linear')
            new_arr[:, i, j] = f(new_x)

    return new_arr


def sum_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.view(tensor.size(0), -1).sum(axis=-1)
# 加载数据
# gen = np.load('1generate_TrafficBJ_32.npz',allow_pickle=True)['gen_traffic'][0,2]
#
#
gen = np.load('generate.npz',allow_pickle=True)['gen_traffic']
mask = np.load('mask.npz',allow_pickle=True)['mask']
tar = np.load('target.npz',allow_pickle=True)['tar_traffic']

# real = np.concatenate((tar[1,5,5][:,:4,3], tar[1,5,5][:,:4,4], tar[1,5,2][:,:4,3], tar[1,5,2][:,:4,4]), axis=-1)
# our_gen = np.concatenate((gen[1,5,5][:,:4,3], gen[1,5,5][:,:4,4], gen[1,5,2][:,:4,3], gen[1,5,2][:,:4,4]), axis=-1)
# real_res = np.transpose(real, axes=(1,0))
# our_res = np.transpose(our_gen, axes=(1,0))
#
# np.savez('real_res.npz',real=real_res)
# np.savez('our_res.npz',predict=our_res)


# gen = np.load('TrafficNC3_gen.npz',allow_pickle=True)['pre']
# tar = np.load('TrafficNC3_tar.npz',allow_pickle=True)['tar']
# # # # # #是否要扩充数据--------------------------------------------------------
gen_ex = gen
tar_ex = tar
for i in range(32):
    plt.plot(gen_ex[5, i], label='Generated data', linewidth=2)
    plt.plot(tar_ex[5, i], '--', label='Real data', linewidth=2)
    plt.xlabel('Time (hours)', fontsize=14)  # X轴标签
    plt.ylabel('Mobile Traffic data', fontsize=14)  # Y轴标签
    plt.grid()
    plt.show()
#---------------------------------------------------------------------------------


gen_ex = gen
tar_ex = tar
for i in range(7):
    j = 3
    aaa=gen_ex[:, i, j]
    plt.plot(gen_ex[:,i,j], label='Generated data', linewidth=2)
    plt.plot(tar_ex[:,i,j],'--', label='Real data', linewidth=2)
    plt.xlabel('Time (hours)',fontsize = 14)  # X轴标签
    plt.ylabel('Mobile Traffic data',fontsize = 14)  # Y轴标签

    unequal_index = None
    for k in range(len(gen_ex[:, i, j])):
        if gen_ex[k, i, j] != tar_ex[k, i, j]:
            unequal_index = k
            break

    if unequal_index is not None:
        plt.axvline(x=unequal_index, color='red', linestyle='--')
        plt.axvspan(unequal_index, len(gen_ex[:, i, j]) - 1, color='blue', alpha=0.3)

    plt.legend(loc = 'lower left',fontsize = 14)  # 添加图例
    plt.grid()
    plt.show()
# 定义当前索引
current_index = [0]

# 定义更新热力图的函数
def update_heatmap(event):
    current_index[0] = (current_index[0] + 1) % tar.shape[0]
    axes[0].clear()
    axes[1].clear()

    axes[2].clear()

    sns.heatmap(gen[current_index[0]], annot=False, fmt=".2f", cmap='coolwarm', cbar=False,ax=axes[0])
    axes[0].set_title(f"Heatmap for Generated Index {current_index[0]}")

    sns.heatmap(tar[current_index[0]], annot=False, fmt=".2f", cmap='coolwarm',cbar=False, ax=axes[1])
    axes[1].set_title(f"Heatmap for Target Index {current_index[0]}")

    sns.heatmap(mask[current_index[0]], annot=False, fmt=".2f", cmap='coolwarm', cbar=False, ax=axes[2])
    axes[2].set_title(f"Heatmap for Mask Index {current_index[0]}")
    plt.draw()

# 创建图形和子图
fig, axes = plt.subplots(3, 1, figsize=(6, 10))

# 初始化显示第一个热力图
sns.heatmap(gen[current_index[0]], annot=False, fmt=".2f", cmap='coolwarm', ax=axes[0])
axes[0].set_title(f"Heatmap for Generated Index {current_index[0]}")
sns.heatmap(tar[current_index[0]], annot=False, fmt=".2f", cmap='coolwarm', ax=axes[1])
axes[1].set_title(f"Heatmap for Target Index {current_index[0]}")

sns.heatmap(mask[current_index[0]], annot=False, fmt=".2f", cmap='coolwarm', ax=axes[2])
axes[2].set_title(f"Heatmap for mask Index {current_index[0]}")

# 添加按钮
button_ax = plt.axes([0.45, 0.01, 0.1, 0.05])  # [left, bottom, width, height]
button = Button(button_ax, 'Next')
button.on_clicked(update_heatmap)

plt.show()
#
