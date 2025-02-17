import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from method.geological_units.get_geological_units import get_units
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent


def generate_permittivity_map_png():
    # 读取数据
    permittivity_data = np.load("data/permittivity.npy", allow_pickle=True)
    lon_axis = np.load("data/lon_axis.npy", allow_pickle=True)
    lat_axis = np.load("data/lat_axis.npy", allow_pickle=True)
    # 画图
    # 创建figure和axes
    fig, ax = plt.subplots()
    # 处理坐标问题
    # 设置标题，并指定字体大小
    ax.set_title("Global Permittivity Map", fontsize=8)
    ax.set_xlabel("Longitude", fontsize=6)
    ax.set_ylabel("Latitude", fontsize=6)
    ax.set_xticks(np.linspace(0, len(lon_axis) - 1, 10))
    # i只保留两位小数
    ax.set_xticklabels(
        [
            "{:.2f}E".format(i) if i > 0 else "{:.2f}W".format(-i)
            for i in np.linspace(lon_axis[0], lon_axis[-1], 10)
        ],
        rotation=45,
    )
    ax.set_yticks(np.linspace(0, len(lat_axis) - 1, 10))
    # i只保留两位小数
    ax.set_yticklabels(
        [
            "{:.2f}N".format(i) if i > 0 else "{:.2f}S".format(-i)
            for i in np.linspace(lat_axis[-1], lat_axis[0], 10)
        ],
        rotation=45,
    )
    # 设置坐标轴刻度标签的字体大小
    ax.tick_params(axis="x", labelsize=4)
    ax.tick_params(axis="y", labelsize=4)
    # 画出permittivity数据
    img = ax.imshow(permittivity_data, cmap="turbo", interpolation="nearest")
    # 设置colormap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = fig.colorbar(img, cax=cax)
    # 设置colorbar的刻度字体大小
    cbar.ax.tick_params(labelsize=4)
    # colorbar的框线设置为1px细
    cbar.outline.set_linewidth(0.1)
    plt.savefig("data/img/permittivity_map.png", dpi=300, bbox_inches="tight")
    print("permittivity_map.png has been saved")


def generate_units_map_for_permittivity():
    # 读取数据
    permittivity_data = np.load("data/permittivity.npy", allow_pickle=True)
    lon_axis = np.load("data/lon_axis.npy", allow_pickle=True)
    lat_axis = np.load("data/lat_axis.npy", allow_pickle=True)
    # meshgrid
    lon, lat = np.meshgrid(lon_axis, lat_axis)
    units = get_units(lon, lat)
    # 保存为npy文件
    save_path = os.path.join(BASE_DIR, "data", "units_for_permittivity.npy")
    np.save(save_path, units, allow_pickle=True)
    print("units_for_permittivity.npy has been saved")


if __name__ == "__main__":
    # generate_permittivity_map_png()
    generate_units_map_for_permittivity()
