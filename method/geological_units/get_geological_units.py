from osgeo import ogr, osr
import numpy as np
from tqdm import tqdm
import time
import os
from method.mola.mola import get_projection
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
grid_unit_path = os.path.join(BASE_DIR, "data", "global_units_grid.npy")

# 0.1度一个单元格
interval = 0.1

grid_unit = np.load(grid_unit_path, allow_pickle=True)


def get_units(lon, lat):
    # 如果lon没有shape属性，则说明是单个点
    if not hasattr(lon, "shape"):
        lon = np.array([lon])
        lat = np.array([lat])
    lon_lat_shape = lon.shape
    # 展成一维
    lon = lon.reshape(-1)
    lat = lat.reshape(-1)
    # 四舍五入到0.1度
    lon = np.round(lon / interval) * interval
    lat = np.round(lat / interval) * interval
    # 转成整数索引
    lon_index = np.int32((lon + 180) / interval)
    lat_index = np.int32((90 - lat) / interval)
    # # 打印最大值最小值
    # print("lon_index max:", np.max(lon_index))
    # print("lon_index min:", np.min(lon_index))
    # print("lat_index max:", np.max(lat_index))
    # print("lat_index min:", np.min(lat_index))
    
    lon_index[lon_index == 3600] = 0
    lat_index[lat_index == 1800] = 0
    # 从grid_unit中取出对应的单元格
    units = grid_unit[lat_index, lon_index]
    # 转成原始形状
    units = units.reshape(lon_lat_shape)
    if not hasattr(lon, "shape"):
        return units[0]
    else:
        return units
