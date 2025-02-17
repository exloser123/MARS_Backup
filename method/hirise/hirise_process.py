import os
import tqdm
import numpy as np
from method.mola import mola
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
from osgeo import gdal
from scipy.signal import convolve
import scipy.signal as signal

import django

# 首先初始化django的环境，使得可以在其他py文件中使用django的模型
# 设置Django的设置模块路径。
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mars_sim_web.settings")
# 初始化Django
django.setup()
import sim_site.models as models

# 建立基础路径，使得在其他py文件中可以直接引用
BASE_DIR = Path(__file__).resolve().parent
jpg_path = os.path.join(BASE_DIR, "data", "JPG")

# 从数据库中获取所有的HiRISE DTM数据
hirise_dtm_data = models.HiriseDtmData.objects.all()


# 输出图像在mola数据中的位置
def get_region_in_mola():
    print("开始更新所有的HiRISE DTM数据的在mola数据的位置信息")
    # 遍历所有的数据
    for i in tqdm.tqdm(range(len(hirise_dtm_data)), desc="get region in mola"):
        data = hirise_dtm_data[i]
        png_path = os.path.join(jpg_path, data.PRODUCT_ID + ".png")
        if not os.path.exists(png_path):
            # 获取MINIMUM_LATITUDE，MAXIMUM_LATITUDE，MINIMUM_LONGITUDE，MAXIMUM_LONGITUDE
            min_lat, max_lat, min_lon, max_lon = (
                data.MINIMUM_LATITUDE,
                data.MAXIMUM_LATITUDE,
                data.MINIMUM_LONGITUDE,
                data.MAXIMUM_LONGITUDE,
            )
            # 获取CORNER1_LATITUDE，CORNER1_LONGITUDE，CORNER2_LATITUDE，CORNER2_LONGITUDE，CORNER3_LATITUDE，CORNER3_LONGITUDE，CORNER4_LATITUDE，CORNER4_LONGITUDE
            corner1_lat, corner1_lon = data.CORNER1_LATITUDE, data.CORNER1_LONGITUDE
            corner2_lat, corner2_lon = data.CORNER2_LATITUDE, data.CORNER2_LONGITUDE
            corner3_lat, corner3_lon = data.CORNER3_LATITUDE, data.CORNER3_LONGITUDE
            corner4_lat, corner4_lon = data.CORNER4_LATITUDE, data.CORNER4_LONGITUDE

            # lon_range,lat_range 扩展本身范围的200%
            lon_range = [
                min_lon - (max_lon - min_lon) * 2,
                max_lon + (max_lon - min_lon) * 2,
            ]
            lat_range = [
                min_lat - (max_lat - min_lat) * 2,
                max_lat + (max_lat - min_lat) * 2,
            ]
            # 如果 lon_range,lat_range 超出了 [-180,180],[-90,90]的范围，则设置为 [-180,180],[-90,90]
            lon_range = [max(-180, lon_range[0]), min(180, lon_range[1])]
            lat_range = [max(-90, lat_range[0]), min(90, lat_range[1])]
            # 把四个角处理成一个字典
            polygons = {
                "polygons": [
                    [
                        (corner1_lon, corner1_lat),
                        (corner2_lon, corner2_lat),
                        (corner3_lon, corner3_lat),
                        (corner4_lon, corner4_lat),
                    ]
                ],
                "color": ["red"],
            }

            mola.plot_mola(
                lon_range,
                lat_range,
                polygons=polygons,
                title=data.PRODUCT_ID,
                save_path=png_path,
            )
            plt.close()
        else:
            print(data.PRODUCT_ID + " exists")
    print("get_region_in_mola 更新完成")


def calculate_correlation_length(A, dr):
    """
    相关长度的计算函数，1D/2D
    :param A:  1D/2D粗糙面的高度坐标
    :param dr:  横向分辨率（如果是2D粗糙面，假设是x方向和y方向分辨率是一样的）
    :return:  1D粗糙面返回1个值
            2D粗糙面返回一个1*2的向量
                     第一个为A[:,0]方向
                     第二个为A[0,:]方向
    """
    er = 1e-1  # 误差
    m = A.shape
    if len(m) > 2 or len(m) == 0:
        print("input error")
        return []
    elif len(m) == 1:
        avg = np.mean(A)
        A = A - avg

        n = np.prod(A.shape)
        x = np.arange(0, n) * dr

        C = signal.correlate(A, A)
        h = max(C)
        dff = C[n - 1 :] - h / np.exp(1)
        if np.sum(np.abs(dff) < er) == 1:
            return x[np.abs(dff) < er][0]
        else:
            index = np.where(dff < 0)[0][0]
            x1 = x[index]
            x2 = x[index - 1]
            y1 = abs(dff[index])
            y2 = dff[index - 1]
            return (y1 * x1 + y2 * x2) / (y1 + y2)
    else:
        avg = np.mean(A)
        A = A - avg

        C1 = np.zeros(m[0])
        for ii in range(m[0]):
            C1[ii] = np.sum(A[0 : -ii or None, :] * A[ii:, :])

        x = np.arange(0, m[0]) * dr
        h = max(C1)
        dff = C1 - h / np.exp(1)
        if np.sum(np.abs(dff) < er) == 1:
            val = np.array([x[np.abs(dff) < er][0]])
        else:
            index = np.where(dff < 0)[0][0]
            x1 = x[index]
            x2 = x[index - 1]
            y1 = abs(dff[index])
            y2 = dff[index - 1]
            val = np.array([(y1 * x1 + y2 * x2) / (y1 + y2)])

        C1 = np.zeros(m[1])
        for ii in range(m[1]):
            C1[ii] = np.sum(A[:, 0 : -ii or None] * A[:, ii:])

        x = np.arange(0, m[1]) * dr
        h = max(C1)
        dff = C1 - h / np.exp(1)
        if np.sum(np.abs(dff) < er) == 1:
            return np.append(val, x[np.abs(dff) < er][0])
        else:
            index = np.where(dff < 0)[0][0]
            x1 = x[index]
            x2 = x[index - 1]
            y1 = abs(dff[index])
            y2 = dff[index - 1]
            return np.append(val, (y1 * x1 + y2 * x2) / (y1 + y2))


def calculate_root_mean_square_height(A):
    """
    均方根高度的计算函数，1D/2D
    :param A:  1D/2D粗糙面的高度坐标
    :return:  1D粗糙面返回1个值
            2D粗糙面返回一个1*2的向量
                     第一个为A[:,0]方向
                     第二个为A[0,:]方向
    """
    m = A.shape
    if len(m) > 2 or len(m) == 0:
        print("input error")
        return []
    elif len(m) == 1:
        # 数据中可能会有nan，需要先去除
        return np.sqrt(np.nanmean(A**2))
    else:
        return np.array(
            [np.sqrt(np.nanmean(A**2, axis=0)), np.sqrt(np.nanmean(A**2, axis=1))]
        )


def read_IMG(path):
    """
    读取IMG文件
    :param path:  图像的路径
    :return:    data: 图像的数据
                ignore_value: 无效值
    """
    # 打开数据
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    # 读取数据
    data = dataset.ReadAsArray()
    data = np.array(data)
    # ignore_value
    ignore_value = dataset.GetRasterBand(1).GetNoDataValue()
    return data, ignore_value


def dem_filter(data, interval):
    """
    滤波函数,得到大尺度的趋势
    :param data: 一维数组
    :param interval: 间隔
    :return: origin_data: 原始数据
            filtered_data: 滤波后的数据
    """
    origin_data = np.pad(data, (interval, interval), "edge")
    # 对间隔interval的数据进行拟合，得到大范围的data的趋势
    filter_kernel = np.ones(interval) / interval
    # 应用滤波器,边界填充方式为镜像填充
    filtered_data = convolve(origin_data, filter_kernel, mode="same", method="auto")
    # 去除边界
    filtered_data = filtered_data[interval:-interval]
    origin_data = origin_data[interval:-interval]
    return origin_data, filtered_data


def outlier_handling(data):
    """
    异常值处理
    :param data: 一维数组
    :return: data: 处理后的数据
    """
    # 计算均值和标准差
    median = np.nanmedian(data)
    std = np.nanstd(data)
    # 2倍标准差以外的数据视为异常值
    data = data[(data > median - 2 * std) & (data < median + 2 * std)]

    return data


def process_main(sample, actual_interval):
    # IMG文件的路径
    img_path = os.path.join(BASE_DIR, "data", "DTM", sample.PRODUCT_ID + ".IMG")
    # 读取数据
    dem_data, dem_ignore_value = read_IMG(img_path)
    # 间隔
    map_scale = sample.MAP_SCALE
    # 索引间隔
    index_interval = int(actual_interval / map_scale)
    # 空的相关长度array和均方根高度array,长度为dem_data.shape[0]
    correlation_length_list = np.empty(dem_data.shape[0], dtype=np.float32)
    root_mean_square_height_list = np.empty(dem_data.shape[0], dtype=np.float32)
    # 处理每一行
    pbar = tqdm.tqdm(total=dem_data.shape[0], desc=sample.PRODUCT_ID + " 计算中")
    for i in range(dem_data.shape[0]):
        # 获取一行数据
        row_of_data = dem_data[i, :]
        # 去除无效值
        row_of_data = row_of_data[row_of_data != dem_ignore_value]
        if row_of_data.shape[0] < 500:
            correlation_length_list[i] = np.nan
            root_mean_square_height_list[i] = np.nan
            pbar.update(1)
            continue
        else:
            # 过滤数据
            origin_data, filtered_data = dem_filter(row_of_data, index_interval)
            # 小尺度数据
            small_scale_data = origin_data - filtered_data
            # 计算相关长度
            single_correlation_length = calculate_correlation_length(
                small_scale_data, map_scale
            )
            # 计算均方根高度
            single_root_mean_square_height = calculate_root_mean_square_height(
                small_scale_data
            )
            # 存储数据
            correlation_length_list[i] = single_correlation_length
            root_mean_square_height_list[i] = single_root_mean_square_height
        pbar.update(1)
    pbar.close()
    return correlation_length_list, root_mean_square_height_list


def update_result():
    print("开始更新所有的HiRISE DTM数据的粗糙度结果数据")
    pbar = tqdm.tqdm(total=len(hirise_dtm_data), desc="update result")
    for i in range(len(hirise_dtm_data)):
        data = hirise_dtm_data[i]
        # 保存数据
        corL_path = os.path.join(
            BASE_DIR, "data", "result", data.PRODUCT_ID + "_corL.npy"
        )
        rmsH_path = os.path.join(
            BASE_DIR, "data", "result", data.PRODUCT_ID + "_rmsH.npy"
        )
        if os.path.exists(corL_path) and os.path.exists(rmsH_path):
            pbar.update(1)
            continue
        corL, rmsH = process_main(data, 20)
        # 异常值处理
        corL = outlier_handling(corL)
        rmsH = outlier_handling(rmsH)

        np.save(corL_path, corL, allow_pickle=True)
        np.save(rmsH_path, rmsH, allow_pickle=True)
        pbar.update(1)
    pbar.close()
    print("update_result 更新完成")


if __name__ == "__main__":
    get_region_in_mola()
    # update_result()
    abc = 1
