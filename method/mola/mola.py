from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

# mola.py所在的目录
BASE_DIR = os.path.dirname(__file__)

# MOLA数据所在的目录
MOLA_DIR = os.path.join(BASE_DIR, "data", "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif")


# 获取Projection
def get_projection():
    """
    :return: 返回MOLA数据的Projection
    """
    mola_data_set = gdal.Open(MOLA_DIR, gdal.GA_ReadOnly)
    return mola_data_set.GetProjection()


def dynamic_font_size(fig):
    # 假设fig的宽度为10时，我们想要的字体大小为10
    return fig.get_figwidth() / 1


# 画出MOLA数据加点或者多边形，或者线
def plot_mola(
    lon_range,
    lat_range,
    downSamplingRate=1,
    cmap="turbo",
    points=None,
    polygons=None,
    lines=None,
    title=None,
    save_path=None,
):
    """
    :param lon_range: 一个长度为2的list，表示经度范围，范围为[-180,180]
    :param lat_range: 一个长度为2的list，表示纬度范围，范围为[-90,90]
    :param downSamplingRate: 降采样率，用于降低MOLA数据的分辨率，数字越大，分辨率越低
    :param cmap: 画图的颜色，默认为turbo
    :param points: 一个字典，形如："points"，"color"，"size"，分别表示点的坐标，颜色，大小，且三者长度相等
    {
        "points": [[(lon1, lat1), (lon2, lat2), ...],[...],...],
        "color": ["red", "blue", ...],
        "size": [10, 20, ...]
    }
    :param polygons: 一个字典，形如："polygons"，"color"，分别表示多边形的坐标（连线顺序），颜色，且两者长度相等
    {
        "polygons": [[(lon1, lat1), (lon2, lat2), ...],[...],...],
        "color": ["red", "blue", ...]
    }
    :param lines: 一个字典，形如："lines"，"color"，分别表示线的坐标（连线顺序），颜色，且两者长度相等
    {
        "lines": [[(lon1, lat1), (lon2, lat2), ...],[...],...],
        "color": ["red", "blue", ...]
    }
    :param title: 图片的标题,如果为None，则不设置标题
    :param save_path: 所画出的图片的保存路径，如果为None，则不保存
    :return: 没有返回值，只有报错会返回错误信息
    """
    # 获取MOLA数据
    (
        mola_data,
        _,
        _,
        buf_x_size,
        buf_y_size,
        mola_GeoTransform,
    ) = get_mola(lon_range, lat_range, downSamplingRate=downSamplingRate)

    # 创建figure和axes
    fig, ax = plt.subplots()
    font_size = dynamic_font_size(fig)

    # 处理坐标问题
    ax.set_xlabel("Longitude", fontsize=font_size)
    ax.set_ylabel("Latitude", fontsize=font_size)
    ax.set_xticks(np.linspace(0, buf_x_size - 1, 5))
    # i只保留两位小数
    ax.set_xticklabels(
        [
            "{:.2f}E".format(i) if i > 0 else "{:.2f}W".format(-i)
            for i in np.linspace(lon_range[0], lon_range[1], 5)
        ],
        rotation=45,
    )
    ax.set_yticks(np.linspace(0, buf_y_size - 1, 5))
    # i只保留两位小数
    ax.set_yticklabels(
        [
            "{:.2f}N".format(i) if i > 0 else "{:.2f}S".format(-i)
            for i in np.linspace(lat_range[1], lat_range[0], 5)
        ],
        rotation=45,
    )
    # 设置坐标轴刻度标签的字体大小
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    if title:
        ax.set_title(title, fontsize=font_size)

    # 画出MOLA数据
    img = ax.imshow(mola_data, cmap=cmap)

    # 设置colormap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(img, cax=cax)
    # 设置colorbar的刻度字体大小
    cbar.ax.tick_params(labelsize=font_size)

    # 处理点
    if points:
        for i in range(len(points["points"])):
            # 转换为行列号，考虑到降采样率
            points["points"][i] = [
                (
                    int(
                        (p[0] - lon_range[0]) / mola_GeoTransform[1] / downSamplingRate
                    ),
                    int(
                        (p[1] - lat_range[0]) / mola_GeoTransform[5] / downSamplingRate
                    ),
                )
                for p in points["points"][i]
            ]
            ax.scatter(
                [p[0] for p in points["points"][i]],
                [p[1] for p in points["points"][i]],
                color=points["color"][i],
                s=points["size"][i],
            )

    # 处理多边形
    if polygons:
        for i in range(len(polygons["polygons"])):
            # 转换为行列号，考虑到降采样率
            polygons["polygons"][i] = [
                (
                    int(
                        (p[0] - lon_range[0]) / mola_GeoTransform[1] / downSamplingRate
                    ),
                    int(
                        (p[1] - lat_range[0]) / mola_GeoTransform[5] / downSamplingRate
                    ),
                )
                for p in polygons["polygons"][i]
            ]
            # 第一个点复制到最后一个点之后，方便画出多边形
            polygons["polygons"][i].append(polygons["polygons"][i][0])
            ax.plot(
                [p[0] for p in polygons["polygons"][i]],
                [p[1] for p in polygons["polygons"][i]],
                color=polygons["color"][i],
            )

    # 处理线
    if lines:
        for i in range(len(lines["lines"])):
            # 转换为行列号，考虑到降采样率
            lines["lines"][i] = [
                (
                    int(
                        (p[0] - lon_range[0]) / mola_GeoTransform[1] / downSamplingRate
                    ),
                    int(
                        (p[1] - lat_range[0]) / mola_GeoTransform[5] / downSamplingRate
                    ),
                )
                for p in lines["lines"][i]
            ]
            ax.plot(
                [p[0] for p in lines["lines"][i]],
                [p[1] for p in lines["lines"][i]],
                color=lines["color"][i],
            )

    # 保存图片
    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
    # 显示图片
    else:
        plt.show()


# 获取mola数据以及对应的经纬度索引
def get_mola(lon_range, lat_range, downSamplingRate=1):
    """
    :param lon_range: 一个长度为2的list，表示经度范围，范围为[-180,180]
    :param lat_range: 一个长度为2的list，表示纬度范围，范围为[-90,90]
    :param downSamplingRate: 降采样率，用于降低MOLA数据的分辨率，数字越大，分辨率越低
    :return: mola_data, lon_index, lat_index
    """
    mola_data_set = gdal.Open(MOLA_DIR, gdal.GA_ReadOnly)
    mola_GeoTransform = mola_data_set.GetGeoTransform()
    ignore_value = mola_data_set.GetRasterBand(1).GetNoDataValue()
    # lon_range由小到大，lat_range由大到小，排序，方便后面的判断
    lon_range.sort()
    lat_range.sort(reverse=True)
    # 根据lon_range和lat_range，计算出MOLA数据的行列号
    mola_start_col = int((lon_range[0] - mola_GeoTransform[0]) / mola_GeoTransform[1])
    mola_end_col = int((lon_range[1] - mola_GeoTransform[0]) / mola_GeoTransform[1])
    mola_start_row = int((lat_range[0] - mola_GeoTransform[3]) / mola_GeoTransform[5])
    mola_end_row = int((lat_range[1] - mola_GeoTransform[3]) / mola_GeoTransform[5])
    # 计算出ReadAsArray所需的参数
    x_off = mola_start_col
    y_off = mola_start_row
    x_size = mola_end_col - mola_start_col
    y_size = mola_end_row - mola_start_row
    # 考虑到降采样率，计算出ReadAsArray所需的参数
    buf_x_size = int(x_size / downSamplingRate)
    buf_y_size = int(y_size / downSamplingRate)
    # 读取MOLA数据
    mola_data = mola_data_set.ReadAsArray(
        xoff=x_off,
        yoff=y_off,
        xsize=x_size,
        ysize=y_size,
        buf_xsize=buf_x_size,
        buf_ysize=buf_y_size,
        buf_type=None,
        resample_alg=gdal.GRIORA_NearestNeighbour,
        callback=None,
        callback_data=None,
    )
    mola_data = np.float32(mola_data)
    # ignore_value
    mola_data[mola_data == ignore_value] = np.nan
    # 处理经纬度索引
    lon_index = np.linspace(lon_range[0], lon_range[1], buf_x_size)
    lat_index = np.linspace(lat_range[1], lat_range[0], buf_y_size)
    return mola_data, lon_index, lat_index, buf_x_size, buf_y_size, mola_GeoTransform


if __name__ == "__main__":
    points = {
        "points": [[(0, 0), (1, 1), (-1, -1)], [(-1, 1), (1, -1)]],
        "color": ["red", "blue"],
        "size": [20, 10],
    }
    polygons = {
        "polygons": [[(1.5, 1.5), (-1.5, 1.5), (-1.5, -1.5), (1.5, -1.5)]],
        "color": ["green"],
    }
    lines = {"lines": [[(0, 0), (1.8, 1.8), (-0.5, 1)]], "color": ["yellow"]}
    title = "test"
    downSamplingRate = 5
    plot_mola(
        lon_range=[-2.01000035, 2],
        lat_range=[-2, 2],
        downSamplingRate=downSamplingRate,
        cmap="turbo",
        points=points,
        polygons=polygons,
        lines=lines,
        title=title,
    )
