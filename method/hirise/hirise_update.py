from pathlib import Path
import method.download_method as download_method
import os
import time
from bs4 import BeautifulSoup
import pandas as pd
import requests
import re
import pdr
import django
from method.hirise.hirise_process import update_result, get_region_in_mola
from method.geological_units.get_geological_units import get_units
from method.mola.mola import plot_mola

# 首先初始化django的环境，使得可以在其他py文件中使用django的模型
# 设置Django的设置模块路径。
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mars_sim_web.settings")
# 初始化Django
django.setup()
import sim_site.models as models

# 建立基础路径，使得在其他py文件中可以直接引用
BASE_DIR = Path(__file__).resolve().parent

# LBL和TAB数据路径
# index
index_url = "https://www.uahirise.org/PDS/INDEX"
# LBL文件
index_lbl_url = index_url + "/DTMCUMINDEX.LBL"
# TAB文件
index_tab_url = index_url + "/DTMCUMINDEX.TAB"
# 保存路径
index_lbl_path = str(BASE_DIR / "data/DTMCUMINDEX.LBL")
index_tab_path = str(BASE_DIR / "data/DTMCUMINDEX.TAB")
# IMG文件url前缀
img_prefix = "https://www.uahirise.org/PDS/"
# jpg文件url前缀
jpg_prefix = "https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/"
# DTM文件基础路径
dtm_dir = str(BASE_DIR / "data/DTM/")
# jpg文件基础路径
jpg_dir = str(BASE_DIR / "data/JPG/")


# 更新LBL和TAB数据
def __update_lbl_tab():
    start_time = time.time()
    # 去index_url爬取DTMCUMINDEX.LBL和DTMCUMINDEX.TAB的文件大小
    r = requests.get(index_url)
    # 访问失败，重新访问
    while r.status_code != 200:
        r = requests.get(index_url)
    soup = BeautifulSoup(r.text, "html.parser")
    # 正则表达式匹配文件名及其后面的文件大小
    pattern = r'<a href="([^"]+)">\1</a>\s*\d{2}-\w{3}-\d{4} \d{2}:\d{2}\s+(\d+)'
    # index_url下的所有文件名及其大小
    matches = re.findall(pattern, str(soup))
    result_dict = {filename: int(size) for filename, size in matches}
    # DTMCUMINDEX.LBL和DTMCUMINDEX.TAB的文件大小
    lbl_size = result_dict["DTMCUMINDEX.LBL"]
    tab_size = result_dict["DTMCUMINDEX.TAB"]
    if os.path.exists(index_lbl_path):
        # 如果文件存在，获取文件大小
        lbl_size_old = os.path.getsize(index_lbl_path)
    else:
        # 如果文件不存在，设为0
        lbl_size_old = 0
    if os.path.exists(index_tab_path):
        # 如果文件存在，获取文件大小
        tab_size_old = os.path.getsize(index_tab_path)
    else:
        # 如果文件不存在，设为0
        tab_size_old = 0
    # 如果文件大小不一致，更新文件
    if lbl_size != lbl_size_old or tab_size != tab_size_old:
        print("开始更新DTMCUMINDEX.LBL和DTMCUMINDEX.TAB")
        # 下载DTMCUMINDEX.LBL和DTMCUMINDEX.TAB
        urls = [index_lbl_url, index_tab_url]
        file_names = ["DTMCUMINDEX.LBL", "DTMCUMINDEX.TAB"]
        # 下载
        download_method.download_with_requests(
            urls, file_names, str(BASE_DIR / "data"), overwrite=True
        )
    print(
        "DTMCUMINDEX.LBL和DTMCUMINDEX.TAB已是最新。"
        + "\n用时："
        + str(time.time() - start_time)
        + "秒。"
    )


# 读取LBL和TAB数据，生成DTM索引
def __read_lbl_tab():
    # 读取lbl文件
    lbl = pdr.read(index_lbl_path)
    index_tab = lbl["RDR_INDEX_TABLE"]
    # 选取DATA_TYPE为DTM的行
    require_DATA_TYPE = "DTM"
    # 在后方加上空格至16位,因为DATA_TYPE的长度为16
    require_DATA_TYPE = require_DATA_TYPE + " " * (16 - len(require_DATA_TYPE))
    # 选取DATA_TYPE为DTM的行
    index_tab = index_tab[index_tab["DATA_TYPE"] == require_DATA_TYPE]
    # 索引重建
    index_tab = index_tab.reset_index(drop=True)
    return index_tab


# 筛选出数据库中没有的数据，筛选列为PRODUCT_ID
def __filter_index_tab(index_tab):
    # 创建一个空的要加入数据库的DataFrame，列名与index_tab相同
    index_to_sql = []
    # 循环index_tab，筛选出数据库中没有的数据，筛选列为PRODUCT_ID
    for index, row in index_tab.iterrows():
        # 如果数据库中没有该PRODUCT_ID，将该行index加入index_to_sql
        if not models.HiriseDtmData.objects.filter(
            PRODUCT_ID=row["PRODUCT_ID"]
        ).exists():
            index_to_sql.append(index)
    # 将index_tab中的index_to_sql行加入index_tab_to_sql
    index_tab_to_sql = index_tab.iloc[index_to_sql]
    return index_tab_to_sql


# 处理index_tab_to_sql异常值
def __process_index_tab_to_sql(index_tab):
    # MINIMUM_LATITUDE 和 MAXIMUM_LATITUDE 的mean
    mean_lat = index_tab[["MINIMUM_LATITUDE", "MAXIMUM_LATITUDE"]].mean(axis=1)
    # MINIMUM_LONGITUDE 和 MAXIMUM_LONGITUDE 的mean
    mean_lon = index_tab[["MINIMUM_LONGITUDE", "MAXIMUM_LONGITUDE"]].mean(axis=1)
    # CORNER1_LATITUDE 和 CORNER2_LATITUDE 和 CORNER3_LATITUDE 和 CORNER4_LATITUDE 的mean
    mean_lat1 = index_tab[
        ["CORNER1_LATITUDE", "CORNER2_LATITUDE", "CORNER3_LATITUDE", "CORNER4_LATITUDE"]
    ].mean(axis=1)
    # CORNER1_LONGITUDE 和 CORNER2_LONGITUDE 和 CORNER3_LONGITUDE 和 CORNER4_LONGITUDE 的mean
    mean_lon1 = index_tab[
        [
            "CORNER1_LONGITUDE",
            "CORNER2_LONGITUDE",
            "CORNER3_LONGITUDE",
            "CORNER4_LONGITUDE",
        ]
    ].mean(axis=1)
    # 找到abs(mean_lat - mean_lat1) >1 且 abs(mean_lon - mean_lon1) >1 的行,并且去掉这些行
    index_tab = index_tab[
        ~((abs(mean_lat - mean_lat1) > 1) & (abs(mean_lon - mean_lon1) > 1))
    ]
    # 将所有表示经度的列转换为[-180,180]的形式
    # MINIMUM_LONGITUDE
    index_tab.loc[:, "MINIMUM_LONGITUDE"] = index_tab["MINIMUM_LONGITUDE"].apply(
        lambda x: x - 360 if x > 180 else x
    )
    # MAXIMUM_LONGITUDE
    index_tab.loc[:, "MAXIMUM_LONGITUDE"] = index_tab["MAXIMUM_LONGITUDE"].apply(
        lambda x: x - 360 if x > 180 else x
    )
    # PROJECTION_CENTER_LONGITUDE
    index_tab.loc[:, "PROJECTION_CENTER_LONGITUDE"] = index_tab[
        "PROJECTION_CENTER_LONGITUDE"
    ].apply(lambda x: x - 360 if x > 180 else x)
    # CORNER1_LONGITUDE
    index_tab.loc[:, "CORNER1_LONGITUDE"] = index_tab["CORNER1_LONGITUDE"].apply(
        lambda x: x - 360 if x > 180 else x
    )
    # CORNER2_LONGITUDE
    index_tab.loc[:, "CORNER2_LONGITUDE"] = index_tab["CORNER2_LONGITUDE"].apply(
        lambda x: x - 360 if x > 180 else x
    )
    # CORNER3_LONGITUDE
    index_tab.loc[:, "CORNER3_LONGITUDE"] = index_tab["CORNER3_LONGITUDE"].apply(
        lambda x: x - 360 if x > 180 else x
    )
    # CORNER4_LONGITUDE
    index_tab.loc[:, "CORNER4_LONGITUDE"] = index_tab["CORNER4_LONGITUDE"].apply(
        lambda x: x - 360 if x > 180 else x
    )

    # 索引重建
    index_tab = index_tab.reset_index(drop=True)
    return index_tab


# 存入数据库
def __save_to_sql(index_tab):
    start_time = time.time()
    print("开始存入数据库")
    for index, rows in index_tab.iterrows():
        # 数据库中是否存在该数据
        if models.HiriseDtmData.objects.filter(PRODUCT_ID=rows["PRODUCT_ID"]).exists():
            print(str(index) + "  :  " + rows["PRODUCT_ID"] + "已存在，跳过")
            continue
        hirise_dtm_data = models.HiriseDtmData()
        hirise_dtm_data.FILE_NAME_SPECIFICATION = rows["FILE_NAME_SPECIFICATION"]
        hirise_dtm_data.PRODUCT_ID = rows["PRODUCT_ID"]
        hirise_dtm_data.RATIONALE_DESC = rows["RATIONALE_DESC"]
        hirise_dtm_data.IMAGE_LINES = rows["IMAGE_LINES"]
        hirise_dtm_data.LINE_SAMPLES = rows["LINE_SAMPLES"]
        hirise_dtm_data.NORTH_AZIMUTH = rows["NORTH_AZIMUTH"]
        hirise_dtm_data.MINIMUM_LATITUDE = rows["MINIMUM_LATITUDE"]
        hirise_dtm_data.MAXIMUM_LATITUDE = rows["MAXIMUM_LATITUDE"]
        hirise_dtm_data.MINIMUM_LONGITUDE = rows["MINIMUM_LONGITUDE"]
        hirise_dtm_data.MAXIMUM_LONGITUDE = rows["MAXIMUM_LONGITUDE"]
        hirise_dtm_data.MAP_SCALE = rows["MAP_SCALE"]
        hirise_dtm_data.MAP_RESOLUTION = rows["MAP_RESOLUTION"]
        hirise_dtm_data.MAP_PROJECTION_TYPE = rows["MAP_PROJECTION_TYPE"]
        hirise_dtm_data.PROJECTION_CENTER_LATITUDE = rows["PROJECTION_CENTER_LATITUDE"]
        hirise_dtm_data.PROJECTION_CENTER_LONGITUDE = rows[
            "PROJECTION_CENTER_LONGITUDE"
        ]
        hirise_dtm_data.LINE_PROJECTION_OFFSET = rows["LINE_PROJECTION_OFFSET"]
        hirise_dtm_data.SAMPLE_PROJECTION_OFFSET = rows["SAMPLE_PROJECTION_OFFSET"]
        hirise_dtm_data.CORNER1_LATITUDE = rows["CORNER1_LATITUDE"]
        hirise_dtm_data.CORNER1_LONGITUDE = rows["CORNER1_LONGITUDE"]
        hirise_dtm_data.CORNER2_LATITUDE = rows["CORNER2_LATITUDE"]
        hirise_dtm_data.CORNER2_LONGITUDE = rows["CORNER2_LONGITUDE"]
        hirise_dtm_data.CORNER3_LATITUDE = rows["CORNER3_LATITUDE"]
        hirise_dtm_data.CORNER3_LONGITUDE = rows["CORNER3_LONGITUDE"]
        hirise_dtm_data.CORNER4_LATITUDE = rows["CORNER4_LATITUDE"]
        hirise_dtm_data.CORNER4_LONGITUDE = rows["CORNER4_LONGITUDE"]
        hirise_dtm_data.save()
        print(str(index) + "  :  " + rows["PRODUCT_ID"] + "已存入数据库")
    end_time = time.time()
    print("存入数据库完成，耗时：" + str(end_time - start_time))


# 下载所有数据库里的DTM数据，如果存在则检查大小
def __download_dtm():
    # 从数据库中获取所有DTM数据
    dtm_data = models.HiriseDtmData.objects.all()
    # 先下载DTM数据
    dtm_urls = dtm_data.values_list("FILE_NAME_SPECIFICATION", flat=True)
    dtm_urls = [img_prefix + url for url in dtm_urls]
    dtm_names = dtm_data.values_list("PRODUCT_ID", flat=True)
    dtm_names = [name + ".IMG" for name in dtm_names]
    download_method.download_with_requests(dtm_urls, dtm_names, dtm_dir)
    # while直到所有文件都存在
    while True:
        for i in range(len(dtm_names)):
            if not os.path.exists(os.path.join(dtm_dir, dtm_names[i])):
                break
        else:
            break
    # 下载JPG数据
    jpg_urls = dtm_data.values_list("FILE_NAME_SPECIFICATION", flat=True)
    jpg_urls = [jpg_prefix + url.replace(".IMG", ".ca.jpg") for url in jpg_urls]
    jpg_names = dtm_data.values_list("PRODUCT_ID", flat=True)
    jpg_names = [name + ".ca.jpg" for name in jpg_names]
    download_method.download_with_requests(jpg_urls, jpg_names, jpg_dir)
    # while直到所有文件都存在
    while True:
        for i in range(len(jpg_names)):
            if not os.path.exists(os.path.join(jpg_dir, jpg_names[i])):
                break
        else:
            break
    # 输出
    print("所有数据下载完成")


def update_units():
    # 从数据库中获取所有DTM数据
    dtm_data = models.HiriseDtmData.objects.all()
    # 遍历所有数据
    print("开始更新地质类型")
    for data in dtm_data:
        if data.GEOLOGICAL_TYPE == "unknown":
            # 获取MINIMUM_LATITUDE 和 MAXIMUM_LATITUDE 的mean
            mean_lat = (data.MINIMUM_LATITUDE + data.MAXIMUM_LATITUDE) / 2
            # 获取MINIMUM_LONGITUDE 和 MAXIMUM_LONGITUDE 的mean
            mean_lon = (data.MINIMUM_LONGITUDE + data.MAXIMUM_LONGITUDE) / 2
            # 获取地质类型数据
            units = get_units(mean_lon, mean_lat)
            # 更新数据库
            data.GEOLOGICAL_TYPE = units[0]
            data.save()
            print(data.PRODUCT_ID + "更新完成")


def get_global_locations_of_dtm():
    lon_list = []
    lat_list = []
    # 从数据库中获取所有DTM数据
    dtm_data = models.HiriseDtmData.objects.all()
    # 遍历所有数据
    print("开始更新全球DTM数据的位置图片")
    for data in dtm_data:
        # 获取MINIMUM_LATITUDE 和 MAXIMUM_LATITUDE 的mean
        mean_lat = (data.MINIMUM_LATITUDE + data.MAXIMUM_LATITUDE) / 2
        # 获取MINIMUM_LONGITUDE 和 MAXIMUM_LONGITUDE 的mean
        mean_lon = (data.MINIMUM_LONGITUDE + data.MAXIMUM_LONGITUDE) / 2
        lon_list.append(mean_lon)
        lat_list.append(mean_lat)
    # 出图
    lon_range = [-180, 180]
    lat_range = [-90, 90]
    downSamplingRate = 10
    """
    :param points: 一个字典，形如："points"，"color"，"size"，分别表示点的坐标，颜色，大小，且三者长度相等
    {
        "points": [[(lon1, lat1), (lon2, lat2), ...],[...],...],
        "color": ["red", "blue", ...],
        "size": [10, 20, ...]
    }
    """
    # lon_list和lat_list对应元素组成一个元组，再组成一个列表
    points_list = list(zip(lon_list, lat_list))
    points = {"points": [points_list], "color": ["red"], "size": [10]}
    title = "Global Dtm Data Locations"
    save_path = os.path.join(BASE_DIR, "data/JPG", "global_dtm_data_locations.png")
    plot_mola(
        lon_range,
        lat_range,
        downSamplingRate=downSamplingRate,
        points=points,
        title=title,
        save_path=save_path,
    )
    print("全球DTM数据的位置图片已生成")


def update_all():
    """
    更新流程
    1.更新LBL和TAB数据
    2.读取LBL和TAB数据，生成DTM索引
    3.筛选出数据库中没有的数据，筛选列为PRODUCT_ID
    4.处理index_tab_to_sql异常值
    5.存入数据库
    6.更新数据库中的DMT对应的地质类型
    7.画出全球DTM数据的位置图片
    8.下载所有数据库里的DTM数据，如果存在则检查大小
    9.画出所有DTM数据，在mola中对应的区域
    10.更新所有DTM数据的粗糙度计算结果
    :return:
    """
    __update_lbl_tab()
    index_tab_1 = __read_lbl_tab()
    index_tab_1 = __process_index_tab_to_sql(index_tab_1)
    index_tab_to_sql = __filter_index_tab(index_tab_1)
    # if len(index_tab_to_sql) == 0:
    #     print("所有数据已是最新。")
    #     return
    __save_to_sql(index_tab_to_sql)
    update_units()
    get_global_locations_of_dtm()
    __download_dtm()
    get_region_in_mola()
    update_result()


if __name__ == "__main__":
    update_all()
    # get_global_locations_of_dtm()
