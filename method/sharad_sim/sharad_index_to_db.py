import pdr
import pandas as pd
import numpy as np
from pathlib import Path
import os
import datetime
import django
from tqdm import tqdm
import requests

# 设置Django的设置模块路径。
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mars_sim_web.settings")
# 初始化Django
django.setup()
import sim_site.models as models

BASE_DIR = Path(__file__).resolve().parent

RDR_index_lbl_url = "https://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v2/mrosh_2101/index/index.lbl"
RDR_index_tab_url = "https://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v2/mrosh_2101/index/index.tab"
index_lbl_save_path = os.path.join(BASE_DIR, "data", "RDR", "index.lbl")
index_tab_save_path = os.path.join(BASE_DIR, "data", "RDR", "index.tab")


def download_sharad_rdr_index_file():
    r = requests.get(RDR_index_lbl_url)
    with open(index_lbl_save_path, "wb") as f:
        f.write(r.content)
    r = requests.get(RDR_index_tab_url)
    with open(index_tab_save_path, "wb") as f:
        f.write(r.content)


# 把这个格式的时间转换成datetime：2006-12-06T02:08:22.929
def convert_time(time_str):
    time_str = time_str.replace("T", " ")
    # 如果有毫秒，即.929
    if len(time_str.split(".")) == 2:
        return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    # 如果没有毫秒
    else:
        return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")


def update_sharad_rdr_db():
    rdr_index = pdr.read(index_lbl_save_path)
    index_tab = rdr_index["INDEX_TABLE"]
    # 应用于index_tab的START_TIME和STOP_TIME,PRODUCT_CREATION_TIME
    index_tab["START_TIME"] = index_tab["START_TIME"].apply(convert_time)
    index_tab["STOP_TIME"] = index_tab["STOP_TIME"].apply(convert_time)
    index_tab["PRODUCT_CREATION_TIME"] = index_tab["PRODUCT_CREATION_TIME"].apply(
        convert_time
    )
    # 新建一个dataframe，用来存储每个文件的信息(合并index_tab的连续两行)
    rdr_index_df = pd.DataFrame(
        columns=[
            "VOLUME_ID",
            "RGRAM_FILE_SPECIFICATION_NAME",
            "GEOM_FILE_SPECIFICATION_NAME",
            "PRODUCT_ID",
            "PRODUCT_CREATION_TIME",
            "ORBIT_NUMBER",
            "START_TIME",
            "STOP_TIME",
            "MRO:START_SUB_SPACECRAFT_LATITUDE",
            "MRO:STOP_SUB_SPACECRAFT_LATITUDE",
            "MRO:START_SUB_SPACECRAFT_LONGITUDE",
            "MRO:STOP_SUB_SPACECRAFT_LONGITUDE",
        ]
    )
    rdr_index_df["VOLUME_ID"] = index_tab["VOLUME_ID"].iloc[::2].values
    rdr_index_df["RGRAM_FILE_SPECIFICATION_NAME"] = (
        index_tab["FILE_SPECIFICATION_NAME"].iloc[::2].values
    )
    rdr_index_df["GEOM_FILE_SPECIFICATION_NAME"] = (
        index_tab["FILE_SPECIFICATION_NAME"].iloc[1::2].values
    )
    # 所有的values去掉_RGRAM
    rdr_index_df["PRODUCT_ID"] = index_tab["PRODUCT_ID"].iloc[::2].values
    rdr_index_df["PRODUCT_ID"] = rdr_index_df["PRODUCT_ID"].apply(
        lambda x: x.replace("_RGRAM", "")
    )
    rdr_index_df["PRODUCT_CREATION_TIME"] = (
        index_tab["PRODUCT_CREATION_TIME"].iloc[::2].values
    )
    rdr_index_df["ORBIT_NUMBER"] = index_tab["ORBIT_NUMBER"].iloc[::2].values
    rdr_index_df["START_TIME"] = index_tab["START_TIME"].iloc[::2].values
    rdr_index_df["STOP_TIME"] = index_tab["STOP_TIME"].iloc[::2].values
    rdr_index_df["MRO:START_SUB_SPACECRAFT_LATITUDE"] = (
        index_tab["MRO:START_SUB_SPACECRAFT_LATITUDE"].iloc[::2].values
    )
    rdr_index_df["MRO:STOP_SUB_SPACECRAFT_LATITUDE"] = (
        index_tab["MRO:STOP_SUB_SPACECRAFT_LATITUDE"].iloc[::2].values
    )
    rdr_index_df["MRO:START_SUB_SPACECRAFT_LONGITUDE"] = (
        index_tab["MRO:START_SUB_SPACECRAFT_LONGITUDE"].iloc[::2].values
    )
    rdr_index_df["MRO:STOP_SUB_SPACECRAFT_LONGITUDE"] = (
        index_tab["MRO:STOP_SUB_SPACECRAFT_LONGITUDE"].iloc[::2].values
    )
    # 处理经度使得和MOLA一致，即东经为正，西经为负
    rdr_index_df["MRO:START_SUB_SPACECRAFT_LONGITUDE"] = rdr_index_df[
        "MRO:START_SUB_SPACECRAFT_LONGITUDE"
    ].apply(lambda x: x if x < 180 else x - 360)
    rdr_index_df["MRO:STOP_SUB_SPACECRAFT_LONGITUDE"] = rdr_index_df[
        "MRO:STOP_SUB_SPACECRAFT_LONGITUDE"
    ].apply(lambda x: x if x < 180 else x - 360)
    # 读取数据库中的所有PRODUCT_ID
    product_id_list = models.SharadRDRTable.objects.values_list("PRODUCT_ID", flat=True)
    # 去掉重复的PRODUCT_ID
    rdr_index_df = rdr_index_df[~rdr_index_df["PRODUCT_ID"].isin(product_id_list)]
    # 遍历rdr_index_df，将每一行数据存储到数据库中，如果对应的PRODUCT_ID已经存在，则不存储
    pbar = tqdm(total=len(rdr_index_df))
    for i in range(len(rdr_index_df)):
        # 获取每一行数据
        row = rdr_index_df.iloc[i]
        # 获取PRODUCT_ID
        product_id = row["PRODUCT_ID"]
        # 不存在则存储
        models.SharadRDRTable.objects.create(
            VOLUME_ID=row["VOLUME_ID"],
            RGRAM_FILE_SPECIFICATION_NAME=row["RGRAM_FILE_SPECIFICATION_NAME"],
            GEOM_FILE_SPECIFICATION_NAME=row["GEOM_FILE_SPECIFICATION_NAME"],
            PRODUCT_ID=row["PRODUCT_ID"],
            PRODUCT_CREATION_TIME=row["PRODUCT_CREATION_TIME"],
            ORBIT_NUMBER=row["ORBIT_NUMBER"],
            START_TIME=row["START_TIME"],
            STOP_TIME=row["STOP_TIME"],
            MRO_START_SUB_SPACECRAFT_LATITUDE=row["MRO:START_SUB_SPACECRAFT_LATITUDE"],
            MRO_STOP_SUB_SPACECRAFT_LATITUDE=row["MRO:STOP_SUB_SPACECRAFT_LATITUDE"],
            MRO_START_SUB_SPACECRAFT_LONGITUDE=row[
                "MRO:START_SUB_SPACECRAFT_LONGITUDE"
            ],
            MRO_STOP_SUB_SPACECRAFT_LONGITUDE=row["MRO:STOP_SUB_SPACECRAFT_LONGITUDE"],
        )
        pbar.update(1)
    pbar.close()
    print("存储完成")


if __name__ == "__main__":
    # download_sharad_rdr_index_file()
    update_sharad_rdr_db()
