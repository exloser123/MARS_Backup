import pdr
from method.download_method import download_with_requests
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import os
import django

# 设置Django的设置模块路径。
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mars_sim_web.settings")
# 初始化Django
django.setup()
import sim_site.models as models

rdr_url_prefix = (
    "https://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v2/mrosh_2101/"
)
BASE_DIR = Path(__file__).resolve().parent
save_path = os.path.join(BASE_DIR, "data", "RDR", "rdr_data")


def get_sharad_rdr_data(PRODUCT_ID):
    # 从数据库中获取数据
    data = models.SharadRDRTable.objects.get(PRODUCT_ID=PRODUCT_ID)
    # 去除空格
    GEOM_FILE_SPECIFICATION_NAME = data.GEOM_FILE_SPECIFICATION_NAME.strip()
    RGRAM_FILE_SPECIFICATION_NAME = data.RGRAM_FILE_SPECIFICATION_NAME.strip()
    geom_lbl_url = rdr_url_prefix + GEOM_FILE_SPECIFICATION_NAME
    geom_table_url = geom_lbl_url.replace("LBL", "TAB")
    rdr_lbl_url = rdr_url_prefix + RGRAM_FILE_SPECIFICATION_NAME
    rdr_img_url = rdr_lbl_url.replace("LBL", "IMG")
    # 下载数据
    urls = [geom_lbl_url, geom_table_url, rdr_lbl_url, rdr_img_url]
    filenames = [Path(url).name for url in urls]
    download_with_requests(urls, filenames, save_path, overwrite=False)
    # 读取GEOM
    geom = pdr.read(os.path.join(save_path, filenames[0]))
    geom = geom["TABLE"]
    # 调整LONGITUDE的值，使其从0-360变为-180到180之间
    geom.loc[geom["LONGITUDE"] > 180, "LONGITUDE"] -= 360
    # 读取RGRAM
    rgram = pdr.read(os.path.join(save_path, filenames[2]))
    rgram = rgram["IMAGE"]
    if rgram.shape[1] > 4500:
        rgram = rgram[:, :4500]
        # geom也要截断
        geom = geom[geom["RADARGRAM COLUMN"] <= 4500]

    return geom, rgram


def process_img_for_plot(img):
    # log处理
    imgScale = np.log10(img + 1e-30)
    # 获取实际的值
    imgValid = imgScale[img != 0]
    # 使图像分布在0-255
    p10 = np.percentile(imgValid, 10)
    m = 255 / (imgValid.max() - p10)
    b = -p10 * m
    # 去除最小值
    img = imgScale * m + b
    img[img < 0] = 0
    return img
