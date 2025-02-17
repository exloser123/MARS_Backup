import os
import django
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde
import pickle
from matplotlib import pyplot as plt

# 设置Django的设置模块路径。
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mars_sim_web.settings")
# 初始化Django
django.setup()
import sim_site.models as models

hirise_dir = Path(__file__).resolve().parent.parent / "hirise"
BASE_DIR = Path(__file__).resolve().parent


def generate_corL_rmsH_db_sort_by_unit():
    geology_table = models.SIM3292_Global_Geology.objects.all()
    hirise_dtm_table = models.HiriseDtmData.objects.all()
    # 建立一个dict,key为geology_table中的unit，value为一个空的ndarray
    unit_corL_dict = {}
    unit_rmsH_dict = {}
    for i in geology_table:
        unit_corL_dict[i.unit] = np.array([])
        unit_rmsH_dict[i.unit] = np.array([])
    for dtm in hirise_dtm_table:
        # 读取对应的result文件,E:\PycharmProjects\mars_sim_web\method\hirise\data\result
        corL_result_file_path = os.path.join(
            hirise_dir, "data", "result", str(dtm.PRODUCT_ID + "_corL.npy")
        )
        rmsH_result_file_path = os.path.join(
            hirise_dir, "data", "result", str(dtm.PRODUCT_ID + "_rmsH.npy")
        )
        # 读取文件
        corL_result = np.load(corL_result_file_path, allow_pickle=True)
        rmsH_result = np.load(rmsH_result_file_path, allow_pickle=True)
        # 读取对应的geology
        geology = dtm.GEOLOGICAL_TYPE
        # 把结果和dict中对应的ndarray合并（1d和1d，组成一个1d）
        unit_corL_dict[geology] = np.hstack((unit_corL_dict[geology], corL_result))
        unit_rmsH_dict[geology] = np.hstack((unit_rmsH_dict[geology], rmsH_result))
    # 删除key为'None'的键值对
    del unit_corL_dict["None"]
    del unit_rmsH_dict["None"]
    # 保存这两个字典，到database/hirise文件夹下
    np.save(
        os.path.join(BASE_DIR, "database", "hirise", "corL_dict.npy"),
        unit_corL_dict,
        allow_pickle=True,
    )
    np.save(
        os.path.join(BASE_DIR, "database", "hirise", "rmsH_dict.npy"),
        unit_rmsH_dict,
        allow_pickle=True,
    )


def generate_kde_for_units():
    # 读取corL_dict.npy和rmsH_dict.npy
    corL_dict = np.load(
        os.path.join(BASE_DIR, "database", "hirise", "corL_dict.npy"),
        allow_pickle=True,
    ).item()
    rmsH_dict = np.load(
        os.path.join(BASE_DIR, "database", "hirise", "rmsH_dict.npy"),
        allow_pickle=True,
    ).item()
    # 对每个unit的corL和rmsH进行核密度估计，保存到database/hirise文件夹下
    for i in corL_dict.keys():
        corL_data = corL_dict[i]
        rmsH_data = rmsH_dict[i]
        if len(corL_data) == 0 or len(rmsH_data) == 0:
            print("unit：{}没有数据".format(i))
            continue
        # 只对[0,20]范围内的数据进行核密度估计
        corL_data = corL_data[(corL_data <= 20) & (corL_data >= 0)]
        # 只对[0,3]范围内的数据进行核密度估计
        rmsH_data = rmsH_data[(rmsH_data <= 3) & (rmsH_data >= 0)]
        corL_kde = gaussian_kde(corL_data)
        rmsH_kde = gaussian_kde(rmsH_data)
        corL_kde_path = os.path.join(
            BASE_DIR, "database", "hirise", "corL_kde", str(i + ".pkl")
        )
        rmsH_kde_path = os.path.join(
            BASE_DIR, "database", "hirise", "rmsH_kde", str(i + ".pkl")
        )
        with open(corL_kde_path, "wb") as f:
            pickle.dump(corL_kde, f)
        with open(rmsH_kde_path, "wb") as f:
            pickle.dump(rmsH_kde, f)
        print("kde for {} has been generated".format(i))


def generate_plot_from_kde():
    # 获取method/attributes_database/database/hirise/corL_kde文件夹下的所有文件
    corL_kde_path = os.path.join(BASE_DIR, "database", "hirise", "corL_kde")
    corL_kde_files = os.listdir(corL_kde_path)
    # 获取method/attributes_database/database/hirise/rmsH_kde文件夹下的所有文件
    rmsH_kde_path = os.path.join(BASE_DIR, "database", "hirise", "rmsH_kde")
    rmsH_kde_files = os.listdir(rmsH_kde_path)
    for i in corL_kde_files:
        # 读取对应的kde
        file_path = os.path.join(corL_kde_path, i)
        with open(file_path, "rb") as f:
            corL_kde = pickle.load(f)
        # 生成对应的图像
        x = np.linspace(0, 20, 200)
        plt.plot(x, corL_kde(x))
        plt.title("{} corL gaussian kde".format(i[:-4]))
        plt.xlabel("corL/m")
        plt.ylabel("density")
        # 保存到method/attributes_database/database/hirise/img文件夹下
        save_path = os.path.join(
            BASE_DIR, "database", "hirise", "img", "{}_corL_kde.png".format(i[:-4])
        )
        plt.savefig(save_path, dpi=300)
        plt.close()
    for i in rmsH_kde_files:
        # 读取对应的kde
        file_path = os.path.join(rmsH_kde_path, i)
        with open(file_path, "rb") as f:
            rmsH_kde = pickle.load(f)
        # 生成对应的图像
        x = np.linspace(0, 3, 90)
        plt.plot(x, rmsH_kde(x))
        plt.title("{} rmsH gaussian kde".format(i[:-4]))
        plt.xlabel("rmsH/m")
        plt.ylabel("density")
        # 保存到method/attributes_database/database/hirise/img文件夹下
        save_path = os.path.join(
            BASE_DIR, "database", "hirise", "img", "{}_rmsH_kde.png".format(i[:-4])
        )
        plt.savefig(save_path, dpi=300)
        plt.close()
    print("plot for roughness attributes has been generated")


def get_corL_rmsH(units):
    # 找到units中的unique值
    unique_units = np.unique(units)
    # 如果unique_units中存在有'Ap',用'Apu'代替，并替换units中的值
    if "Ap" in unique_units:
        units[units == "Ap"] = "Apu"
        unique_units = np.unique(units)
    if "lHvf" in unique_units:
        units[units == "lHvf"] = "lHv"
        unique_units = np.unique(units)

    # 加载对应的corL_pkl文件，并存成字典
    corL_pkl_dict = {}
    for i in unique_units:
        file_path = os.path.join(
            BASE_DIR, "database", "hirise", "corL_kde", str(i + ".pkl")
        )
        with open(file_path, "rb") as f:
            corL_pkl_dict[i] = pickle.load(f)
    # 加载对应的rmsH_pkl文件，并存成字典
    rmsH_pkl_dict = {}
    for i in unique_units:
        file_path = os.path.join(
            BASE_DIR, "database", "hirise", "rmsH_kde", str(i + ".pkl")
        )
        with open(file_path, "rb") as f:
            rmsH_pkl_dict[i] = pickle.load(f)
    # 生成对应的corL和rmsH
    corL = np.zeros_like(units, dtype=np.float32)
    rmsH = np.zeros_like(units, dtype=np.float32)
    for i in unique_units:
        # 找到units中对应的索引
        index = units == i
        if i == "None":
            corL[index] = 5
            rmsH[index] = 0.5
        # 生成对应的corL
        corL[index] = corL_pkl_dict[i].resample(np.sum(index))[0]
        # 生成对应的rmsH
        rmsH[index] = rmsH_pkl_dict[i].resample(np.sum(index))[0]
    return corL, rmsH


if __name__ == "__main__":
    # generate_corL_rmsH_db_sort_by_unit()
    # generate_kde_for_units()
    # generate_plot_from_kde()
    units = np.array(["Ap", "mNh", "HNt", "Ap", "mNh", "HNt"])
    corL, rmsH = get_corL_rmsH(units)
