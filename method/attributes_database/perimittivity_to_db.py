import os
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde
import pickle
from matplotlib import pyplot as plt
import django
from tqdm import tqdm

# 设置Django的设置模块路径。
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mars_sim_web.settings")
# 初始化Django
django.setup()
import sim_site.models as models

permittivity_path = Path(__file__).resolve().parent.parent / "permittivity"
BASE_DIR = Path(__file__).resolve().parent


def generate_permittivity_db_sort_by_unit():
    # 读取geology_table
    geology_table = models.SIM3292_Global_Geology.objects.all()
    # 打开文件
    permittivity_file_path = os.path.join(
        permittivity_path, "data", str("permittivity.npy")
    )
    units_file_path = os.path.join(
        permittivity_path, "data", str("units_for_permittivity.npy")
    )
    permittivity = np.load(permittivity_file_path, allow_pickle=True)
    units = np.load(units_file_path, allow_pickle=True)
    # 建立一个dict,key为geology_table中的unit，value为一个空的ndarray
    unit_permittivity_dict = {}
    for i in geology_table:
        unit_permittivity_dict[i.unit] = np.array([])
    # 展平permittivity和units
    permittivity = permittivity.flatten()
    units = units.flatten()
    # 遍历units，把permittivity中对应的值加入到dict中
    pbar = tqdm(total=len(units), desc="正在生成permittivity数据库")
    for i in range(len(units)):
        # 如果permittivity[i]是nan，就跳过
        if np.isnan(permittivity[i]):
            pbar.update(1)
            continue
        if units[i] is None:
            unit_permittivity_dict["None"] = np.hstack(
                (unit_permittivity_dict["None"], permittivity[i])
            )
        unit_permittivity_dict[units[i]] = np.hstack(
            (unit_permittivity_dict[units[i]], permittivity[i])
        )
        pbar.update(1)
    pbar.close()
    # 删除key为'None'的键值对
    del unit_permittivity_dict["None"]
    # 保存这个字典，到database/permittivity文件夹下
    np.save(
        os.path.join(BASE_DIR, "database", "permittivity", "permittivity_dict.npy"),
        unit_permittivity_dict,
        allow_pickle=True,
    )


def generate_kde_for_units():
    # 读取permittivity_dict.npy
    permittivity_dict = np.load(
        os.path.join(BASE_DIR, "database", "permittivity", "permittivity_dict.npy"),
        allow_pickle=True,
    ).item()
    # 生成kde
    kde_dict = {}
    for unit in permittivity_dict.keys():
        permittivity_data = permittivity_dict[unit]
        if len(permittivity_data) == 0:
            print("unit：{}没有数据".format(unit))
            continue
        # permittivity_data 小于1.01的值去除
        permittivity_data = permittivity_data[permittivity_data > 1.01]
        permittivity_kde = gaussian_kde(permittivity_data)
        permittivity_kde_path = os.path.join(
            BASE_DIR,
            "database",
            "permittivity",
            "permittivity_kde",
            str(unit + ".pkl"),
        )
        # 保存kde
        with open(permittivity_kde_path, "wb") as f:
            pickle.dump(permittivity_kde, f)
        print("kde for {} has been generated".format(unit))


def generate_plot_from_kde():
    # 获取method/attributes_database/database/permittivity/permittivity_kde文件夹下的所有文件
    permittivity_kde_path = os.path.join(
        BASE_DIR, "database", "permittivity", "permittivity_kde"
    )
    permittivity_kde_files = os.listdir(permittivity_kde_path)
    # 生成plot
    for i in permittivity_kde_files:
        # 读取对应的kde
        permittivity_kde_path = os.path.join(
            BASE_DIR, "database", "permittivity", "permittivity_kde", i
        )
        with open(permittivity_kde_path, "rb") as f:
            permittivity_kde = pickle.load(f)
        # 生成plot
        x = np.linspace(0, 15, 150)
        plt.plot(x, permittivity_kde(x))
        plt.title("{} permittivity gaussian kde".format(i[:-4]))
        plt.xlabel("permittivity")
        plt.ylabel("density")
        # 保存到method/attributes_database/database/permittivity/img文件夹下
        save_path = os.path.join(
            BASE_DIR, "database", "permittivity", "img", "{}_kde.png".format(i[:-4])
        )
        plt.savefig(save_path)
        plt.close()
    print("plot for permittivity has been generated")


def get_permittivity(units):
    # 找到units中的unique值
    unique_units = np.unique(units)
    # 加载对应的corL_pkl文件，并存成字典
    permittivity_dict = {}
    for i in unique_units:
        permittivity_kde_path = os.path.join(
            BASE_DIR, "database", "permittivity", "permittivity_kde", str(i + ".pkl")
        )
        with open(permittivity_kde_path, "rb") as f:
            permittivity_kde = pickle.load(f)
        permittivity_dict[i] = permittivity_kde
    # 生成permittivity
    permittivity = np.zeros(len(units))
    for i in unique_units:
        index = units == i
        if i == "None":
            permittivity[index] = 5
        permittivity_value = permittivity_dict[i].resample(np.sum(index))[0]
        permittivity_value[permittivity_value < 1.01] = 1.1
        permittivity[index] = permittivity_value
    return permittivity


if __name__ == "__main__":
    # generate_permittivity_db_sort_by_unit()
    generate_kde_for_units()
    generate_plot_from_kde()
