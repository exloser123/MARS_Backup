import json
import os
import subprocess
import time
import requests
from tqdm import tqdm

thunder_path = r"C:\Program Files (x86)\Thunder Network\Thunder"
thunder_start_path = thunder_path + r"\Program\ThunderStart.exe"
thunder_exe_path = thunder_path + r"\Program\Thunder.exe"
thunder_config_path = thunder_path + r"\Profiles\config.json"


# 改变保存路径
def __change_save_path(save_path):
    """
    修改迅雷的下载路径
    :param save_path: 必须为绝对路径，例如："E:\\PycharmProjects\\mars_sim_web\\method\\hirise\\data\\JPG"
    :return:
    """
    # 关闭迅雷程序
    result = subprocess.run(
        ["taskkill", "/F", "/IM", "Thunder.exe"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        print("迅雷关闭成功。")
    else:
        print("关闭迅雷失败。")

    time.sleep(2)
    # 格式化路径
    # 规范化路径格式
    save_path = os.path.normpath(save_path)
    try:
        # 打开json文件，修改下载路径
        with open(thunder_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config["PathAndCategory"]["LastUsedPath"] = save_path
        config["TaskDefaultSettings"]["DefaultPath"] = save_path
        # 保存修改后的配置
        with open(thunder_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
    except FileNotFoundError:
        print("文件未找到：", thunder_config_path)
    except json.JSONDecodeError:
        print("JSON 解码错误")


# 下载函数
def download_with_thunder(urls, file_names, save_path):
    """
    使用迅雷下载
    :param urls: 一个list，表示下载链接
    :param file_names: 一个list，表示文件名
    :param save_path: 一个str，表示保存路径
    """
    # 修改下载路径
    __change_save_path(save_path)

    # urls 和 file_names 组成一个元组的列表
    mission = list(zip(urls, file_names))
    nums = 40

    # 如果文件已经存在，则不下载
    mission = [
        (url, file_name)
        for url, file_name in mission
        if not os.path.exists(os.path.join(save_path, file_name))
    ]

    # 如果任务小于10个，则全部加入mission_to_download，否则只加入10个
    if len(mission) == 0:
        print("所有文件已经存在。")
        return
    elif len(mission) <= nums:
        mission_to_download = mission
        mission = []
    else:
        mission_to_download = mission[:nums]
        mission = mission[nums:]

    # 已经启动的任务集合
    started_tasks = set()

    # 启动任务
    while mission_to_download or (
        mission
        and any((url, file_name) not in started_tasks for url, file_name in mission)
    ):
        # 检查文件是否已经存在
        to_remove = []
        for url, file_name in mission_to_download:
            if os.path.exists(os.path.join(save_path, file_name)):
                print(file_name + "下载完成。")
                to_remove.append((url, file_name))
            elif (url, file_name) not in started_tasks:
                os.system(
                    r'""C:\Program Files (x86)\Thunder Network\Thunder\Program\ThunderStart.exe"" {url}'.format(
                        url=url
                    )
                )
                print("启动任务：" + file_name)
                started_tasks.add((url, file_name))
                time.sleep(2)  # 等待2秒再启动下一个任务

        # 从mission_to_download中移除已存在的文件
        for item in to_remove:
            mission_to_download.remove(item)

        # 从mission中补充任务到mission_to_download，确保mission_to_download中有10个任务
        while len(mission_to_download) < nums and mission:
            mission_to_download.append(mission.pop(0))

        # 每1秒检查一次
        time.sleep(0.5)


def download_with_requests(urls, file_names, save_path, overwrite=False):
    """
    使用requests下载
    :param urls: 一个list，表示下载链接
    :param file_names: 一个list，表示文件名
    :param save_path: 一个str，表示保存路径
    """
    if overwrite:
        mission = list(zip(urls, file_names))
    else:
        # 如果文件已经存在，则不下载
        mission = [
            (url, file_name)
            for url, file_name in list(zip(urls, file_names))
            if not os.path.exists(os.path.join(save_path, file_name))
        ]
    if len(mission) == 0:
        print("所有文件已经存在。")
        return
    for url, file_name in mission:
        start_time = time.time()
        print("开始下载：{file_name}".format(file_name=file_name))
        # 下载文件
        r = requests.get(url)
        # 保存文件
        with open(os.path.join(save_path, file_name), "wb") as f:
            f.write(r.content)
        end_time = time.time()
        print(
            "下载完成：{file_name}，耗时：{time:.2f}s".format(
                file_name=file_name, time=end_time - start_time
            )
        )


if __name__ == "__main__":
    from pathlib import Path

    url_prefix = (
        "https://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v2/mrosh_2101/"
    )
    geom_lbl_url = url_prefix + "DATA/GEOM/S_0016XX/S_00168901_GEOM.LBL"
    geom_table_url = url_prefix + "DATA/GEOM/S_0016XX/S_00168901_GEOM.TAB"
    rdr_lbl_url = url_prefix + "DATA/RGRAM/S_0016XX/S_00168901_RGRAM.LBL"
    rdr_img_url = url_prefix + "DATA/RGRAM/S_0016XX/S_00168901_RGRAM.IMG"
    save_path = "data/RDR/rdr_data"

    download_with_requests([geom_lbl_url], [Path(geom_lbl_url).name], [save_path])
