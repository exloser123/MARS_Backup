import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from method.mola.mola import get_mola
from pyresample import geometry, kd_tree
from method.sharad_sim.sharad_rdr_data import get_sharad_rdr_data
from method.geological_units.get_geological_units import get_units
from method.attributes_database.hirise_to_db import get_corL_rmsH
from method.attributes_database.perimittivity_to_db import get_permittivity
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
result_list = [
    "geom_data",
    "rgram_data",
    "sim_area_lonlat",
    "sat_lon_lat",
    "point_count",
    "sat_radius",
    "sim_area_cartesian",
    "facet_index",
    "center_point",
    "normal_vector",
    "area",
    "corL",
    "rmsH",
    "permittivity",
    "incident_angle",
    "delay",
    "distance",
    "delay_index",
    "sgm_hh",
    "sgm_vv",
    "rgram_hh",
    "rgram_vv",
    "Es",
]

mars_radius = 3396190
velocity_light = 299792458
sharad_freq = 20e6
delay_interval = 0.0375 * 1e-6  # 0.075 microseconds
k = 2 * np.pi * velocity_light / sharad_freq


# 经纬度转笛卡尔坐标,输入一个dt_lonlat类型的数组，输出一个dt_cartesian类型的数组
def lonlat_to_cartesian(lon_lat, radius=mars_radius):
    lon = lon_lat[0]
    lat = lon_lat[1]
    # 转换为弧度
    lon = lon * np.pi / 180
    lat = lat * np.pi / 180
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return np.array([x, y, z])


# 笛卡尔坐标转经纬度,输入一个dt_cartesian类型的数组，输出一个dt_lonlat类型的数组
def cartesian_to_lonlat(cartesian):
    x = cartesian[0]
    y = cartesian[1]
    z = cartesian[2]
    lon = np.arctan2(y, x)
    lat = np.arcsin(z / np.linalg.norm(cartesian, axis=0))
    # 转换为角度
    lon = np.degrees(lon)
    lat = np.degrees(lat)
    return np.array([lon, lat])


def get_sim_area_lonlat(geom):
    # 星下点数量
    _point_count = geom.shape[0]
    _sat_radius = geom["SPACECRAFT RADIUS"].values * 1000
    # 初始化笛卡尔和经纬度坐标存储数组
    _sim_area_cartesian = np.zeros((_point_count, 3001, 3, 3))
    _sim_area_lonlat = np.zeros((_point_count, 3001, 3, 2))
    # 提取所有星下点经纬度，转化为笛卡尔坐标
    _sat_lon_lat = geom[["LONGITUDE", "LATITUDE"]].values.T
    # 转换为dt_cartesian类型
    sat_cartesian = lonlat_to_cartesian(_sat_lon_lat)
    # 计算速度向量，最后一个点的速度向量近似为倒数第二个点的速度向量,如何定义两个dt_cartesian类型的数组相减？
    sat_v_vec = sat_cartesian[:, 1:] - sat_cartesian[:, :-1]
    # 将sat_v_vec的最后一个值加在sat_v_vec末尾
    sat_v_vec = np.hstack((sat_v_vec, sat_v_vec[:, -1].reshape(3, 1)))
    # 计算crosstrack速度向量的单位向量
    sat_v_crosstrack_vec = np.cross(sat_cartesian, sat_v_vec, axis=0)
    sat_v_crosstrack_vec = sat_v_crosstrack_vec / np.linalg.norm(
        sat_v_crosstrack_vec, axis=0
    )
    # 计算alongtrack速度向量的单位向量
    sat_v_alongtrack_vec = np.cross(sat_v_crosstrack_vec, sat_cartesian, axis=0)
    sat_v_alongtrack_vec = sat_v_alongtrack_vec / np.linalg.norm(
        sat_v_alongtrack_vec, axis=0
    )
    # 设置步长
    step_cross_track = 30
    step_along_track = 200
    # 以sat_cartesian为中心，向两侧扩展
    _sim_area_cartesian[:, 1500, 1, :] = sat_cartesian.T
    _sim_area_lonlat[:, 1500, 1, :] = _sat_lon_lat.T
    for i in range(1, 1501):
        _sim_area_cartesian[:, 1500 - i, 1, :] = (
            sat_cartesian.T - sat_v_crosstrack_vec.T * step_cross_track * i
        )
        _sim_area_cartesian[:, 1500 + i, 1, :] = (
            sat_cartesian.T + sat_v_crosstrack_vec.T * step_cross_track * i
        )
    # 初始化alongtrack方向的向量
    sat_v_alongtrack_vec_list = np.zeros((_point_count, 3001, 3))
    # 每一个point_count，sat_v_alongtrack_vec_list都是一样的等于对应的sat_v_alongtrack_vec * step_alongtrack
    for i in range(_point_count):
        sat_v_alongtrack_vec_list[i, :, :] = (
            sat_v_alongtrack_vec.T[i] * step_along_track
        )
    # alongtrack方向的向量+和-，分别对应sim_area_cartesian的第一列和第三列
    _sim_area_cartesian[:, :, 0, :] = (
        _sim_area_cartesian[:, :, 1, :] + sat_v_alongtrack_vec_list
    )
    _sim_area_cartesian[:, :, 2, :] = (
        _sim_area_cartesian[:, :, 1, :] - sat_v_alongtrack_vec_list
    )
    _sim_area_lonlat = cartesian_to_lonlat(
        _sim_area_cartesian.reshape(_point_count * 3001 * 3, 3).T
    ).T
    _sim_area_lonlat = _sim_area_lonlat.reshape(_point_count, 3001, 3, 2)
    return _sim_area_lonlat, _sat_lon_lat, _point_count, _sat_radius


def get_sim_area_cartesian(sim_area_lonlat, _point_count):
    _sim_area_cartesian = np.zeros((_point_count, 3001, 3, 3))
    pbar = tqdm(total=sim_area_lonlat.shape[0], desc="插值仿真区域的DEM")
    for i in range(sim_area_lonlat.shape[0]):
        max_lon = sim_area_lonlat[i, :, :, 0].max()
        min_lon = sim_area_lonlat[i, :, :, 0].min()
        max_lat = sim_area_lonlat[i, :, :, 1].max()
        min_lat = sim_area_lonlat[i, :, :, 1].min()
        # 生成经纬度范围
        lon_range = [min_lon, max_lon]
        lat_range = [min_lat, max_lat]
        # 获取mola数据
        mola_data, lon_index, lat_index, _, _, _ = get_mola(lon_range, lat_range)
        # 如果mola_data中有nan，输出lon_range，lat_range
        if np.isnan(mola_data).any():
            print(lon_range, lat_range)
            pbar.update(1)
            continue
        lon_grid, lat_grid = np.meshgrid(lon_index, lat_index)
        # 获取mola数据的经纬度
        mola_data = mola_data.flatten()
        lon_grid = lon_grid.flatten()
        lat_grid = lat_grid.flatten()
        swath_def = geometry.SwathDefinition(lons=lon_grid, lats=lat_grid)

        # 定义目标区域的空间信息
        lons_target, lats_target = (
            sim_area_lonlat[i, :, :, 0].flatten(),
            sim_area_lonlat[i, :, :, 1].flatten(),
        )
        target_def = geometry.SwathDefinition(lons=lons_target, lats=lats_target)

        # 重采样
        dem_inter = kd_tree.resample_gauss(
            swath_def,
            mola_data,
            target_def,
            radius_of_influence=300,
            sigmas=200,
            fill_value=None,
        )
        # 获得半径
        radius = dem_inter + mars_radius
        # 将radius，lat_inter，lon_inter转换为3001*3的数组
        radius = radius.reshape(3001, 3)
        lats_target = lats_target.reshape(3001, 3)
        lons_target = lons_target.reshape(3001, 3)
        # 获得笛卡尔坐标
        # 将经纬度转换为笛卡尔坐标
        _sim_area_cartesian[i, :, :, 0] = (
            radius
            * np.cos(lats_target * np.pi / 180)
            * np.cos(lons_target * np.pi / 180)
        )
        _sim_area_cartesian[i, :, :, 1] = (
            radius
            * np.cos(lats_target * np.pi / 180)
            * np.sin(lons_target * np.pi / 180)
        )
        _sim_area_cartesian[i, :, :, 2] = radius * np.sin(lats_target * np.pi / 180)
        pbar.update(1)
    pbar.close()
    return _sim_area_cartesian


def cal_center_point(_sim_area_cartesian, facet_index, _point_count):
    # 初始化存储中心点的数组
    center_point = np.zeros((_point_count, 12000, 3))
    for i in range(_point_count):
        # 一次性提取所有三角形的三个顶点
        p1 = _sim_area_cartesian[i, facet_index[:, 0, 0], facet_index[:, 0, 1]]
        p2 = _sim_area_cartesian[i, facet_index[:, 1, 0], facet_index[:, 1, 1]]
        p3 = _sim_area_cartesian[i, facet_index[:, 2, 0], facet_index[:, 2, 1]]
        # 一次性计算所有三角形的中心点
        center_point[i] = (p1 + p2 + p3) / 3.0
    return center_point


def cal_facet_idx():
    # 每一个(3001,3)网格点的nas索引，用来表达3000*4个facet,每一个point_count的剖分都是一样的
    facet_index = np.zeros((3000 * 4, 3, 2), dtype=int)
    # 例如：第一个三角形的三个点的索引为[[0,0],[0,1]],[1,0]],第二个三角形的三个点的索引为[[0,1],[1,1]],[1,0]]
    # 遍历每一个方格
    counter = 0
    for i in range(3000):
        for j in range(2):
            # 定义四个点
            top_left = [i, j]
            top_right = [i, j + 1]
            bottom_left = [i + 1, j]
            bottom_right = [i + 1, j + 1]

            # 第一个三角形
            facet_index[counter, 0] = top_left
            facet_index[counter, 1] = bottom_left
            facet_index[counter, 2] = top_right
            counter += 1

            # 第二个三角形
            facet_index[counter, 0] = top_right
            facet_index[counter, 1] = bottom_right
            facet_index[counter, 2] = bottom_left
            counter += 1
    return facet_index


def cal_normal_vector_and_area(geom, _sim_area_cartesian, _facet_index, _sat_lon_lat):
    # 星下点数量
    _point_count = geom.shape[0]
    _normal_vector = np.zeros((_point_count, 12000, 3))
    _area = np.zeros((_point_count, 12000))
    # 提取所有星下点经纬度，转化为笛卡尔坐标
    _sat_lon_lat = geom[["LONGITUDE", "LATITUDE"]].values.T
    sat_cartesian = lonlat_to_cartesian(_sat_lon_lat).T
    for i in range(_point_count):
        point = _sim_area_cartesian[i].reshape(3001 * 3, 3)

        # 使用facet_index一次性提取所有三角形的三个顶点
        p1 = point[_facet_index[:, 0, 0] * 3 + _facet_index[:, 0, 1]]
        p2 = point[_facet_index[:, 1, 0] * 3 + _facet_index[:, 1, 1]]
        p3 = point[_facet_index[:, 2, 0] * 3 + _facet_index[:, 2, 1]]

        # 计算法线向量
        _normal_vector[i] = np.cross(p2 - p1, p3 - p1)

        # 计算面积
        _area[i] = np.linalg.norm(_normal_vector[i], axis=1) / 2

        # 确保所有的normal_vector都是指向外的
        dot = np.sum(_normal_vector[i] * sat_cartesian[i], axis=1)
        norm = np.linalg.norm(_normal_vector[i], axis=1) * np.linalg.norm(
            sat_cartesian[i]
        )
        theta = np.arccos(dot / norm)
        theta = np.rad2deg(theta).T
        condition = theta > 90
        _normal_vector[i][condition] = -_normal_vector[i][condition]
        _normal_vector[i] = _normal_vector[i] / _area[i].reshape(12000, 1)
    return _normal_vector, _area


def get_attributes(_center_point):
    _center_point_shape = _center_point.shape
    _center_point = _center_point.reshape(-1, 3).T
    # 转换为经纬度
    _center_point_lon_lat = cartesian_to_lonlat(_center_point)
    # 获取地质单元
    units = get_units(_center_point_lon_lat[0, :], _center_point_lon_lat[1, :])
    # 获取corL和rmsH
    _corL, _rmsH = get_corL_rmsH(units)
    # 获取permittivity
    _permittivity = get_permittivity(units)
    # 转换为原始形状
    _corL = _corL.reshape(_center_point_shape[:-1])
    _rmsH = _rmsH.reshape(_center_point_shape[:-1])
    _permittivity = _permittivity.reshape(_center_point_shape[:-1])
    return _corL, _rmsH, _permittivity


def get_incidence_angle_and_delay(
    _sat_lon_lat, _center_point, _normal_vector, _sat_radius
):
    sat_pos = lonlat_to_cartesian(_sat_lon_lat, _sat_radius).T
    # sat_pos扩展为(center_point.shape[0], center_point.shape[1]),在axis =1上复制值
    sat_pos = np.tile(sat_pos, (_center_point.shape[1], 1, 1)).transpose((1, 0, 2))
    sat_pos = sat_pos.reshape((_center_point.shape[0] * _center_point.shape[1], 3))
    shape = _center_point.shape
    _center_point = _center_point.reshape(
        (_center_point.shape[0] * _center_point.shape[1], 3)
    )
    incident_vector = _center_point - sat_pos
    _distance = np.linalg.norm(incident_vector, axis=1)
    relative_height = np.linalg.norm(sat_pos, axis=1)
    delay = 2 * (_distance - relative_height + mars_radius) / velocity_light
    # normal_vector[i]里每个点的法向量
    _normal_vector = _normal_vector.reshape(
        (_normal_vector.shape[0] * _normal_vector.shape[1], 3)
    )
    dot = np.sum(-incident_vector * _normal_vector, axis=1)
    norm = _distance * np.linalg.norm(_normal_vector, axis=1)
    theta = np.arccos(dot / norm)
    # theta = np.rad2deg(theta)
    incident_angle = theta.reshape(shape[:-1])
    delay = delay.reshape(shape[:-1])
    _distance = _distance.reshape(shape[:-1])
    return incident_angle, delay, _distance


def cal_sgm(_permittivity, _corL, _rmsH, _incident_angle, _center_point, _distance):
    sim_area_permittivity = _permittivity.reshape(-1)
    correlation_length = _corL.reshape(-1)
    root_mean_square_height = _rmsH.reshape(-1)
    incident_angle = _incident_angle.reshape(-1)
    # tk=2*pi*freq/0.3;
    tk = 2 * np.pi * sharad_freq / velocity_light
    # R0 = (sqrt(epsilon)-1)/(sqrt(epsilon)+1);
    R0 = (np.sqrt(sim_area_permittivity) - 1) / (np.sqrt(sim_area_permittivity) + 1)
    # R02 = (abs(R0))^2;
    R02 = np.abs(R0) ** 2
    # Cos_Thi = cos(thetai);
    Cos_Thi = np.cos(incident_angle)
    # Sin_Thi = sin(thetai);
    Sin_Thi = np.sin(incident_angle)
    # B = ( lc/(2*h*Cos_Thi) )^2;
    B = (correlation_length / (2 * root_mean_square_height * Cos_Thi)) ** 2
    # A = exp(-B*Sin_Thi^2);
    A = np.exp(-B * Sin_Thi**2)
    # sgm_KA = B * A/(Cos_Thi^2) * R02;
    sgm_KA = B * A / (Cos_Thi**2) * R02
    # aa = sqrt(epsilon-Sin_Thi^2);
    aa = np.sqrt(sim_area_permittivity - Sin_Thi**2)
    # fee = -2*tk*(epsilon-1)*Cos_Thi/(Cos_Thi+aa)^2;
    fee = -2 * tk * (sim_area_permittivity - 1) * Cos_Thi / (Cos_Thi + aa) ** 2
    # para = ( epsilon*Cos_Thi+ aa)^2;
    para = (sim_area_permittivity * Cos_Thi + aa) ** 2
    # y1addy2 =  epsilon + (epsilon-1)*Sin_Thi^2;
    y1addy2 = sim_area_permittivity + (sim_area_permittivity - 1) * Sin_Thi**2
    # fhh = 2*tk*(epsilon-1)*Cos_Thi/para*y1addy2;
    fhh = 2 * tk * (sim_area_permittivity - 1) * Cos_Thi / para * y1addy2
    # W_spect = (h*lc)^2/(4*pi)*exp(-(tk*Sin_Thi/lc)^2);
    W_spect = (
        (root_mean_square_height * correlation_length) ** 2
        / (4 * np.pi)
        * np.exp(-((tk * Sin_Thi / correlation_length) ** 2))
    )
    # sgm_hh_SPM = 4*pi*tk^2*Cos_Thi*W_spect*(abs(fee)^2);
    sgm_hh_SPM = 4 * np.pi * tk**2 * Cos_Thi * W_spect * (np.abs(fee) ** 2)
    # sgm_vv_SPM = 4*pi*tk^2*Cos_Thi*W_spect*(abs(fhh)^2);
    sgm_vv_SPM = 4 * np.pi * tk**2 * Cos_Thi * W_spect * (np.abs(fhh) ** 2)
    # sgm_hh = sgm_KA + sgm_hh_SPM;
    sgm_hh = sgm_KA + sgm_hh_SPM
    # sgm_vv = sgm_KA + sgm_vv_SPM;
    sgm_vv = sgm_KA + sgm_vv_SPM
    # # 沿着 axis=1 找到每一行的最小值
    # min_values = np.min(_distance, axis=1)
    # # _distance处理
    # _distance = _distance - (min_values - 0.01)[:, np.newaxis]
    _distance = _distance.reshape(-1)
    # 距离衰减因子
    sgm_hh = sgm_hh * (1 / _distance**4)
    sgm_vv = sgm_vv * (1 / _distance**4)
    sgm_hh = sgm_hh.reshape(_center_point.shape[:-1])
    sgm_vv = sgm_vv.reshape(_center_point.shape[:-1])
    return sgm_hh, sgm_vv


def cal_Es(
    _permittivity,
    _corL,
    _rmsH,
    _incident_angle,
    _center_point,
    _distance,
    _normal_vector,
    _area,
):
    k1 = k * np.sqrt(_permittivity)
    # Rh = (k * cos_theta_i - sqrt(k1 ^ 2 - k ^ 2 * sin_theta_i. ^ 2)). / (
    #             k * cos_theta_i + sqrt(k1 ^ 2 - k ^ 2 * sin_theta_i. ^ 2));
    Rh = k * np.cos(_incident_angle) - np.sqrt(
        k1**2 - k**2 * np.sin(_incident_angle) ** 2
    ) / (
        k * np.cos(_incident_angle)
        + np.sqrt(k1**2 - k**2 * np.sin(_incident_angle) ** 2)
    )
    # Rv = (epsilon_r * k * cos_theta_i - sqrt(k1 ^ 2 - k ^ 2 * sin_theta_i. ^ 2)). / (
    #             epsilon_r * k * cos_theta_i + sqrt(k1 ^ 2 - k ^ 2 * sin_theta_i. ^ 2));
    Rv = _permittivity * k * np.cos(_incident_angle) - np.sqrt(
        k1**2 - k**2 * np.sin(_incident_angle) ** 2
    ) / (
        _permittivity * k * np.cos(_incident_angle)
        + np.sqrt(k1**2 - k**2 * np.sin(_incident_angle) ** 2)
    )

    hi = np.cross(k, _normal_vector, axis=1)
    hi = hi / np.linalg.norm(hi, axis=1)[:, np.newaxis]
    vi = np.cross(hi, k, axis=1)
    E = _center_point
    F = (
        -np.einsum("...i,...i", E, hi)
        * np.einsum("...i,...i", _normal_vector, k)
        * (1 - Rh)
        * hi
        + np.einsum("...i,...i", E, vi)
        * np.cross(_normal_vector, hi, axis=1)
        * (1 + Rv)
        + np.einsum("...i,...i", E, hi)
        * np.cross(-k, np.cross(_normal_vector, hi, axis=1), axis=1)
        * (1 + Rh)
        + np.einsum("...i,...i", E, vi)
        * np.einsum("...i,...i", _normal_vector, k)
        * np.cross(-k, hi, axis=1)
        * (1 - Rv)
    )
    I = np.exp(-2j * k * _distance.T)
    Es = (1j * k * F).T / (4 * np.pi * _distance.T) ** 2 * I * _area
    return Es


def cal_delay_index(_delay):
    # delay_axis是一个以delay_interval为间隔的等差数列，长度为3600，第1800个元素为0
    delay_axis = np.linspace(-delay_interval * 1799, delay_interval * 1800, 3600)
    delay_axis_min = delay_axis.min()
    delay = _delay.reshape(-1)
    # delay-时延的最小值，再除以时延间隔，得到时延的索引
    delay_index = (delay - delay_axis_min) / delay_interval
    # 取整
    delay_index = np.int32(np.round(delay_index))
    delay_index = delay_index.reshape((_delay.shape[0], _delay.shape[1]))
    return delay_index


def cal_rgram(_delay, _point_count, _delay_index, _sgm_hh, _sgm_vv, _area):
    rgram_hh = np.zeros((3600, _point_count))
    rgram_vv = np.zeros((3600, _point_count))
    # delay[i]存储了sgm_hh[i]中的每个元素应该加在img[:,i]的哪个位置
    for i in range(_delay.shape[0]):
        # 使用delay[i]作为索引，将sgm_hh[i]的值加到img的相应位置
        np.add.at(rgram_hh[:, i], _delay_index[i], _sgm_hh[i])
        np.add.at(rgram_vv[:, i], _delay_index[i], _sgm_hh[i])
    return rgram_hh, rgram_vv


def update_result_dict(result_path):
    # 获取save_path下的所有文件名
    file_list = os.listdir(result_path)
    # 去除result.json
    file_list.remove("result.json")
    # 去除文件名中的后缀
    file_list = [file.split(".")[0] for file in file_list]
    # 读取result.json
    with open(result_path + "/result.json", "r") as f:
        result_dict = json.load(f)
    # 将file_list中的文件名对应的result_dict的value改为True，其他为False
    for key in result_dict.keys():
        if key in file_list:
            result_dict[key] = True
        else:
            result_dict[key] = False
    # 保存到json
    with open(result_path + "/result.json", "w") as f:
        json.dump(result_dict, f)
    return result_dict


def sim_main(_PRODUCT_ID):
    # 创建文件夹
    result_path = str(BASE_DIR / "data" / "result" / _PRODUCT_ID)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # 创建一个字典，用来存储已有的数据，先读取json,如果没有，就创建一个空字典
    if os.path.exists(result_path + "/result.json"):
        result_dict = update_result_dict(result_path)
    else:
        # 根据result_list创建一个字典，value都为false
        result_dict = {key: False for key in result_list}
        # 保存到json
        with open(result_path + "/result.json", "w") as f:
            json.dump(result_dict, f)
        result_dict = update_result_dict(result_path)

    # 获取sharad数据
    if result_dict["geom_data"] and result_dict["rgram_data"]:
        geom_data = np.load(result_path + "/geom_data.npy", allow_pickle=True)
        rgram_data = np.load(result_path + "/rgram_data.npy", allow_pickle=True)
    else:
        geom_data, rgram_data = get_sharad_rdr_data(_PRODUCT_ID)
        np.save(result_path + "/geom_data.npy", geom_data, allow_pickle=True)
        np.save(result_path + "/rgram_data.npy", rgram_data, allow_pickle=True)
        result_dict = update_result_dict(result_path)
    print("获取sharad数据完成")

    # 获取仿真区域的经纬度
    # 返回参数 sim_area_lonlat, sat_lon_lat, point_count, sat_radius
    if (
        result_dict["sim_area_lonlat"]
        and result_dict["sat_lon_lat"]
        and result_dict["point_count"]
        and result_dict["sat_radius"]
    ):
        sim_area_lonlat = np.load(
            result_path + "/sim_area_lonlat.npy", allow_pickle=True
        )
        sat_lon_lat = np.load(result_path + "/sat_lon_lat.npy", allow_pickle=True)
        point_count = np.load(result_path + "/point_count.npy", allow_pickle=True)
        sat_radius = np.load(result_path + "/sat_radius.npy", allow_pickle=True)
    else:
        sim_area_lonlat, sat_lon_lat, point_count, sat_radius = get_sim_area_lonlat(
            geom_data
        )
        np.save(
            result_path + "/sim_area_lonlat.npy", sim_area_lonlat, allow_pickle=True
        )
        np.save(result_path + "/sat_lon_lat.npy", sat_lon_lat, allow_pickle=True)
        np.save(result_path + "/point_count.npy", point_count, allow_pickle=True)
        np.save(result_path + "/sat_radius.npy", sat_radius, allow_pickle=True)
        result_dict = update_result_dict(result_path)
    print("获取仿真区域的经纬度完成")

    # 获取仿真区域的笛卡尔坐标
    if result_dict["sim_area_cartesian"]:
        sim_area_cartesian = np.load(
            result_path + "/sim_area_cartesian.npy", allow_pickle=True
        )
    else:
        sim_area_cartesian = get_sim_area_cartesian(sim_area_lonlat, point_count)
        np.save(
            result_path + "/sim_area_cartesian.npy",
            sim_area_cartesian,
            allow_pickle=True,
        )
        result_dict = update_result_dict(result_path)
    print("获取仿真区域的笛卡尔坐标完成")

    # 获取每个三角形的三个顶点的索引
    if result_dict["facet_index"]:
        facet_index = np.load(result_path + "/facet_index.npy", allow_pickle=True)
    else:
        facet_index = cal_facet_idx()
        np.save(result_path + "/facet_index.npy", facet_index, allow_pickle=True)
        result_dict = update_result_dict(result_path)
    print("获取每个三角形的三个顶点的索引完成")

    # 获取每个三角形的中心点
    if result_dict["center_point"]:
        center_point = np.load(result_path + "/center_point.npy", allow_pickle=True)
    else:
        center_point = cal_center_point(sim_area_cartesian, facet_index, point_count)
        np.save(result_path + "/center_point.npy", center_point, allow_pickle=True)
        result_dict = update_result_dict(result_path)
    print("获取每个三角形的中心点完成")

    # 获取每个三角形的法向量和面积
    if result_dict["normal_vector"] and result_dict["area"]:
        normal_vector = np.load(result_path + "/normal_vector.npy", allow_pickle=True)
        area = np.load(result_path + "/area.npy", allow_pickle=True)
    else:
        normal_vector, area = cal_normal_vector_and_area(
            geom_data, sim_area_cartesian, facet_index, sat_lon_lat
        )
        np.save(result_path + "/normal_vector.npy", normal_vector, allow_pickle=True)
        np.save(result_path + "/area.npy", area, allow_pickle=True)
        result_dict = update_result_dict(result_path)
    print("获取每个三角形的法向量和面积完成")

    # 获取每个三角形的属性
    if result_dict["corL"] and result_dict["rmsH"] and result_dict["permittivity"]:
        corL = np.load(result_path + "/corL.npy", allow_pickle=True)
        rmsH = np.load(result_path + "/rmsH.npy", allow_pickle=True)
        permittivity = np.load(result_path + "/permittivity.npy", allow_pickle=True)
    else:
        corL, rmsH, permittivity = get_attributes(center_point)
        np.save(result_path + "/corL.npy", corL, allow_pickle=True)
        np.save(result_path + "/rmsH.npy", rmsH, allow_pickle=True)
        np.save(result_path + "/permittivity.npy", permittivity, allow_pickle=True)
        result_dict = update_result_dict(result_path)
    print("获取每个三角形的属性完成")

    # 获取入射角和时延
    if (
        result_dict["incident_angle"]
        and result_dict["delay"]
        and result_dict["distance"]
    ):
        incident_angle = np.load(result_path + "/incident_angle.npy", allow_pickle=True)
        delay = np.load(result_path + "/delay.npy", allow_pickle=True)
        distance = np.load(result_path + "/distance.npy", allow_pickle=True)
    else:
        incident_angle, delay, distance = get_incidence_angle_and_delay(
            sat_lon_lat, center_point, normal_vector, sat_radius
        )
        np.save(result_path + "/incident_angle.npy", incident_angle, allow_pickle=True)
        np.save(result_path + "/delay.npy", delay, allow_pickle=True)
        np.save(result_path + "/distance.npy", distance, allow_pickle=True)
        result_dict = update_result_dict(result_path)
    print("获取入射角和时延完成")

    # 计算sgm
    if result_dict["sgm_hh"] and result_dict["sgm_vv"]:
        sgm_hh = np.load(result_path + "/sgm_hh.npy", allow_pickle=True)
        sgm_vv = np.load(result_path + "/sgm_vv.npy", allow_pickle=True)
    else:
        sgm_hh, sgm_vv = cal_sgm(
            permittivity, corL, rmsH, incident_angle, center_point, distance
        )
        np.save(result_path + "/sgm_hh.npy", sgm_hh, allow_pickle=True)
        np.save(result_path + "/sgm_vv.npy", sgm_vv, allow_pickle=True)
        result_dict = update_result_dict(result_path)
    print("计算sgm完成")

    # 计算时延索引
    if result_dict["delay_index"]:
        delay_index = np.load(result_path + "/delay_index.npy", allow_pickle=True)
    else:
        delay_index = cal_delay_index(delay)
        np.save(result_path + "/delay_index.npy", delay_index, allow_pickle=True)
        result_dict = update_result_dict(result_path)
    print("计算时延索引完成")

    # 计算rgram
    if result_dict["rgram_hh"] and result_dict["rgram_vv"]:
        rgram_hh = np.load(result_path + "/rgram_hh.npy", allow_pickle=True)
        rgram_vv = np.load(result_path + "/rgram_vv.npy", allow_pickle=True)
    else:
        rgram_hh, rgram_vv = cal_rgram(
            delay, point_count, delay_index, sgm_hh, sgm_vv, area
        )
        np.save(result_path + "/rgram_hh.npy", rgram_hh, allow_pickle=True)
        np.save(result_path + "/rgram_vv.npy", rgram_vv, allow_pickle=True)
        result_dict = update_result_dict(result_path)
    print("计算rgram完成")


if __name__ == "__main__":
    # # 获取sharad数据
    # geom_data, rgram_data = get_sharad_rdr_data("S_00170101")
    # sim_area_lonlat, sat_lon_lat, point_count = get_sim_area_lonlat(geom_data)
    # sim_area_cartesian = get_sim_area_cartesian(sim_area_lonlat, point_count)
    # facet_index = cal_facet_idx()
    # center_point = cal_center_point(sim_area_cartesian, facet_index, point_count)
    # normal_vector, area = cal_normal_vector_and_area(
    #     geom_data, sim_area_cartesian, facet_index, sat_lon_lat
    # )
    # corL, rmsH, permittivity = get_attributes(sim_area_lonlat)
    # incident_angle, delay = get_incidence_angle_and_delay(
    #     sat_lon_lat, center_point, normal_vector
    # )
    # sgm_hh, sgm_vv = cal_sgm(permittivity, corL, rmsH, incident_angle, center_point)
    # delay_index = cal_delay_index(delay)
    # rgram_hh, rgram_vv = cal_rgram(
    #     delay, point_count, delay_index, sgm_hh, sgm_vv, area
    # )
    sim_main("S_00170101")
    # 播放音效提示
    import winsound

    winsound.Beep(600, 1000)
    print("")

    # S_00170101
    # S_01224201
    # S_03280801
    # S_02622401
    # S_03289201
    # S_07639501
    # S_00183004 不好
    # S_01294501
    # S_00519201
    # S_01389101

    # S_01926401
    # S_03554901
    # S_03821902
    # S_04365901
    # S_04725901
    # S_05464301
    # S_06337301
