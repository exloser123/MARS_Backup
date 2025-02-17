from django.db import models


# # Create your models here.
class HiriseDtmData(models.Model):
    # 设置索引为递增的主键
    id = models.AutoField(primary_key=True, verbose_name="id")
    FILE_NAME_SPECIFICATION = models.CharField(max_length=150, verbose_name="文件相对路径")
    PRODUCT_ID = models.CharField(max_length=50, verbose_name="产品ID")
    RATIONALE_DESC = models.CharField(max_length=150, verbose_name="产品描述")
    IMAGE_LINES = models.IntegerField(verbose_name="图像行数")
    LINE_SAMPLES = models.IntegerField(verbose_name="图像列数")
    NORTH_AZIMUTH = models.FloatField(verbose_name="北方方位角")
    MINIMUM_LATITUDE = models.FloatField(verbose_name="最小纬度")
    MAXIMUM_LATITUDE = models.FloatField(verbose_name="最大纬度")
    MINIMUM_LONGITUDE = models.FloatField(verbose_name="最小经度")
    MAXIMUM_LONGITUDE = models.FloatField(verbose_name="最大经度")
    MAP_SCALE = models.FloatField(verbose_name="像素间隔")
    MAP_RESOLUTION = models.FloatField(verbose_name="分辨率")
    MAP_PROJECTION_TYPE = models.CharField(max_length=50, verbose_name="投影类型")
    PROJECTION_CENTER_LATITUDE = models.FloatField(verbose_name="投影中心纬度")
    PROJECTION_CENTER_LONGITUDE = models.FloatField(verbose_name="投影中心经度")
    LINE_PROJECTION_OFFSET = models.FloatField(verbose_name="行投影偏移")
    SAMPLE_PROJECTION_OFFSET = models.FloatField(verbose_name="列投影偏移")
    CORNER1_LATITUDE = models.FloatField(verbose_name="角点1纬度")
    CORNER1_LONGITUDE = models.FloatField(verbose_name="角点1经度")
    CORNER2_LATITUDE = models.FloatField(verbose_name="角点2纬度")
    CORNER2_LONGITUDE = models.FloatField(verbose_name="角点2经度")
    CORNER3_LATITUDE = models.FloatField(verbose_name="角点3纬度")
    CORNER3_LONGITUDE = models.FloatField(verbose_name="角点3经度")
    CORNER4_LATITUDE = models.FloatField(verbose_name="角点4纬度")
    CORNER4_LONGITUDE = models.FloatField(verbose_name="角点4经度")
    GEOLOGICAL_TYPE = models.CharField(
        max_length=10, verbose_name="地质类型", default="unknown"
    )

    class Meta:
        verbose_name = "HiRISE DTM数据"
        verbose_name_plural = verbose_name
        db_table = "hirise_dtm_data"
        ordering = ["id"]


class SIM3292_Global_Geology(models.Model):
    """
    ['unit', 'unit_name', 'description', 'ADDITIONAL CHARACTERISTICS',
       'interpretation', 'RGB', 'main_unit', 'main_unit_desc',
       'corresponding_value']
    """

    id = models.AutoField(primary_key=True, verbose_name="id")
    unit = models.CharField(max_length=10, verbose_name="分类缩写")
    unit_name = models.CharField(max_length=100, verbose_name="分类名称")
    description = models.CharField(max_length=1000, verbose_name="分类描述")
    additional_characteristics = models.CharField(max_length=1000, verbose_name="附加特征")
    interpretation = models.CharField(max_length=1000, verbose_name="解释")
    RGB = models.CharField(max_length=50, verbose_name="颜色")
    main_unit = models.CharField(max_length=100, verbose_name="主分类")
    main_unit_desc = models.CharField(max_length=1000, verbose_name="主分类描述")
    corresponding_value = models.FloatField(verbose_name="对应值")

    class Meta:
        verbose_name = "SIM3292全球地质数据"
        verbose_name_plural = verbose_name
        db_table = "sim3292_global_geology"
        ordering = ["id"]


class SharadRDRTable(models.Model):
    id = models.AutoField(primary_key=True, verbose_name="id")
    """
    ['VOLUME_ID',
     'RGRAM_FILE_SPECIFICATION_NAME',
     'GEOM_FILE_SPECIFICATION_NAME',
     'PRODUCT_ID',
     'PRODUCT_CREATION_TIME',
     'ORBIT_NUMBER',
     'START_TIME',
     'STOP_TIME',
     'MRO:START_SUB_SPACECRAFT_LATITUDE',
     'MRO:STOP_SUB_SPACECRAFT_LATITUDE',
     'MRO:START_SUB_SPACECRAFT_LONGITUDE',
     'MRO:STOP_SUB_SPACECRAFT_LONGITUDE']
    """
    VOLUME_ID = models.CharField(max_length=20, verbose_name="卷ID")
    RGRAM_FILE_SPECIFICATION_NAME = models.CharField(
        max_length=100, verbose_name="RGRAM文件相对路径"
    )
    GEOM_FILE_SPECIFICATION_NAME = models.CharField(
        max_length=100, verbose_name="GEOM文件相对路径"
    )
    PRODUCT_ID = models.CharField(max_length=50, verbose_name="产品ID")
    # PRODUCT_CREATION_TIME形如：2021-05-19 16:15:43的datetime.datetime类型
    PRODUCT_CREATION_TIME = models.DateTimeField(verbose_name="产品创建时间")
    ORBIT_NUMBER = models.IntegerField(verbose_name="轨道号")
    # START_TIME形如：2021-05-19 16:15:43的datetime.datetime类型
    START_TIME = models.DateTimeField(verbose_name="开始时间")
    # STOP_TIME形如：2021-05-19 16:15:43的datetime.datetime类型
    STOP_TIME = models.DateTimeField(verbose_name="结束时间")
    # MRO:START_SUB_SPACECRAFT_LATITUDE形如：-85.0的float类型
    MRO_START_SUB_SPACECRAFT_LATITUDE = models.FloatField(verbose_name="MRO：子航天器起始纬度")
    # MRO:STOP_SUB_SPACECRAFT_LATITUDE形如：-85.0的float类型
    MRO_STOP_SUB_SPACECRAFT_LATITUDE = models.FloatField(verbose_name="MRO：子航天器结束纬度")
    # MRO:START_SUB_SPACECRAFT_LONGITUDE形如：-85.0的float类型
    MRO_START_SUB_SPACECRAFT_LONGITUDE = models.FloatField(verbose_name="MRO：子航天器起始经度")
    # MRO:STOP_SUB_SPACECRAFT_LONGITUDE形如：-85.0的float类型
    MRO_STOP_SUB_SPACECRAFT_LONGITUDE = models.FloatField(verbose_name="MRO：子航天器结束经度")

    class Meta:
        verbose_name = "SHARAD RDR数据"
        verbose_name_plural = verbose_name
        db_table = "sharad_rdr_table"
        ordering = ["id"]
