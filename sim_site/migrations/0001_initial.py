# Generated by Django 4.2.6 on 2023-10-26 15:36

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="HiriseDtmData",
            fields=[
                (
                    "id",
                    models.AutoField(
                        primary_key=True, serialize=False, verbose_name="id"
                    ),
                ),
                (
                    "FILE_NAME_SPECIFICATION",
                    models.CharField(max_length=150, verbose_name="文件相对路径"),
                ),
                ("PRODUCT_ID", models.CharField(max_length=50, verbose_name="产品ID")),
                (
                    "RATIONALE_DESC",
                    models.CharField(max_length=150, verbose_name="产品描述"),
                ),
                ("IMAGE_LINES", models.IntegerField(verbose_name="图像行数")),
                ("LINE_SAMPLES", models.IntegerField(verbose_name="图像列数")),
                ("NORTH_AZIMUTH", models.FloatField(verbose_name="北方方位角")),
                ("MINIMUM_LATITUDE", models.FloatField(verbose_name="最小纬度")),
                ("MAXIMUM_LATITUDE", models.FloatField(verbose_name="最大纬度")),
                ("MINIMUM_LONGITUDE", models.FloatField(verbose_name="最小经度")),
                ("MAXIMUM_LONGITUDE", models.FloatField(verbose_name="最大经度")),
                ("MAP_SCALE", models.FloatField(verbose_name="像素间隔")),
                ("MAP_RESOLUTION", models.FloatField(verbose_name="分辨率")),
                (
                    "MAP_PROJECTION_TYPE",
                    models.CharField(max_length=50, verbose_name="投影类型"),
                ),
                (
                    "PROJECTION_CENTER_LATITUDE",
                    models.FloatField(verbose_name="投影中心纬度"),
                ),
                (
                    "PROJECTION_CENTER_LONGITUDE",
                    models.FloatField(verbose_name="投影中心经度"),
                ),
                ("LINE_PROJECTION_OFFSET", models.FloatField(verbose_name="行投影偏移")),
                ("SAMPLE_PROJECTION_OFFSET", models.FloatField(verbose_name="列投影偏移")),
                ("CORNER1_LATITUDE", models.FloatField(verbose_name="角点1纬度")),
                ("CORNER1_LONGITUDE", models.FloatField(verbose_name="角点1经度")),
                ("CORNER2_LATITUDE", models.FloatField(verbose_name="角点2纬度")),
                ("CORNER2_LONGITUDE", models.FloatField(verbose_name="角点2经度")),
                ("CORNER3_LATITUDE", models.FloatField(verbose_name="角点3纬度")),
                ("CORNER3_LONGITUDE", models.FloatField(verbose_name="角点3经度")),
                ("CORNER4_LATITUDE", models.FloatField(verbose_name="角点4纬度")),
                ("CORNER4_LONGITUDE", models.FloatField(verbose_name="角点4经度")),
                (
                    "GEOLOGICAL_TYPE",
                    models.CharField(
                        default="unknown", max_length=10, verbose_name="地质类型"
                    ),
                ),
                (
                    "CORRELATION_LENGTH",
                    models.FloatField(default=0, verbose_name="相关长度"),
                ),
                (
                    "ROOT_MEAN_SQUARE_HEIGHT",
                    models.FloatField(default=0, verbose_name="均方根高度"),
                ),
            ],
            options={
                "verbose_name": "HiRISE DTM数据",
                "verbose_name_plural": "HiRISE DTM数据",
                "db_table": "hirise_dtm_data",
                "ordering": ["id"],
            },
        ),
    ]
