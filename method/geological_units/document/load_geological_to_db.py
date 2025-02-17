import pandas as pd
import django
import os

# 首先初始化django的环境，使得可以在其他py文件中使用django的模型
# 设置Django的设置模块路径。
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mars_sim_web.settings")
# 初始化Django
django.setup()
import sim_site.models as models

# 读取数据
df = pd.read_excel("SIM3292_Global_Geology.xlsx")
# 把空值替换成'None'
df = df.fillna("None")


# 把df的数据存入数据库,遍历df数据的每一行
for index, row in df.iterrows():
    # 如果数据库中已经存在该数据，则跳过
    if models.SIM3292_Global_Geology.objects.filter(unit=row["unit"]).exists():
        continue
    geological_db = models.SIM3292_Global_Geology()
    geological_db.unit = row["unit"]
    geological_db.unit_name = row["unit_name"]
    geological_db.description = row["description"]
    geological_db.additional_characteristics = row["additional_characteristics"]
    geological_db.interpretation = row["interpretation"]
    geological_db.RGB = row["RGB"]
    geological_db.main_unit = row["main_unit"]
    geological_db.main_unit_desc = row["main_unit_desc"]
    geological_db.corresponding_value = row["corresponding_value"]
    geological_db.save()
    abc = 1
