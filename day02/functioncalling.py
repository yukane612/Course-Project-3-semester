from zhipuai import ZhipuAI
import json

import requests  #数据采集（训练ai）

client = ZhipuAI()
prompt = "今天湘潭的天气怎么样？"
location = input("请输入要查询的天气")
