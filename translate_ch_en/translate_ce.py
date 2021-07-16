# encoding=utf-8

# Time : 2021/3/29 17:23

# Author : 啵啵

# File : ETL_data.py

# Software: PyCharm

from urllib import request, parse
import json


def fanyi(content):
    req_url = \
        'http://fanyi.youdao.com/translate'
    head_data = {}
    head_data['Referer'] = \
        'http://fanyi.youdao.com/'
    head_data['User-Agent'] = \
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.3'
    form_data = {}
    form_data['i'] = content
    form_data['doctype'] = 'json'
    data = parse.urlencode(form_data).encode('utf-8')
    req = request.Request(req_url, data, head_data)
    response = request.urlopen(req)
    html = response.read().decode('utf-8')
    translate_results = json.loads(html)
    translate_results = translate_results['translateResult'][0][0]['tgt']
    return translate_results

def Translate(chn):
     return fanyi(chn)

Translate("你好")



