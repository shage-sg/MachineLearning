# encoding=utf-8

# Time : 2022/1/4 17:07 

# Author : 啵啵

# File : test.py 

# Software: PyCharm

from lxml import etree
import requests

if __name__ == "__main__":
    url = "https://pic.netbian.com/4kmeinv/"
    headers = {
        'User-Agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
    }
    page_text = requests.get(url=url, headers=headers).text

    # 解析数据：src属性 alt属性
    tree = etree.HTML(page_text)
    li_list = tree.xpath("//div[@class='slist']/ul/li")
    for li in li_list:
        img_src = 'https://pic.netbian.com'+li.xpath("./a/img/@src")[0]
        # 通用处理中文乱码的解决方案
        img_name = li.xpath("./a/img/@alt")[0].encode('iso-8859-1').decode('gbk')+".jpg"

        print(img_name,img_src)