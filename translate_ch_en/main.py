# encoding=utf-8

# Time : 2021/3/30 17:30 

# Author : 啵啵

# File : main.py 

# Software: PyCharm


import requests
from HandleJs import Py4Js


def translate(tk, content):
    if len(content) > 4891:
        print("翻译的长度超过限制！！！")
        return

    param = {'tk': tk, 'q': content}

    result = requests.get("""http://translate.google.cn/translate_a/single?client=t&sl=en
        &tl=zh-CN&hl=zh-CN&dt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss
        &dt=t&ie=UTF-8&oe=UTF-8&clearbtn=1&otf=1&pc=1&srcrom=0&ssel=0&tsel=0&kc=2""", params=param)

    # 返回的结果为Json，解析为一个嵌套列表
    for text in result.json():
        print(text)


def main():
    js = Py4Js()

    content = """你好"""

    tk = js.getTk(content)
    translate(tk, content)


if __name__ == "__main__":
    main()


