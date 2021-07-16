# encoding=utf-8

# Time : 2021/3/30 18:26 

# Author : 啵啵

# File : tr_test.py 

# Software: PyCharm

import pydeepl

def language_setting():

    translation = pydeepl.translate("你好", "EN")
    return translation

language_setting()
