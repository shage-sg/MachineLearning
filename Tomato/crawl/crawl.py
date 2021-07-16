# encoding=utf-8

# Author : 啵啵

# File : crawl.py

# Software: PyCharm

import csv
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait

def crawl_data():
    url = "https://mp.weixin.qq.com/s?__biz=MzI0MjA0NTQyNg==&mid=209268321&idx=1&sn=5a4d6a1d6d91908031e51beae2e3abd2&scene=21#wechat_redirect"
    driver = webdriver.Chrome()
    driver.get(url=url)
    wait = WebDriverWait(driver,25)
    # 窗口最大化
    # driver.maximize_window()
    # 疾病名称，危害特征，防治方法
    Disease_Names = []
    Hazard_Characterization = []
    Control_Method = []
    for i in range(3,124,5):
        DN = driver.find_element_by_xpath(f"//*[@id='js_content']/section[{i}]/span/section")
        Disease_Names.append(DN.text)
    for i in range(6,127,5):
        HC = driver.find_element_by_xpath(f"//*[@id='js_content']/section[{i}]")
        Hazard_Characterization.append(HC.text)
    for i in range(7,128,5):
        CM = driver.find_element_by_xpath(f"//*[@id='js_content']/section[{i}]")
        Control_Method.append(CM.text)
    print(Disease_Names)
    for i in zip(Disease_Names,Hazard_Characterization,Control_Method):
        with open(f"./tomato_disease/{i[0]}.txt",'w',encoding='utf-8') as writer:
            writer.write(i[1]+'\n'+i[2])
 
if __name__ == "__main__":
    crawl_data()
