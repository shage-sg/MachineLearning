# encoding=utf-8

# Author : 啵啵

# File : Log_inf.py 

# Software: PyCharm

import logging
from datetime import date

class Log_inf():

    def __init__(self):
        self.logger = None
        self.fileHandler = None

    def __log_init(self):
        # 记录器
        self.logger = logging.getLogger()
        time = date.today()
        # 设置级别
        self.logger.setLevel(logging.INFO)
        # 没有给handler指定日志级别 默认使用logger的级别

        # 写入文件
        self.fileHandler = logging.FileHandler(filename=f"../log/{time}_VGG19.log", mode='a')
        # formatter格式
        formatter = logging.Formatter(fmt="%(asctime)s-%(levelname)s-%(filename)s-[line:%(levelno)s]-%(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        # 给处理器设置格式
        self.fileHandler.setFormatter(formatter)
        # 记录器设置处理器
        self.logger.addHandler(self.fileHandler)



    def log(self,inf):
        self.__log_init()
        self.logger.info(inf)
        self.logger.removeHandler(self.fileHandler)
        return self.logger


