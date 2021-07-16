# encoding=utf-8

# Author : 啵啵

# File : File_monitor.py 

# Software: PyCharm

import os, threading, time, numpy as np
import MySQLdb

from watchdog.observers import Observer
from watchdog.events import *
from watchdog.utils.dirsnapshot import DirectorySnapshot, DirectorySnapshotDiff

from Tomato.function_models.Deep_pipline import Deep_pipline
from Tomato.function_models.Log_inf import Log_inf
from Tomato.function_models.VGG19_model import VGG19_model


class FileEventHandler(FileSystemEventHandler):
    def __init__(self, aim_path):
        FileSystemEventHandler.__init__(self)
        self.aim_path = aim_path
        self.timer = None
        self.snapshot = DirectorySnapshot(self.aim_path)
        self.vgg19 = VGG19_model().VGG19()
        self.conn = None
        self.cursor = None
        Log_inf().log('Load Model Success!')

    def on_any_event(self, event):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(0.2, self.checkSnapshot, )
        self.timer.start()

    def connect_mysql(self):
        self.conn = MySQLdb.connect(
            # 主机IP地址
            host='10.102.25.16',
            # 用户名
            user='root',
            # 数据库名
            db='tomato',
            # 密码
            password='root',
            # 端口
            port=3306,
        )
        self.cursor = self.conn.cursor()
        Log_inf().log('Connect MySQL Success!')


    def checkSnapshot(self, ):
        snapshot = DirectorySnapshot(self.aim_path)
        diff = DirectorySnapshotDiff(self.snapshot, snapshot)
        self.snapshot = snapshot
        pre_dict = {
            0: "Bacterial_spot",
            1: "Early_blight",
            2: "healthy",
            3: "Late_blight",
            4: "Leaf_Mold"
        }
        if len(diff.files_created) != 0:
            for file in diff.files_created:
                Log_inf().log(f"file was created:{file}")
                predict_X = Deep_pipline().process([file])
                pre = self.vgg19.predict(predict_X)
                predict = np.argmax(pre[0])
                self.connect_mysql()
                sql = 'UPDATE photo SET category="%s"'%(pre_dict.get(predict))+ \
                ', knowmapurl="%s.png"'%(pre_dict.get(predict)) + \
                ' WHERE photourl="%s"' %(file.split("\\")[1])
                self.cursor.execute(sql)
                self.cursor.close()
                self.conn.commit()
                self.conn.close()
                Log_inf().log('Put Data To MySQL Success!')


        if len(diff.files_deleted) != 0:
            for file in diff.files_deleted:
                Log_inf().log(f"file was deleted:{file}")


class DirMonitor(object):
    """文件夹监视类"""

    def __init__(self, aim_path):
        """构造函数"""

        self.aim_path = aim_path
        self.observer = Observer()

    def start(self):
        """启动"""

        event_handler = FileEventHandler(self.aim_path)
        self.observer.schedule(event_handler, self.aim_path, recursive=True)
        self.observer.start()


    def stop(self):
        """停止"""
        self.observer.stop()


if __name__ == "__main__":
    monitor = DirMonitor(r"../ftp_pictures")
    monitor.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        monitor.stop()
