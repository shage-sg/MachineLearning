# encoding=utf-8

# Time : 2021/3/29 17:23 

# Author : 啵啵

# File : ETL_data.py 

# Software: PyCharm



# # 处理训练集中文的格式
# def solve_train_chn():
#     import jieba
#     with open("E:\PycharmProjects\MachineLearning\ML\data\english-chinese.txt","r",encoding="utf-8") as r:
#
#         while True:
#         # for i in  range(100):
#             s = r.readline()
#             if len(s) < 5:
#                 break
#             se = s.split("\t")[0]
#             sc = ' '.join(list(jieba.cut(s.split("\t")[1])))
#
#             with open("E:\PycharmProjects\MachineLearning\ML\data\eng-chn.txt","a",encoding="utf-8") as w:
#                 w.writelines(se+"\t"+sc)

# 处理测试集的中文格式
def solve_test_chn():
    import re
    with open("./data/complex_zh_en.00_筛选的已标注且校对数据(带标签，UTF-8) - 前10000句对.txt",'r',encoding="utf-8") as r_test_chn:
        for i in range(20000):
            test_chn_data = r_test_chn.readline().strip()
            if len(test_chn_data) > 5:
                test_chn_data_list = test_chn_data.split("-end>")
                ss = ''
                for data in test_chn_data_list:
                    if data != '':
                        chn_data = re.findall(r"<.*-start>(.*)<.*?", data)
                        if len(chn_data) == 0:
                            chn_data.append("bad_data")
                        ss = ss+chn_data[0]+"   "
                if "bad_data" in ss:
                    continue
                with open("./data/complex_zh_en.txt", "a",encoding="utf-8") as w:
                    w.write(ss)
                    w.write("\n")
if __name__ == "__main__":
    solve_test_chn()