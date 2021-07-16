# encoding=utf-8

# Time : 2021/3/29 10:26 

# Author : 啵啵

# File : Sentence_Alignment.py 

# Software: PyCharm


class Sentence_Alignment(object):

    def __init__(self, path):

        self.path = path
        self.eng = []
        self.chn = []
        self.eng_chn_dict = {}

    def __call__(self):

        self.__read_data()

    def __read_data(self):

        with open(r"E:\PycharmProjects\MachineLearning\ML\data\complex_zh_en.txt", 'r', encoding="utf-8") as r:
            # 下标索引从零开始
            data = r.readlines()[:]
            for da in data:
                if data.index(da) % 2 == 0:
                    self.chn.append(da.strip())
                else:
                    self.eng.append(da.strip())

        for chn_eng_data in zip(self.chn, self.eng):
            chn_data = chn_eng_data[0]
            eng_data = chn_eng_data[1]
            if len(chn_data.strip().split("   ")) >= 2:
                self.__translate(chn_data, eng_data)
            else:
                boolean = "T"
                self.__write(boolean, chn_data.strip(), eng_data.strip())

    def __translate(self, chn, eng_original):
        # from translate import Translator
        # translator = Translator(from_lang='chinese', to_lang='english')
        from translate_ch_en import translate_ce
        # 中文翻译成英文
        self.chn_list = chn.strip().split("   ")
        self.eng_list = eng_original.strip().split("   ")
        for ch in self.chn_list:
            for en in self.eng_list:
                eng_result = translate_ce.Translate(ch[3:])
                self.__compare(ch, en, eng_result)
        self.__judge(self.chn_list)

    def __compare(self, ch, eng_original, eng_result):
        import difflib
        import Levenshtein
        # 计算莱文斯坦比
        sim = Levenshtein.ratio(eng_original, eng_result)
        self.eng_chn_dict[sim] = [ch, eng_original]
        print(f'Levenshtein.ratio similarity of {[ch, eng_original]} : ', sim)

    def __judge(self, ls: len):
        # 字典按键值倒序排序
        keys = sorted(self.eng_chn_dict, reverse=True)
        judge = "T"
        print(keys[:len(ls)])
        # 按照切分子句数来判断
        for key in keys[:len(ls)]:
            if self.eng_chn_dict[key][0].strip()[:3] == self.eng_chn_dict[key][1].strip()[:3]:
                judge = "T"
            else:
                judge = "F"
                self.chn_list[self.chn_list.index(self.eng_chn_dict[key][0])] = \
                    self.eng_chn_dict[key][1].strip()[:3] + self.eng_chn_dict[key][0].strip()[3:]
        self.__write(judge, "".join(self.chn_list), "".join(self.eng_list))

    def __write(self, boolean, chn, eng):
        with open("./data/complex_zh_en_new.txt", 'a', encoding="utf-8") as w:
            w.write(boolean + " " + chn + "\n")
            w.write(boolean + " " + eng + "\n")


path = r"E:\PycharmProjects\MachineLearning\ML\data\complex_zh_en.txt"
Sentence_Alignment(path)()
