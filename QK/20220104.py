# encoding=utf-8

# Time : 2022/1/4 16:18 

# Author : 啵啵

# File : 20220104.py 

# Software: PyCharm

# 任务一
print("Hello Python")

# 任务二
def person(username):
    # 用户名
    UserName = username
    # 用户等级
    UserLevel = 100
    # 用户经验值
    UserExper = 3000
    # 用户血量
    UserBl = 100
    # 用户魔法值
    UserMv = 100

person = person("XiaoMing")

# 任务三
print("欢迎您来到", input("请您输入籍贯:"))

# 任务四
data = 0
num = 1
while num < 5:
    data += int(input(f"请您输入第{num}个数:"))
    num += 1
print(f"四个数的和等于{data}")

# 任务五

def score_sum(name,subject):
    score = 0
    while subject:
        score += int(input(f"请您输入{subject[-1]}的分数："))
        subject.pop()
    print(f"{name}的综合成绩为{score}")
score_sum("小米", ["语文","数学", "英语"])

# 任务六
def Trapezoidal_area(bottom,top,high):
    area = (bottom+top)*high / 2
    print(f"梯形的面积为{area}")

Trapezoidal_area(1,2,3)

# 任务七
days = int(input("请您输入天数："))
print(f"{int(days/7)}周{days%7}天")

# 任务八
num1 = int(input("请您输入一个数:"))
num2 = int(input("请您输入另外一个数:"))
print(f"输入的数值为{num1}和{num2}")
num1, num2 = num2, num1
print(f"交换后的数值为{num1}和{num2}")
