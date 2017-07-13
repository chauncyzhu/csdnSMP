# coding=gbk
"""
    将数据分为五份，大概每份有8百兆
"""
import numpy as np
import utils.data_path as dp

file = open(dp.BlogContentSegPath+"file1.txt", encoding="utf8")
lines = file.readlines()
lines_len = len(lines)
print("lines length----",lines_len)
#将数据分为5份
numbers= 5
#需要写入的文件
file_names = [dp.BlogContentSegPath+'file1_'+str(i)+'.txt' for i in range(numbers)]

def seg_list(ls,n):
    """
    将列表等分成n份
    :param ls: 需要等分的列表
    :param n: 等分的份数
    :return:
    """
    if not isinstance(ls,list) or not isinstance(n,int):
        return []
    ls_len = len(ls)
    if n<=0 or ls_len == 0:
        return []
    if n>ls_len:
        return []
    elif n == ls:
        return [[i] for i in ls]
    else:
        k = int(ls_len/n)
        ls_return = []
        for i in np.arange(0,(n-1)*k,k):
            ls_return.append(ls[int(i):int(i+k)])  #不确定最后一个是否全部包含
        ls_return.append(ls[(n-1)*k:])  #因此需要在这里加上最后一段
        return ls_return


div_list = seg_list(list(range(lines_len)),numbers)

print("div list is:",div_list)

def get_line(lines,list_indexs,file):
    for i in list_indexs:
        if i % 10000 == 0:
            print("now lines index---",i)
        file.write(lines[i])
    if file:
        file.flush()
        file.close()

for i in range(len(div_list)):   #div_list长度和filenames长度一样
    if len(file_names) != len(div_list):
        print("file names length not equal div list length!")
        break
    print("now iter "+str(i)+" list! file name "+file_names[i])
    file = open(file_names[i],'w',encoding="utf8")
    get_line(lines,div_list[i],file)   #已经在里面处理好文件输出和关闭

lines = []
del lines