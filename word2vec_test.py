# coding=gbk
"""
    对Word2vec进行测试
"""
import gensim
import utils.data_path as dp
import jieba

numbers = 5
# 需要写入的文件
file_names = [dp.BlogContentSegPath + 'file1_' + str(i) + '.txt' for i in range(numbers)]
target_model_names = [dp.CSDNMODELPath + "csdn_model_1_" + str(i) + ".m" for i in range(numbers)]

"""
#首先对某个文件数据直接训练
file = open(file_names[0],encoding="utf8")
line = file.readline()
count = 0
result = []
while line:
    if count % 10000 == 0:
        print("已经对"+str(count)+"进行了分词")
    line = line.strip().split("\001")
    result.append(list(jieba.cut(line[2])))
    line = file.readline()
    count += 1
file.close()

model = gensim.models.Word2Vec(result, min_count=1)
model.save(target_model_names[0])
"""

#加载原先的模型，更新voca
model = gensim.models.Word2Vec.load(dp.CSDNMODELPath + "csdn_model_0.m")
file = open(file_names[0], encoding="utf8")
line = file.readline()
count = 0
result = []
while line:
    if count % 10000 == 0:
        print("已经对" + str(count) + "进行了分词")
    line = line.strip().split("\001")
    result.append(list(jieba.cut(line[2])))
    line = file.readline()
    count += 1
file.close()

model.build_vocab(result)   #更新vocabulary
model.train(result)   #更新参数
model.save(target_model_names[1])   #注意，这里是存放在model_1_1中的，上面是加载file_1_0
