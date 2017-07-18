# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:14:40 2017

@author: chauncy
"""
"""
现在已经训练好了词向量，应该如何用呢？
"""
import itertools
import pandas as pd
import utils.data_path as dp

"""
下面使用规则来找出用户的标签
1.对于测试集中的每个用户，去找数据集中所有关联的博文，然后比较博文重合率大不大，
"""
pd_train = pd.read_csv(dp.TrainCsv, index_col=0, encoding="utf8")
pd_train["blog_uid"] = pd_train["blog_uid"].apply(eval)
train_blog_uids = list(itertools.chain(*list(pd_train["blog_uid"])))
print("train_blog_uids length---", len(train_blog_uids))
train_unique_blog_uids = list(set(train_blog_uids))
print("train_blog_uids length---", len(train_unique_blog_uids))

# voca = {word:index for index,word in enumerate(train_unique_blog_uids)}

pd_dev = pd.read_csv(dp.DevCsv, index_col=0, encoding="utf8")
pd_dev["blog_uid"] = pd_dev["blog_uid"].apply(eval)
dev_blog_uids = list(itertools.chain(*list(pd_dev["blog_uid"])))
print("dev_blog_uids length---", len(dev_blog_uids))
dev_unique_blog_uids = list(set(dev_blog_uids))
print("dev_blog_uids length---", len(dev_unique_blog_uids))

print("in train but not in dev blog length---", len(set(train_unique_blog_uids) - set(dev_unique_blog_uids)))
print("in train also in dev blog length---", len(set(train_unique_blog_uids) & set(dev_unique_blog_uids)))

# 获取所有需要的blog_id,并写入文件
with open("test.txt", 'w') as f:
    blog_id_list = []
    union = list(set(train_unique_blog_uids) | set(dev_unique_blog_uids))
    for i in union:
        blog_id_list.append(i)
        f.write(i + "\n")

# 获取blog id对应的博文
blog_source = open(dp.BlogContentTxt, encoding="utf8")
line = blog_source.readline()
total_blog_ids = []
while line:
    line = line.strip()
    blog_id_temp = line[:line.index("\001")]
    print("blog id:", blog_id_temp)
    total_blog_ids.append(blog_id_temp)
    line = blog_source.readline()

blog_id_index = []
for i in blog_id_list:
    print("blog id:", i)
    print("blog id:", total_blog_ids.index(i))
    blog_id_index.append(total_blog_ids.index(i))

blog_id_index_sort = sorted(blog_id_index)

target_file = open("/home/cike/PycharmProjects/pythondata/csdnSMP/Train_DATA/some_blogcontent.txt", 'w')
# 获取blog id对应的博文
blog_source = open(dp.BlogContentTxt, encoding="utf8")
line = blog_source.readline()
count_temp = 0
count_stop = 0
while line:
    if count_temp < len(blog_id_index_sort) and blog_id_index_sort[count_temp] == count_stop:
        print("count_temp:", count_temp)
        line = line.strip()
        target_file.write(line + "\n")
        count_temp += 1
    line = blog_source.readline()
    count_stop += 1

if target_file:
    target_file.flush()
    target_file.close()

"""
重复的blog有8193593个
unique的blog有23507个
"""
"""
train blog_uids length---8193593
train unique blog_uids length---23507
test blog_uids length---8948850
test unique blog_uids length---26703
in train also in dev blog length--- 4139
in train but not in dev blog length--- 19368
"""

"""
现在找出每个用户所关联的博客对应的用户
"""
import pandas as pd

pd_post = pd.read_csv(dp.PostTxt, encoding="utf8", sep="\001")
pd_post.columns = ['uid', 'blog_id', 'time']
temp = set(train_unique_blog_uids) - set(list(pd_post['blog_id']))  # 0

pd_brows = pd.read_csv(dp.BrowseTxt, encoding="utf8", sep="\001")
pd_brows.columns = ['uid', 'blog_id', 'time']

pd_comment = pd.read_csv(dp.CommentTxt, encoding="utf8", sep="\001")
pd_comment.columns = ['uid', 'blog_id', 'time']

pd_voteup = pd.read_csv(dp.VoteupTxt, encoding="utf8", sep="\001")
pd_voteup.columns = ['uid', 'blog_id', 'time']

pd_votedown = pd.read_csv(dp.VotedownTxt, encoding="utf8", sep="\001")
pd_votedown.columns = ['uid', 'blog_id', 'time']

pd_favorite = pd.read_csv(dp.FavoriteTxt, encoding="utf8", sep="\001")
pd_favorite.columns = ['uid', 'blog_id', 'time']

"""
read some blog content(in train also in test data), 
which will be cut by jieba
write in the some_blogcontent csv
"""
pd_blog_content = pd.read_csv("/home/cike/PycharmProjects/pythondata/csdnSMP/Train_DATA/some_blogcontent.txt",
                              encoding="utf8", sep="\001")
pd_blog_content.columns = ['blog_id', 'title', 'content']
pd_blog_content['blog_jieba'] = [None] * len(pd_blog_content)

import jieba

for index, row in pd_blog_content.iterrows():
    print("index", index)
    row['blog_jieba'] = list(jieba.cut(row['title'] + row['content']))

pd_blog_content.to_csv('/home/cike/PycharmProjects/pythondata/csdnSMP/Train_DATA/some_blogcontent.csv', encoding="utf8")

# read some_blogcontent csv file by pandas
pd_blog_content = pd.read_csv('/home/cike/PycharmProjects/pythondata/csdnSMP/Train_DATA/some_blogcontent.csv',
                              index_col=0, encoding="utf8")
pd_blog_content['content'] = [None] * len(pd_blog_content)
pd_blog_content["blog_jieba"] = pd_blog_content["blog_jieba"].apply(eval)

"""
get some_blogcontent voca, beacause it's too big to load in the same time
获取词典以及对应的词向量, writer in the csv file
"""
import itertools

pd_voca = pd.DataFrame(list(set(list(itertools.chain(*list(pd_blog_content["blog_jieba"]))))))
pd_voca.columns = ['word']

from imp import reload
import gensim
import utils.data_path as dp

reload(dp)
model = gensim.models.Word2Vec.load(dp.CSDNMODEL)

vector_len = len(model['print'])
pd_voca['vector'] = [[]] * (len(pd_voca))


def f(x):
    if x in model:
        return list(model[x])
    else:
        return [0] * vector_len


pd_voca['vector'] = pd_voca['word'].apply(f)
pd_voca.to_csv('/home/cike/PycharmProjects/pythondata/csdnSMP/Train_DATA/blog_voca.csv', encoding="utf8")

"""
read blog voca from the csv file
transfer vector fron string to vector
"""
import pandas as pd

pd_voca = pd.read_csv("/home/cike/PycharmProjects/pythondata/csdnSMP/Train_DATA/blog_voca.csv", index_col=0,
                      encoding="utf8")
pd_voca['vector'] = pd_voca['vector'].apply(eval)
pd_voca = pd_voca.set_index('word')

# read some_blogcontent csv file by pandas
import numpy as np

pd_blog_content = pd.read_csv('/home/cike/PycharmProjects/pythondata/csdnSMP/Train_DATA/some_blogcontent.csv',
                              index_col=0, encoding="utf8")
pd_blog_content['content'] = [None] * len(pd_blog_content)
pd_blog_content["blog_jieba"] = pd_blog_content["blog_jieba"].apply(eval)

pd_blog_content["some_blog_jieba"] = [[]] * len(pd_blog_content)
# select word which in the pd voca index, because it's too big to calculate in the same time
for index, row in pd_blog_content.iterrows():
    print("index", index)
    # temp = np.array([float(0)]*100)   # vector has 100 feature
    temp = []
    for word in row['blog_jieba']:
        if word in pd_voca.index:
            temp.append(word)

    row["some_blog_jieba"] = temp
    temp = None

# calculate the doc vector
pd_blog_content["blog_jieba_vector"] = [[]] * len(pd_blog_content)
for index, row in pd_blog_content.iterrows():
    print("index", index)
    temp = list(np.sum(list(pd_voca.loc[row["some_blog_jieba"]]['vector']), axis=0))
    # print(temp)
    row["blog_jieba_vector"] = temp

pd_blog_content.to_csv('/home/cike/PycharmProjects/pythondata/csdnSMP/Train_DATA/some_blogcontent_vector.csv',
                       encoding="utf8")

"""
已经计算好了doc的vector
现在将每个vector归一化处理
将训练数据和测试数据转为matrix
"""
pd_train_data = pd.read_csv(dp.TrainCsv, encoding="utf8", index_col=0)
pd_train_data["blog_uid"] = pd_train_data["blog_uid"].apply(eval)

pd_dev_data = pd.read_csv(dp.DevCsv, encoding="utf8", index_col=0)
pd_dev_data["blog_uid"] = pd_dev_data["blog_uid"].apply(eval)

# 加载训练csv数据
import utils.data_path as dp
import pandas as pd

pd_blog_content = pd.read_csv('/home/cike/PycharmProjects/pythondata/csdnSMP/Train_DATA/some_blogcontent_vector.csv',
                              encoding="utf8", index_col=0)
pd_blog_content['blog_jieba_vector'] = pd_blog_content['blog_jieba_vector'].apply(eval)

# normalize
import numpy as np


def f(a):
    a = np.array(a)
    return (a - a.min()) / (a.max() - a.min())


pd_blog_content['blog_jieba_vector'] = pd_blog_content['blog_jieba_vector'].apply(f)

"""
这里不再需要用到分词的结果，只需要找到blog对应的vector
获得voca的embedding matrix
"""


def g(x):
    return list(pd_blog_content['blog_id'][pd_blog_content['blog_id'].isin(x)].index)


pd_train_data["embedding_index"] = pd_train_data["blog_uid"].apply(g)
pd_dev_data["embedding_index"] = pd_dev_data["blog_uid"].apply(g)

from imp import reload

reload(dp)
pd_train_data.to_csv(dp.TrainCsv, encoding="utf8")
pd_dev_data.to_csv(dp.DevCsv, encoding="utf8")

count = 0
for vector in pd_blog_content['blog_jieba_vector']:
    count += 1
    print("vector---", count)
    for i in vector:
        if not isinstance(i, float) or len(vector) != 100:
            print("error")

pd_blog_content.to_pickle(
    '/home/cike/PycharmProjects/pythondata/csdnSMP/Train_DATA/some_blogcontent_vector_normalize.pkl')

"""
begin read blog content of pickle, and training traindata and testdata
"""
pd_blog_content = pd.read_pickle(
    '/home/cike/PycharmProjects/pythondata/csdnSMP/Train_DATA/some_blogcontent_vector_normalize.pkl')

pd_train_data = pd.read_csv(dp.TrainCsv, encoding="utf8", index_col=0)

pd_dev_data = pd.read_csv(dp.DevCsv, encoding="utf8", index_col=0)

columns_train = list(pd_train_data.columns)
del columns_train[1]

columns_dev = list(pd_dev_data.columns)
del columns_dev[0]

import math
import numpy as np


def f(x):
    if x and not (type(x) == float and math.isnan(x)):
        # print("x:",x)
        # print("type:",type(x))
        return eval(x)
    else:
        return []


for i in columns_train:
    print("column name:", i)
    pd_train_data[i] = pd_train_data[i].apply(f)

for i in columns_dev:
    print("column name:", i)
    pd_dev_data[i] = pd_dev_data[i].apply(f)

from imp import reload
import utils.data_path as dp

reload(dp)
pd_train_data.to_pickle(dp.TrainPKL)
pd_dev_data.to_pickle(dp.DevPKL)

pd_train_data_pkl = pd.read_pickle(dp.TrainPKL)
pd_dev_data_pkl = pd.read_pickle(dp.DevPKL)

pd_blog_content = pd.read_pickle(
    '/home/cike/PycharmProjects/pythondata/csdnSMP/Train_DATA/some_blogcontent_vector_normalize.pkl')

