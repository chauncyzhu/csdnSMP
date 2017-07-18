# coding=utf8
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
pd_train = pd.read_csv(dp.TrainCsv,index_col=0,encoding="utf8")
pd_train["blog_uid"] = pd_train["blog_uid"].apply(eval)
train_blog_uids = list(itertools.chain(*list(pd_train["blog_uid"])))
print("train_blog_uids length---",len(train_blog_uids))
train_unique_blog_uids = list(set(train_blog_uids))
print("train_blog_uids length---",len(train_unique_blog_uids))

# voca = {word:index for index,word in enumerate(train_unique_blog_uids)}

pd_dev = pd.read_csv(dp.DevCsv,index_col=0,encoding="utf8")
pd_dev["blog_uid"] = pd_dev["blog_uid"].apply(eval)
dev_blog_uids = list(itertools.chain(*list(pd_dev["blog_uid"])))
print("dev_blog_uids length---",len(dev_blog_uids))
dev_unique_blog_uids = list(set(dev_blog_uids))
print("dev_blog_uids length---",len(dev_unique_blog_uids))

print("in train but not in dev blog length---",len(set(train_unique_blog_uids)-set(dev_unique_blog_uids)))
print("in train also in dev blog length---",len(set(train_unique_blog_uids) & set(dev_unique_blog_uids)))

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

blog_id_list = list(set(train_unique_blog_uids) | set(dev_unique_blog_uids))

# 获取blog id对应的博文
blog_id_total = []
with open(dp.BlogContentTxt,encoding="utf8") as f:
    line = f.readline()
    while line:
        line = line.strip()
        blog_id = line[:line.index("\001")]
        blog_id_total.append(blog_id)
        print("blog id----",blog_id)
        line = f.readline()



"""
获取post，观察两者是否相等
"""
pd_post = pd.read_csv(dp.PostTxt,encoding="utf8",sep="\001")
pd_post.columns = ['uid','blog_id','time']

print(len(blog_id_total))
print(len(pd_post))

print(len(set(blog_id_total)))
print(len(set(list(pd_post['blog_id']))))

