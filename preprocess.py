# coding=gbk
"""
    数据读取
"""
import pandas as pd
import utils.data_path as dp
def read_txt(filename,columns_name,target_file=None):
    """
    读取txt数据
    :param filename:
    :return:
    """

    dic = {}
    lines = open(filename,encoding="utf8").readlines()
    for line in lines:
        line = line.strip().split("\001")
        #print(line)
        if line[0] not in dic:
            dic[line[0]] = ["\001".join(line[1:])]
        else:
            dic[line[0]].append("\001".join(line[1:]))

    print("dic length----"+str(len(dic)))

    for key,value in zip(dic.keys(),dic.values()):
        if dic[key] != value:
            print("ERROR!!!!!!!!!")

    pd_data = pd.DataFrame({columns_name[0]:list(dic.keys()),columns_name[1]:list(dic.values())})

    #if target_file:
    #    pd_data.to_csv(target_file,encoding="utf8")

    return pd_data

def read_special_txt(filename,blog_list,target_file=None):
    """
    读取txt数据
    :param filename:所有的blog数据
    :param blog_list:所有的需要用到的blog uid
    :param target_file:需要写入的target_file
    :return:
    """
    blog_dict = {}
    file = open(filename,encoding="utf8")
    line = file.readline()
    while line:
        line = line.strip().split("\001")
        if line[0] in blog_list:
            blog_dict[line[0]] = line[1]
        line = file.readline()

    print("blog length---------",len(blog_dict))

    pd_data = pd.DataFrame({'blog_uid':blog_dict.keys(),'blog_info':blog_dict.values()})

    if target_file:
        pd_data.to_csv(target_file,encoding="utf8")
        print("文件写入完毕！")


def read_pandas(filename):
    """
    pandas 数据读取
    :param filename: csv文件
    :return:
    """
    pd_data = pd.read_csv(filename,encoding="utf8",index_col=0)
    pd_data['blog_uid'] = pd_data['blog_uid'].apply(eval)
    print(pd_data)

    return pd_data


def train_and_dev_data(train_file=None,dev_file=None):
    pd_post = read_txt(dp.PostTxt,['uid','post'],dp.PostCSV)
    pd_brows = read_txt(dp.BrowseTxt,['uid','brows'],dp.BrowseCSV)
    pd_comment = read_txt(dp.CommentTxt,['uid','comment'],dp.CommentCSV)
    pd_voteup = read_txt(dp.VoteupTxt,['uid','voteup'],dp.VoteupCSV)
    pd_votedown = read_txt(dp.VotedownTxt,['uid','votedown'],dp.VotedownCSV)
    pd_favorite = read_txt(dp.FavoriteTxt,['uid','favorite'],dp.FavoriteCSV)

    #print(pd.merge(pd_post,pd_brows,on="uid"))
    # print(pd_post.join([pd_brows,pd_comment,pd_voteup,pd_votedown,pd_favorite],how="inner"))

    pd_follow = read_txt(dp.FollowTxt,['uid','follow'],dp.FollowCSV)
    pd_letter = read_txt(dp.LetterTxt,['uid','letter'],dp.LetterCSV)

    pd_train = read_txt(dp.TrainTxt,['uid','labels'],dp.TrainCsv)

    with open(dp.DevTxt,encoding="utf8") as f:
        pd_dev = pd.DataFrame({"uid": [line.strip() for line in f]})


    pd_train['post'] = [None]*len(pd_train)
    pd_train['brows'] = [None]*len(pd_train)
    pd_train['comment'] = [None]*len(pd_train)
    pd_train['voteup'] = [None]*len(pd_train)
    pd_train['votedown'] = [None]*len(pd_train)
    pd_train['favorite'] = [None]*len(pd_train)
    pd_train['follow'] = [None]*len(pd_train)
    pd_train['letter'] = [None]*len(pd_train)
    pd_train['blog_id'] = [[]]*len(pd_train)

    pd_dev['post'] = [None] * len(pd_dev)
    pd_dev['brows'] = [None] * len(pd_dev)
    pd_dev['comment'] = [None] * len(pd_dev)
    pd_dev['voteup'] = [None] * len(pd_dev)
    pd_dev['votedown'] = [None] * len(pd_dev)
    pd_dev['favorite'] = [None] * len(pd_dev)
    pd_dev['follow'] = [None] * len(pd_dev)
    pd_dev['letter'] = [None] * len(pd_dev)
    pd_dev['blog_id'] = [[]] * len(pd_dev)

    def f(pd_data,pd_train,index,row,con_column,target_column):
        if row[con_column] in list(pd_data[con_column]):  #如果pd中包含该uid，则找出对一个的value值，同时这个uid都是唯一的
            #添加blog的uid
            temp_list = list(pd_data[pd_data[con_column] == row[con_column]][target_column])[0]
            #print("blog info"+str(pd_train['blog_uid'][index]))
            pd_train['blog_id'][index].extend([i.split('\001')[0] for i in temp_list])
            pd_train['blog_id'][index] = list(set(pd_train['blog_id'][index]))
            return temp_list  #确定只包含一个
        else:
            #print("未找到")
            return None #否则返回空


    def g(pd_data,con_column,target_column):
        if row[con_column] in list(pd_data[con_column]):  #如果pd中包含该uid，则找出对一个的value值，同时这个uid都是唯一的
            return list(pd_data[pd_data[con_column] == row[con_column]][target_column])[0]
        else:
            return None #否则返回空


    """
    对train数据进行merge
    """
    print(pd_train)
    count = 0
    for index,row in pd_train.iterrows():
        print("当前index为："+str(index))
        #行为数据
        pd_train['post'][index] = f(pd_post,pd_train,index,row,'uid','post')
        pd_train['brows'][index] = f(pd_brows,pd_train,index,row,'uid','brows')
        pd_train['comment'][index] = f(pd_comment,pd_train,index,row,'uid','comment')
        pd_train['voteup'][index] = f(pd_voteup,pd_train,index,row,'uid','voteup')
        pd_train['votedown'][index] = f(pd_votedown,pd_train,index,row,'uid','votedown')
        pd_train['favorite'][index] = f(pd_favorite,pd_train,index,row,'uid','favorite')

        if None not in list(row):
            count += 1
        #社交数据
        pd_train['follow'][index] = g(pd_follow,'uid','follow')
        pd_train['letter'][index] = g(pd_letter,'uid','letter')
    print("全部数据都有的有---"+str(count))
    print(pd_train)

    #将blog_id对应的uid找出
    def h():
        pass
    pd_dev['blog_uid'] = [[]]*len(pd_dev)
    pd_dev['blog_uid'] = pd_dev['blog_id'].apply(h)


    """
        对dev数据进行merge
    """
    print(pd_dev)
    count = 0
    for index, row in pd_dev.iterrows():
        print("当前index为：" + str(index))
        # 行为数据
        pd_dev['post'][index] = f(pd_post, pd_dev, index, row, 'uid', 'post')
        pd_dev['brows'][index] = f(pd_brows, pd_dev, index, row, 'uid', 'brows')
        pd_dev['comment'][index] = f(pd_comment, pd_dev, index, row, 'uid', 'comment')
        pd_dev['voteup'][index] = f(pd_voteup, pd_dev, index, row, 'uid', 'voteup')
        pd_dev['votedown'][index] = f(pd_votedown, pd_dev, index, row, 'uid', 'votedown')
        pd_dev['favorite'][index] = f(pd_favorite, pd_dev, index, row, 'uid', 'favorite')

        if None not in list(row):
            count += 1
        # 社交数据
        pd_dev['follow'][index] = g(pd_follow, 'uid', 'follow')
        pd_dev['letter'][index] = g(pd_letter, 'uid', 'letter')
    print("全部数据都有的有---" + str(count))
    print(pd_dev)

    if train_file:
        pass
        # pd_train.to_csv(train_file,encoding="utf8")

    if dev_file:
        pd_dev.to_csv(dev_file,encoding="utf8")

    return (pd_train,pd_dev)

def blog_info(filename):
    """
    取出训练集中的博文信息，最终以csv格式存储
    :param:pd_data 训练数据
    :return:csv
    """
    pd_data = read_pandas(filename)
    blog_list = []
    for index,row in pd_data.iterrows():
        print("当前index为："+str(index))
        blog_list.extend(row['blog_uid'])
    blog_list = list(set(blog_list))
    print("一共有------",len(blog_list),blog_list)
    read_special_txt(dp.BlogContentTxt,blog_list,dp.BlogContentCsv)


if __name__ == '__main__':
    train_and_dev_data(dp.TrainCsv,dp.DevCsv)

    #blog_info(dp.TrainCsv)



"""
训练集中
拥有全部行为、社交数据都有的有---2个用户
拥有全部行为数据有
"""