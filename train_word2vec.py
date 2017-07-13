# coding=gbk
"""
    利用提供的语料库来训练词向量
"""
import gensim
import utils.data_path as dp
import jieba


def target_train(filename,target_model_file):
    """
    直接对指定数据文件训练word2vec，可能会因为数据量过大而内存溢出，经过修改后传入已经分词后的数据
    :param filename:需要读取的数据文件名，已经分好词了
    :param target_model_file:需要写入的模型文件名
    :return:
    """
    file = open(filename,encoding="utf8")
    # lines = file.readlines()
    # print("总共有---"+str(len(lines)))
    # print("第一个数据类型为---"+str(type(lines[0])))

    line = file.readline()
    count = 0
    result = []
    while line:
        #print("当前行--"+str(line))
        if count % 10000 == 0:
            print("已经读取了---"+str(count)+"---行")
        result.append(eval(line.strip()))
        line = file.readline()
        count += 1
    file.close()

    model = gensim.models.Word2Vec(result, min_count=1)
    model.save(target_model_file)


def continue_train(modelname,filename,target_model_file=None):
    """
    从原模型中加载模型，添加新的数据训练该模型，如果需要写入新的模型文件则写入，否则写入原模型文件
    :param modelname:原模型文件名
    :param filename:数据文件名
    :param target_model_file:写入的目标文件名
    :return:
    """
    model = gensim.models.Word2Vec.load(modelname)

    file = open(filename, encoding="utf8")
    line = file.readline()
    count = 0
    result = []
    while line:
        if count % 10000 == 0:
            print("已经读取了"+str(count)+"行")
        line = line.strip().split("\001")
        result.append(list(jieba.cut(line[2])))
        line = file.readline()
        count += 1
    file.close()

    model.update_vocab(result)   #更新vocabulary
    model.train(result)   #更新参数
    if target_model_file:
        model.save(target_model_file)
    else:
        model.save(modelname)



if __name__ == '__main__':
    # 需要写入的文件
    file_name = dp.BlogContentSegPath + "merge_jieba_0_2.txt"
    target_model_name = dp.CSDNMODELPath + "csdn_model_merge_0_2.m"

    target_train(file_name, target_model_name)   #对整个文件进行训练
