# coding=utf-8
"""
    使用神经网络来训练该数据
"""
import pandas as pd
import numpy as np
import data_path as dp
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dropout, Dense, Activation
from keras.utils import np_utils  # 这个使用还不是很清楚

"""
    通过keras实现lstm神经网络，实际上只要确定好标签、输入、字典，就可以训练该神经网络
"""


def get_pickle_data(pd_pkl_name, columns_name):
    """
    产生embedding_matrix
    :param pd_pkl_name: 需要加载的字典内容
    :param columns_name: 需要保留的列
    :return:
    """
    # 从pd_pkl_name中读取文件
    pd_pkl = pd.read_pickle(pd_pkl_name)

    non_selected = list(set(pd_pkl.columns) - set(columns_name))
    pd_pkl = pd_pkl.drop(non_selected, axis=1)  # 去掉没有选择的列，实际上只保留id以及对应的vector

    # print(pd_pkl)
    return pd_pkl


def lstm(trainData, trainMark, testData, embedding_dim, embedding_matrix, maxlen, output_len):
    # 填充数据，将每个序列长度保持一致
    trainData = list(sequence.pad_sequences(trainData, maxlen=maxlen,
                                            dtype='float64'))  # sequence返回的是一个numpy数组，pad_sequences用于填充指定长度的序列，长则阶段，短则补0，由于下面序号为0时，对应值也为0，因此可以这样
    testData = list(sequence.pad_sequences(testData, maxlen=maxlen,
                                           dtype='float64'))  # sequence返回的是一个numpy数组，pad_sequences用于填充指定长度的序列，长则阶段，短则补0

    # 建立lstm神经网络模型
    model = Sequential()  # 多个网络层的线性堆叠，可以通过传递一个layer的list来构造该模型，也可以通过.add()方法一个个的加上层
    # model.add(Dense(256, input_shape=(train_total_vova_len,)))   #使用全连接的输入层
    model.add(Embedding(len(embedding_matrix), embedding_dim, weights=[embedding_matrix], mask_zero=False,
                        input_length=maxlen))  # 指定输入层，将高维的one-hot转成低维的embedding表示，第一个参数大或等于0的整数，输入数据最大下标+1，第二个参数大于0的整数，代表全连接嵌入的维度
    # lstm层，也是比较核心的层
    model.add(LSTM(256))  # 256对应Embedding输出维度，128是输入维度可以推导出来
    model.add(Dropout(0.5))  # 每次在参数更新的时候以一定的几率断开层的链接，用于防止过拟合
    model.add(Dense(output_len))  # 全连接，这里用于输出层，1代表输出层维度，128代表LSTM层维度可以自行推导出来
    model.add(Activation('softmax'))  # 输出用sigmoid激活函数
    # 编译该模型，categorical_crossentropy（亦称作对数损失，logloss），adam是一种优化器，class_mode表示分类模式
    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    # 正式运行该模型,我知道为什么了，因为没有补0！！每个array的长度是不一样的，因此才会报错
    X = np.array(list(trainData))  # 输入数据
    print("X:", X)
    Y = np.array(list(trainMark))  # 标签
    print("Y:", Y)
    # batch_size：整数，指定进行梯度下降时每个batch包含的样本数
    # nb_epoch：整数，训练的轮数，训练数据将会被遍历nb_epoch次
    model.fit(X, Y, batch_size=200, nb_epoch=10)  # 该函数的X、Y应该是多个输入：numpy list(其中每个元素为numpy.array)，单个输入：numpy.array

    # 进行预测
    A = np.array(list(testData))  # 输入数据
    print("A:", A)
    classes = model.predict(A)  # 这个是预测的数据
    return classes


if __name__ == '__main__':
    """
    整理好标签数据
    """
    embedding_dim = 100
    maxlen = 15772

    pd_embedding = get_pickle_data(dp.SOME_BLOGCONTENT_VECTOR_NORMALIZE, columns_name=['blog_id', 'blog_jieba_vector'])

    # 获得训练数据和测试数据labels_name
    pd_train = get_pickle_data(dp.TrainPKL, columns_name=['labels', 'uid', 'embedding_index'])
    pd_test = get_pickle_data(dp.DevPKL, columns_name=['uid', 'embedding_index'])

    # 获得所有的标签
    # attention:the decode way is 'gbk' not 'utf8'
    with open(dp.LabelSpace, encoding='gbk') as f:
        labels_name = [i.strip() for i in f]
    # print("labels_name---",labels_name)

    labels_len = len(labels_name)


    def f(x):
        a = [0] * labels_len
        x = x[0].split('\001')
        for i in x:
            a[labels_name.index(i)] = 1
        return a


    pd_train['labels'] = pd_train['labels'].apply(f)

    # change embedding
    embedding = np.array(list(pd_embedding['blog_jieba_vector'].apply(list)))

    print("pd_train:\n", pd_train)
    print("pd_test:\n", pd_test)

    # begin training
    dev_classes = lstm(list(pd_train['embedding_index'])[:100], list(pd_train['labels'])[:100],
                       list(pd_test['embedding_index'])[:100],
                       embedding_dim, embedding, maxlen, labels_len)

    print("dev classes:", dev_classes)

    # bottleneck
    import bottleneck as bl

    result = []
    labels_name = np.array(labels_name)
    for classes in dev_classes:
        result.append(labels_name[bl.argpartition(-classes, 3)[:3]])

    pd_result = pd.DataFrame(result)
    pd_result.to_csv(dp.ResultTxt, sep="\001", header=False, index=False, encoding='utf8')


