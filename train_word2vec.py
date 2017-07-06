# coding=gbk
"""
    利用提供的语料库来训练词向量
"""
import gensim
import utils.data_path as dp
import jieba
def train(filename):
    file = open(filename,encoding="utf8")
    line = file.readline()
    count = 0
    result = []
    while line:
        if count % 1000 == 0:
            print("已经对"+str(count)+"进行了分词")
        line = line.strip().split("\001")
        #print("文本分词内容为：",str(list(jieba.cut(line[2]))))
        result.append(jieba.cut(line[2]))
        line = file.readline()
        count += 1
        #print(str(list(jieba.cut(line[1]))))
    file.close()
    print(result)

    model = gensim.models.Word2Vec(result, min_count=1)
    model.save(dp.CSDNMODEL)

if __name__ == '__main__':
    train(dp.BlogContentTxt)