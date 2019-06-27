
from random import shuffle
import tensorflow as tf
import gensim
import jieba
import numpy as np


def getWords(input):
    wordList = []
    trans = []
    lineList = []
    # with open(file,'r',encoding='utf-8') as f:
    #     lines = f.readlines()
    # for line in lines:
    trans = jieba.lcut(input.replace('\n', ''), cut_all = False)
    for word in trans:
        if word not in stopWord:
            wordList.append(word)
    lineList.append(wordList)
        # wordList = []
    return lineList

def makeData(posPath):
# def makeData(posPath,negPath):
    #获取词汇，返回类型为[[word1,word2...],[word1,word2...],...]
    pos = getWords(posPath)
    print("The positive data's length is :", len(pos))
    # neg = getWords(negPath)
    # print("The negative data's length is :", len(neg))
    #将评价数据转换为矩阵，返回类型为array
    posArray, posSteps = words2Array(pos)
    # negArray, negSteps = words2Array(neg)
    #将积极数据和消极数据混合在一起打乱，制作数据集
    # Data, Steps, Labels = convert2Data(posArray, negArray, posSteps, negSteps)
    Data, Steps, Labels = convert2Data(posArray, posSteps,)
    return Data, Steps, Labels


def words2Array(lineList):
    linesArray=[]
    wordsArray=[]
    steps = []
    for line in lineList:
        t = 0
        p = 0
        for i in range(MAX_SIZE):
            if i<len(line):
                try:
                    wordsArray.append(model.wv.word_vec(line[i]))
                    p = p + 1
                except KeyError:
                    t=t+1
                    continue
            else:
               wordsArray.append(np.array([0.0]*dimsh))
        for i in range(t):
            wordsArray.append(np.array([0.0]*dimsh))
        steps.append(p)
        linesArray.append(wordsArray)
        wordsArray = []
    linesArray = np.array(linesArray)
    steps = np.array(steps)
    return linesArray, steps

# def convert2Data(posArray, negArray, posStep, negStep):
def convert2Data(posArray,  posStep):
    randIt = []
    data = []
    steps = []
    labels = []
    # for i in range(len(posArray)):
    #     randIt.append([posArray[i], posStep[i], [1,0]])
    for i in range(len(posArray)):
        randIt.append([posArray[i], posStep[i], [0,1]])
    shuffle(randIt)
    for i in range(len(randIt)):
        data.append(randIt[i][0])
        steps.append(randIt[i][1])
        labels.append(randIt[i][2])
    data = np.array(data)
    # steps = np.array(steps)
    return data, steps, labels


def makeStopWord():
    with open('停用词.txt','r',encoding = 'utf-8') as f:
        lines = f.readlines()
    stopWord = []
    for line in lines:
        words = jieba.lcut(line,cut_all = False)
        for word in words:
            stopWord.append(word)
    return stopWord

word2vec_path = './word2vec/word2vec.model'
model = gensim.models.Word2Vec.load(word2vec_path)
dimsh = model.vector_size
MAX_SIZE = 25
stopWord = makeStopWord()

# testData, testSteps, testLabels = makeData('太烂，买个手机，颜色发错了，没有责任心')




    # graph = tf.get_default_graph()
    # new_x = graph().get_tensor_by_name("x:0")
    # new_y = graph().get_tensor_by_name("y:0")
    # x = graph.get_operation_by_name('x'),



if __name__ == '__main__':
    # senceten = input('-------------
    string = '这次小米8是最差一次，电池一天两充不够用，数据线也生硬硬的，好像给人调包了电池和数据线。红米2/红米5/小米3都用过，这次令我太失望啦。'
    string1 = '好，很好，非常好。就是屏幕有点大，感觉不适合女生' #print [0.8230048  0.17699519]
    a, b, c = makeData(string)
    with tf.Session() as sess:
        gragh = tf.get_default_graph()
        new_saver = tf.train.import_meta_graph('./model/model.ckpt.meta')
        new_saver.restore(sess, './model/model.ckpt')
        x = gragh.get_tensor_by_name('x:0')
        y = gragh.get_tensor_by_name('y:0')
        relust = sess.run(y, feed_dict={"x:0": a})
        print(relust)