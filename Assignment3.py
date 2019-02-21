# -*- coding: utf-8 -*-
# @Time    : 2019/2/20 9:02
# @Author  : Yunjie Cao
# @FileName: Assignment3.py
# @Software: PyCharm
# @Email   ：YunjieCao@hotmail.com


import matplotlib.pyplot as plt
import numpy as np
import random
"""
1. a linear regression program using gradient descent
2. linear classifiers using the perceptron algorithm and logistic regression.
"""
datasetFrance = [[36961,2503], [43621,2992], [15694,1042], [36231,2487], [29945,2014], [40588,2805], [75255,5062], [37709,2643],
                 [30899,2126], [25486,1784], [37497,2641], [40398,2766], [74105,5047], [76725,5312], [18317,1215]]
datasetEnglish = [[35680,2217], [42514,2761], [15162,990], [35298,2274], [29800,1865], [40255,2606], [74532,4805], [37464,2396],
                  [31030,1993], [24843,1627],[36172,2375], [39552,2560], [72545,4597], [75352,4871], [18031,1119]]
xE = [d[0] for d in datasetEnglish]
yE = [d[1] for d in datasetEnglish]
xF = [d[0] for d in datasetFrance]
yF = [d[1] for d in datasetFrance]
# feature scaling and normalization
scale_xE = [(x-min(xE))/(max(xE)-min(xE)) for x in xE]
scale_yE = [(y-min(yE))/(max(yE)-min(yE)) for y in yE]
scale_xF = [(x-min(xF))/(max(xF)-min(xF)) for x in xF]
scale_yF = [(y-min(yF))/(max(yF)-min(yF)) for y in yF]
# size of dataset
m = len(xE)
# 0: English dataset 1: France dataset
DataSet = 0


def error_function(theta, X, y):
    diff = np.dot(X, theta) - y
    return (1. / 2 * m) * np.dot(np.transpose(diff), diff)


def gradient_function(theta, X, y):
    diff = np.dot(X, theta) - y
    return (1. / m) * np.dot(np.transpose(X), diff)


def linearRegression(x, y):
    """y = kx + b"""
    batchGradientDescend(x, y)
    stochasticGradientDescend(x, y)


def batchGradientDescend(x, y):
    plt.scatter(x,y)
    x = np.array(x).reshape(m, 1)
    x0 = np.ones((m, 1))
    x = np.hstack((x0, x))  #[m, 2]
    y = np.array(y).reshape(m, 1)  #[m, 1]
    alpha = 0.01 # learning rate
    theta = np.array([1, 1]).reshape(2, 1)
    loss = float("inf")
    eps = 1e-4
    cnt = 0
    maxIter = 1e5
    while loss>eps and cnt<maxIter:
        cnt+=1
        gradient = gradient_function(theta, x, y)
        theta = theta - alpha * gradient
        loss = error_function(theta, x, y)
    vx = np.arange(0,1.1,0.1)
    vy = vx* theta[1]+theta[0]
    print("dataset {} batch gradient descent, result is y = {} x + {}, loss is {}".format(DataSet, theta[1], theta[0], loss))
    plt.plot(vx,vy)
    plt.savefig('BGD'+str(DataSet) + '.jpg') # BGD means batch gradient descent
    # plt.show()


def stochasticGradientDescend(x, y):
    loss = float("inf")
    eps = 1e-4
    alpha = 0.01
    cnt = 0
    maxIter = 1e5
    plt.scatter(x,y)
    x = np.array(x).reshape(m, 1)
    x0 = np.ones((m, 1))
    x = np.hstack((x0, x))
    y = np.array(y).reshape(m, 1)
    theta = np.array([1, 1]).reshape(2, 1)
    while(loss>eps and cnt<maxIter):
        cnt+=1
        randomSample = random.randint(0,m-1)
        tempx = x[randomSample,:].reshape(1,2)
        tempy = y[randomSample,:].reshape(1,1)
        gradient = gradient_function(theta, tempx, tempy)
        theta = theta-alpha*gradient
        loss = error_function(theta,x,y)
    vx = np.arange(0, 1.1, 0.1)
    vy = vx * theta[1] + theta[0]
    print("dataset {} stochastic gradient descent, result is y = {} x + {}, loss is {}".format(DataSet, theta[1], theta[0], loss))
    plt.plot(vx, vy)
    plt.savefig('SGD' + str(DataSet) + '.jpg')  # BGD means batch gradient descent
    # plt.show()


def generateLIBSVM():
    X = xE+xF
    Y = yE+yF
    scale_x = [(x - min(X)) / (max(X) - min(X)) for x in X]
    scale_y = [(y - min(Y)) / (max(Y) - min(Y)) for y in Y]

    lenE = len(xE)
    f = open('data.txt', 'w')
    for i in range(lenE):
        f.write("1 0:{} 1:{}\n".format(scale_x[i], scale_y[i] ))
    for i in range(lenE, len(X)):
        f.write("2 0:{} 1:{}\n".format(scale_x[i], scale_y[i] ))
    f.close()
    return 'data.txt'


# read data in LIBSVM format
def readLIBSVM(dataset):
    train_data = []
    for line in open(dataset, 'r'):
        to_add = []
        nodes = line.split()
        label = int(nodes.pop(0))
        for i in range(len(nodes)):
            (index, value) = nodes[i].split(':')
            value = float(value)
            to_add.append(value)
        if label == 1:
            to_add.append(1)
        else:
            to_add.append(-1)
        train_data.append(to_add)
    return train_data


def Perceptron(TrainData):
    def sign(v):
        if v>=0:
            return 1
        else:
            return -1
    LeaveLimit = 100
    eps = 2
    correctCnt = 0
    for k in range(LeaveLimit):
        random.shuffle(TrainData)
        weight = [0, 0]
        bias = 0
        learning_rate = 0.01
        train_datas = TrainData[1:]
        cnt = 0
        while True and cnt<1e3:
            cnt+=1
            train = random.choice(train_datas)
            x1, x2, y = train
            predict = sign(weight[0] * x1 + weight[1] * x2 + bias)
            if y * predict <= 0:
                weight[0] = weight[0] + learning_rate * y * x1
                weight[1] = weight[1] + learning_rate * y * x2
                bias = bias + learning_rate * y
            wrong = 0
            corrrect = 0
            for i in range(len(train_datas)):
                train = train_datas[i]
                x1, x2, y = train
                predict = sign(weight[0] * x1 + weight[1] * x2 + bias)
                if y * predict <= 0:
                    wrong += 1
                else:
                    corrrect += 1
            if len(train_datas)-corrrect<=eps:
                break
        train = TrainData[0]
        x1, x2, y = train
        predict = sign(weight[0] * x1 + weight[1] * x2 + bias)
        if y * predict >0:
            correctCnt+=1
    print('Perceptron leave-one-out accuracy {}'.format(correctCnt/LeaveLimit))


def LogicRegression(Data):
    """
    :param Data:  label 1, -1
    :return: accuracy
    """
    def sigmoid(z):
        return 1.0/(1+np.exp(-z))
    LeaveLimit = 50
    M = len(Data)
    correctCnt = 0
    wrongCnt = 0
    batch_size = 5
    threshold = 0.5
    train_batch_cnt = 1e3
    learning_rate = 0.001
    MaxIter = 1e2
    f = open('LogicRe.txt','w')
    for k in range(LeaveLimit): # leave-one-out cross validation
        train_cnt = 0
        weights = np.zeros((3, 1))  # (n,1)
        random.shuffle(Data)
        leave_out = random.randint(0,M-1)
        # train batches
        while train_cnt < train_batch_cnt:
            candidate = []
            start = random.randint(0,M-1)
            step = 0
            while len(candidate)<batch_size:
                if ((start+step)%M!=leave_out):
                    candidate.append((start+step)%M)
                step+=1
            train_cnt += 1
            TrainData = []
            for j in candidate:
                TrainData.append(Data[j])
            dataIn = []
            dataLabel = []
            # traindata [batch_size, 2]
            for i in range(len(TrainData)):
                dataIn.append([TrainData[i][0], TrainData[i][1]])
                if TrainData[i][2]==1:
                    dataLabel.append(0)
                else:
                    dataLabel.append(1)
            m = len(TrainData)
            dataIn = np.array(dataIn).reshape([m,2])
            dataLabel = np.array(dataLabel).reshape([m,1])
            dataIn = np.insert(dataIn, 0, 1, axis=1)  #(m,n)
            m, n = dataIn.shape
            cnt = 0
            # update weights according to this batch
            while cnt<MaxIter:
                h = sigmoid(np.dot(dataIn, weights))
                weights = weights + learning_rate * np.dot(np.transpose(dataIn), (dataLabel-h))/m
                cnt+=1
        _x = np.array([1, Data[leave_out][0], Data[leave_out][1]]).reshape((1,3))
        predict_y = sigmoid(np.dot(_x, weights))
        if predict_y>=threshold:
            predict_y = 1
        else:
            predict_y = 0
        label = 1
        if Data[leave_out][2]==1:
            label = 0
        f.write("predict: {} label: {}\n".format(predict_y, label))
        print("predict: {} label: {}\n".format(predict_y, label))
        if predict_y==label:
            correctCnt+=1
        else:
            wrongCnt+=1
    print("logistic regression leave-one-out cross validation accuracy is {}".format(correctCnt/LeaveLimit))
    f.close()



if __name__=="__main__":
    # linearRegression(scale_xE, scale_yE)
    # DataSet = 1
    # linearRegression(scale_xF, scale_yF)
    LIBSVM = generateLIBSVM()
    TrainData = readLIBSVM(LIBSVM)
    #Perceptron(TrainData)
    LogicRegression(TrainData)
