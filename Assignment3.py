# -*- coding: utf-8 -*-
# @Time    : 2019/2/20 9:02
# @Author  : Yunjie Cao
# @FileName: Assignment3.py
# @Software: PyCharm
# @Email   ：YunjieCao@hotmail.com


import matplotlib.pyplot as plt
import numpy as np
import random
import copy
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
    """
    using the whole dataset as a batch
    visualize the result in BGDx.jpg
    """
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
    """
    randomly choose a data to train
    visualize the result in SGDx.jpg
    """
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
    """
    generate file containing data in LIBSVM format
    """
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
    """
    read LIBSVM format data
    :return: [[x0, x1, y]...]
    """
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
    """
    Linear classifier using perceptron algorithm
    evaluate the algorithm using leave-one-out cross validation
    Data : (x0, x1, 1) or (x0, x1, -1)
    """
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
            """
            loss function: L = -y * sign(wx+b)
            """
            for i in range(len(train_datas)):
                train = train_datas[i]
                x1, x2, y = train
                predict = sign(weight[0] * x1 + weight[1] * x2 + bias)
                if y * predict <= 0:
                    wrong += 1
                else:
                    corrrect += 1
            if len(train_datas)-corrrect<=eps: # stop criterion
                break
        train = TrainData[0]
        x1, x2, y = train
        predict = sign(weight[0] * x1 + weight[1] * x2 + bias)
        if y * predict >0:
            correctCnt+=1
    print('Perceptron leave-one-out cross validation accuracy {}'.format(correctCnt/LeaveLimit))


def LogicRegression(Data):
    """
    Linear classifier using logistic regression algorithm
    Stochastic optimizer
    evaluate the algorithm using leave-one-out cross validation
    Data: (x0, x1, 1) or (x0, x1, 0)
    """

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    dataIn = []
    dataLabel = []
    for d in Data:
        x, y, l = d
        if l == 1:
            dataIn.append([x, y])
            dataLabel.append(1)
        else:
            dataIn.append([x, y])
            dataLabel.append(0)
    correctCnt = 0
    M = len(dataIn)
    for i in range(M):
        leave_one_out_in = [dataIn[i]]
        leave_one_out_label = [dataLabel[i]]
        tempin = copy.deepcopy(dataIn)
        templabel = copy.deepcopy(dataLabel)
        del dataIn[i]
        del dataLabel[i]
        dataIn = np.mat(np.insert(dataIn, 0, 1, axis=1))  # (m,n)
        dataLabel = np.mat(dataLabel).transpose()  # (m,1)
        leave_one_out_in = np.mat(np.insert(leave_one_out_in, 0, 1, axis=1))
        leave_one_out_label = np.mat(leave_one_out_label).transpose()
        m, n = dataIn.shape
        weights = np.zeros((n, 1))
        learning_rate = 0.01
        maxIter = 10000
        """
        loss function: L = -y * log(y_hat) - (1-y) * log(1-y_hat)
        """
        for i in range(maxIter):
            y_hat = sigmoid(dataIn * weights)
            weights = weights + learning_rate * dataIn.transpose() * (dataLabel - y_hat)
        predict = sigmoid(leave_one_out_in * weights)
        predict[predict >= 0.5] = 1
        predict[predict <= 0.5] = 0
        if predict[0][0] == leave_one_out_label[0][0]:
            correctCnt = correctCnt + 1
        dataIn = tempin
        dataLabel = templabel
    print("Logistic Regression leave-one-out cross validation accuracy {}".format(correctCnt / M))


if __name__=="__main__":
    linearRegression(scale_xE, scale_yE)
    DataSet = 1
    linearRegression(scale_xF, scale_yF)
    LIBSVM = generateLIBSVM()
    TrainData = readLIBSVM(LIBSVM)
    Perceptron(TrainData)
    LogicRegression(TrainData)
