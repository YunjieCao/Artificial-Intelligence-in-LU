# -*- coding: utf-8 -*-
# @Time    : 2019/2/19 16:15
# @Author  : Yunjie Cao
# @FileName: HMM.py
# @Software: PyCharm
# @Email   ：YunjieCao@hotmail.com

import random
# 8 * 8 grid
# 0:north 1:south 2:west 3:east
dir_ = [[-1,0], [1,0], [0,-1], [0,1]]
def getTransitionProb():
    """
    change direction first, then moves at new direction
    :return:  Transition Probability matrix [256,256]
    """
    Prob = [[0 for i in range(256)]for j in range(256)]
    for p in range(256):
        P_dir = p%4
        i = int(int(p/4)/8)
        j = int(p/4)%8
        if i>0 and i<7 and j>0 and j<7:  # not encountering a wall
            for d in range(4):
                nexti = i + dir_[d][0]
                nextj = j + dir_[d][1]
                nexts = (nexti * 8 + nextj)*4 + d  # the next state
                if d==P_dir:
                    Prob[p][nexts]=0.7
                else:
                    Prob[p][nexts]=0.1
        else:  # may encounter a wall, then determine along the wall or in the corner
            possible_dir = []
            nexti = i + dir_[P_dir][0]
            nextj = j + dir_[P_dir][1]
            if nexti>=0 and nexti<8 and nextj>=0 and nextj<8:  #not encounter wall but attention when it may change direction
                Prob[p][(nexti*8+nextj)*4+P_dir] = 0.7
                for d in range(4):
                    if P_dir!=d:
                        nexti = i + dir_[d][0]
                        nextj = j + dir_[d][1]
                        if nexti >= 0 and nexti < 8 and nextj >= 0 and nextj < 8:
                            possible_dir.append(d)
                for d in possible_dir:
                    nexti = i + dir_[d][0]
                    nextj = j + dir_[d][1]
                    Prob[p][(nexti * 8 + nextj)*4 + d] = 0.3/float(len(possible_dir))
            else:
                for d in range(4):
                    if P_dir!=d:
                        nexti = i + dir_[d][0]
                        nextj = j + dir_[d][1]
                        if nexti >= 0 and nexti < 8 and nextj >= 0 and nextj < 8:
                            possible_dir.append(d)
                for d in possible_dir:
                    nexti = i + dir_[d][0]
                    nextj = j + dir_[d][1]
                    Prob[p][(nexti * 8 + nextj)*4 + d] = 1.0/float(len(possible_dir))

    return Prob


def getEmissionProb():
    """
    :return:  Emission Probability matrix [256, 65]
    """
    Prob = [[0 for i in range(65)]for j in range(256)]
    for p in range(256):
        i = int(int(p / 4) / 8)
        j = int(p / 4) % 8
        Prob[p][i*8+j]=0.1
        possible_surround1 = 0
        possible_surround2 = 0
        for surroundI in range(i-2,i+3):
            for surroundJ in range(j-2,j+3):
                if surroundI>=0 and surroundI<8 and surroundJ>=0 and surroundJ<8:
                    surroundP = surroundI * 8 + surroundJ
                    if abs(surroundI-i)>1 or abs(surroundJ-j)>1:  #n_LS2
                        possible_surround2+=1
                        Prob[p][surroundP]=0.025
                    elif abs(surroundI-i)!=0 and abs(surroundJ-j)!=0:  #n_LS1
                        possible_surround1+=1
                        Prob[p][surroundP]=0.05
        Prob[p][64]=1.0-0.1-0.05*possible_surround1-0.025*possible_surround2
    return Prob


def main():
    transition_prob = getTransitionProb()
    emission_prob = getEmissionProb()
    now_i = random.randint(0,7)
    now_j = random.randint(0,7)
    now_head = random.randint(0,3)
    MaxLoop = 1e4
    cnt = 0
    NotFind = True
    HistorySensor = []
    states = [i for i in range(256)]
    start_prob = [1.0/256.0 for i in range(256)]
    error = 0
    while(cnt<MaxLoop and NotFind):
        now_state = (now_i*8+now_j)*4+now_head
        next_state = moveRobot(transition_prob, now_state)
        next_position = int(next_state/4)
        next_i = int(int(next_state/4)/8)
        next_j = int(next_state/4)%8
        next_head = next_state%4
        print("next postion is ({}, {})".format(next_i, next_j))
        observed_state = obtainSensor(emission_prob, next_state)
        if observed_state==64:
            print("sensor nothing")
        else:
            print("sensor ({}, {})".format(int(observed_state/8), observed_state%8))
        HistorySensor.append(observed_state)
        LocatePosition = LocateRobot(HistorySensor, states, start_prob, transition_prob, emission_prob)
        print("Locate ({}, {})".format(int(LocatePosition / 8), LocatePosition % 8))
        cnt+=1
        error = error + abs(int(LocatePosition/8)-next_i) + abs(LocatePosition%8-next_j)
        if LocatePosition==next_position:
            print("successfully locate robot in {} steps!".format(cnt))
            print("average Manhattan distance {}".format(error/cnt))
            NotFind = False
        now_i = now_i
        now_j = now_j
        now_head = next_head
    if NotFind:
        print("fail to locate in {} steps".format(MaxLoop))
    return error/cnt


def moveRobot(transition_prob, now_state):
    """
    :param transition_prob: transition probability matrix
    :param now_state: now state
    :return: next state
    """
    n = random.random()
    cnt = 0
    for i in range(256):
        n_prob = transition_prob[now_state][i]
        if n_prob==0:
            continue
        else:
            cnt+=n_prob
            if cnt>=n:
                return i
    print("something wrong")


def obtainSensor(emission_prob, now_state):
    """
    :param emission_prob:  emission probability matrix
    :param now_state: now state
    :return: simulated sensor
    """
    n = random.random()
    cnt = 0
    for i in range(65):
        n_prob = emission_prob[now_state][i]
        if n_prob==0:
            continue
        else:
            cnt+=n_prob
            if cnt>=n:
                return i
    print("something is wrong")


def LocateRobot(obs, states, start_p, trans_p, emit_p):
    """
    :param obs: observed sensor information
    :param states: hidden states
    :param start_p: initial probability
    :param trans_p: transition probability matrix
    :param emit_p: emission probability matrix
    :return: predicted position
    """
    V = [[0 for i in range(len(states))]for j in range(len(obs))]
    path = [[0 for i in range(len(obs))]for j in range(len(states))]
    for s in states:
        V[0][s] = start_p[s]*emit_p[s][obs[0]]
        path[s][0] = s
    for t in range(1,len(obs)):
        newPath = [[0 for i in range(len(obs))]for j in range(len(states))]
        for y in states:
            prob = -1
            state = -1
            for y0 in states:
                nprob = V[t-1][y0]*trans_p[y0][y]*emit_p[y][obs[t]]
                if nprob>prob:
                    prob = nprob
                    state = y0
                    V[t][y] = prob
                    for i in range(t):
                        path[state] = newPath[y]
                    newPath[y][t] = y
        path = newPath
    prob = -1
    state = -1
    for y in states:
        if V[len(obs)-1][y]>prob:
            prob = V[len(obs)-1][y]
            state = y
    predictRes = int(path[state][-1]/4)
    return predictRes


if __name__=="__main__":
    evaluation = 0
    for i in range(100):
        evaluation += main()
    print("accuracy is {}".format(evaluation/100))