import numpy as np
import matplotlib.pyplot as plt

def alertLevel(dataPredict):
    max_columns = []
    for i in range(len(dataPredict[0])): # 遍历每一列
        max_val = dataPredict[0][i]
        for j in range(1, len(dataPredict)): # 遍历每一行
            if dataPredict[j][i] > max_val:
                max_val = dataPredict[j][i]
        max_columns.append(max_val)
    max_columns = np.array(max_columns) #找到 [上A最大, 上B最大, 上C最大, 下A最大, 下B最大, 下C最大]

    shang_mean = (max_columns[0]+max_columns[1]+max_columns[2])/3 #上组平均：shang_mean = (上A最大 + 上B最大 + 上C最大) / 3
    xia_mean = (max_columns[3]+max_columns[4]+max_columns[5])/3 #下组平均：xia_mean = (下A最大 + 下B最大 + 下C最大) / 3

    #计算偏差率 = (该点位温度 - 同组平均) / 同组平均
    shangA = (max_columns[0]-shang_mean)/shang_mean
    shangB = (max_columns[1]-shang_mean)/shang_mean
    shangC = (max_columns[2]-shang_mean)/shang_mean

    xiaA = (max_columns[3]-xia_mean)/xia_mean
    xiaB = (max_columns[4]-xia_mean)/xia_mean
    xiaC = (max_columns[5]-xia_mean)/xia_mean
    
    level = max(shangA,shangB,shangC,xiaA,xiaB,xiaC)
    if level < 0.35:
        level = 0
    elif level < 0.8:
        level = 1
    elif level < 0.95:
        level = 2
    else:
        level = 3
    print(level)
    return level

def alertLongTime(dataPredictLongTime,dataHistory):
    r = len(dataPredictLongTime)
    l = len(dataPredictLongTime[0])
    matrix = np.zeros((r, l))
    for i in range(r): # 6
        for j in range(l): # 7
            dataPredictLongTimeSequence = np.array(dataPredictLongTime[i][j]).reshape(1,-1)
            dataHistorySequence = np.array(dataHistory[i][j]).reshape(1,-1)
            fangchaPredict = np.var(dataPredictLongTimeSequence)
            fangchaReal= np.var(dataHistorySequence)
            indicate = [0.168,0.21,0.31]
            if fangchaPredict*(1+indicate[0])>fangchaReal and fangchaReal >(1-indicate[0])*fangchaPredict:
                matrix[i][j] = 0
            elif fangchaPredict*(1+indicate[1])>fangchaReal and fangchaReal > (1-indicate[1])*fangchaPredict:
                matrix[i][j] = 1
            elif fangchaPredict*(1+indicate[2])>fangchaReal and fangchaReal > (1-indicate[2])*fangchaPredict:
                matrix[i][j] = 2
            else:
                matrix[i][j] = 3
    return matrix