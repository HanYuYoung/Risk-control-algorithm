from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
import tensorflow as tf
import keras
from keras.models import load_model
import numpy as np

from alert import alertLevel,alertLongTime
from predictAnswer import predict,predictLongTime
from dataProcess import initdata,addArr,read_TXT,saveAndWrite
from assessment import healthAssessment

def Modelpredict(data,col,p,time,instan):
    
#     1.判断长度 补全
 #    1.选择数据
    pick = data[p]
    
#     2.判断就近的是不是4个点是0，也就是说目前这一个小时里面读不出来任何数据，这个有可能损坏了
    max_val = col.max()
    c = len(pick[0])
    isOpen = np.ones((len(pick), 1), dtype=bool) # 判断是否损坏OR关机
    for i in range(len(pick)):
        if col[i] == 0 and c > 3: # 如果当前新数据为0并且序列长大于3 判断是否有连续4个0表示损坏
            if pick[i][c-1] == 0 and pick[i][c-2] == 0 and pick[i][c-3] == 0 and pick[i][c-4] == 0:
                isOpen[i] = False
            else:
                pick[i][c-1] = max_val #不是的话找最大的填补 去进行预测
            
#    3.根据电流去预测
    newCol = [] #只预测下一个时刻的数据
    for i in range(len(pick)): # 在这里进行单个的预测 7*96
        voltage = instan[i]
        if voltage > 1000: #找到合适的模型
            model = load_model("model_data/model.keras")
        else:
            model = load_model("model_data/model.keras")
        if isOpen[i] == False: # 关机不预测
            newCol = np.append(newCol,0)
        else:
            # 判断长度
            l = len(pick[i])
            if l < 96: # 小于96直接返回传入数据
                newCol = np.append(newCol,pick[i][l-1])
            else: #否则进行预测
                newCol = np.append(newCol,predict(model,pick[i]))
    isOpen = isOpen.reshape(-1,1)

    preCol = [] #长时间预测数组
    if time != 0:
        pt = 4*24*time
        for i in range(len(pick)): # 在这里进行96的预测
            if isOpen[i] == False: 
                temp = []
                for i in range(pt):
                    temp.append(0)
                preCol.append(temp)
            else:
                l = len(pick[i])
                if l < 96:
                    temp = []
                    for i in range(pt):
                        temp.append(0)
                    preCol.append(temp)
                    print("数据量过少，不足以预测"+str(time)+"天的数据")
                else:
                    preCol.append(predictLongTime(model,pick[i],pt))
        preCol = np.array(preCol)
    # newCol是预测下一个时刻的7*1的一列数据 isOpen是开关柜7*1的一列数据 preCol是7*pt长时间预测的矩阵
    return newCol, isOpen, preCol
    


def getDone(data,dataReal,time,instan,rate,fan,dataHistory):
    # 原始数据初始化
    path = "record.txt"

    data = initdata(path,dataReal.shape[0]) # 读取record.txt后面96的数据进行预测
    data = np.array(data)
    print(f'读取数据的形状: {data.shape}') # (6, 7, 96)
    saveAndWrite(path,dataReal)  #将输出的最新的那一批数据追加到record.txt末尾
    print(f'传入数据的形状: {dataReal.shape}') # 7*6 7个变电站 6个点
    
    p = len(dataReal) # 检测开关柜的个数
    dataPredict = np.empty((p, 0)) # 每一个点位的预测矩阵 7*0
    dataPredictLongTime = []
    dataOpen = np.empty((p, 0), dtype=bool) 

    # 补全数据：将 dataReal 追加到 data 的末尾，形成新的时序数据
    tempData = dataReal.T  # (7, 6) -> (6, 7)
    temp_expanded = np.expand_dims(tempData, axis=-1)  # 扩展维度(6, 7) -> (6, 7, 1)
    data_updated = np.concatenate((data, temp_expanded), axis=-1) #data: (6, 7, 96) + temp_expanded: (6, 7, 1) -> data_updated: (6, 7, 97)
    if data_updated.shape[2] > 96:
        data_updated = data_updated[:, :, 1:]  # 删除第一个时间点，保留最新的96 [:,:,1:96]
    data = data_updated  # 更新为包含新数据的时序数据，形状 (6, 7, 96)

#   data list6个 数组长
    for i in range(6): # 6行 一行就是7个开关柜 6是6个触头 遍历写在了里面
        newCol, isop, preCol = Modelpredict(data, dataReal[:,i].reshape(-1,1),i,time,instan)
        newCol = newCol.reshape(-1,1)
        dataPredict = np.concatenate((dataPredict, newCol), axis=1)
        isop = isop.reshape(-1,1)
        dataOpen = np.concatenate((dataOpen,isop),axis=1)
        if time!=0:
            dataPredictLongTime.append(preCol)
    
    dataPredictLongTime = np.array(dataPredictLongTime) # 6*7*pt
    for i in range(len(dataPredictLongTime)):
        for j in range(len(dataPredictLongTime[0])):
            dataPredictLongTime[i][j] = np.array(dataPredictLongTime[i][j])

    #告警
    l = alertLevel(dataPredict)
    # l = alertLongTime(dataPredictLongTime,dataHistory)
    if time != 0:
        isAlert = alertLongTime(dataPredictLongTime,dataHistory)

    # 健康评估模块
    column = len(data[0]) 
    if(len(data[0][0])<96):
        dataAssess = np.zeros((6, column)).reshape(1,-1)
    else:
        dataAssess = []
        for i in range(0,column):
            slipe = data[:,i,:]
            # data = np.array(data).reshape(-1, 1)
            getCondition = healthAssessment(slipe.reshape(-1,1))
            dataAssess.append(getCondition)
        dataAssess = np.array(dataAssess).reshape(1,-1)
    return dataPredict,dataOpen,dataPredictLongTime,l,dataAssess
