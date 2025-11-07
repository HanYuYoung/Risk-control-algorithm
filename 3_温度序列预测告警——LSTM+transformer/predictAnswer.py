import numpy as np

# 预测序列
def predict(model,X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    X = np.expand_dims(X,axis=0)
    y = model.predict(X)
    y = y * std + mean
    return y[0][0]

def predictLongTime(model,X,pt):
    longPredict = []
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    X = np.expand_dims(X,axis=0)

    #pt是time*96 time是预测的天数
    for i in range(int((pt)/10)+1): 
        y = model.predict(X)
        X = np.append(X,y)
        longPredict = np.append(longPredict,y)
        X = X[10:]
        X = X.reshape(1,96)
    for i in range(len(longPredict)):
        longPredict[i] = longPredict[i] * std + mean
    longPredict = longPredict[:96]
    longPredict = np.array(longPredict)
    return longPredict