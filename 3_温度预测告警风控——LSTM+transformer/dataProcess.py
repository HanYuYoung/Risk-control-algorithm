import numpy as np

def addArr(data,col,p):
    # 添加新数据
    col = col.reshape(-1,1)
    pick = data[p]# 第几个点位添加新数据
    if len(pick[0]) == 1:           # 代表断电重启 添加新数据
        for i in range(len(col)):
            pick[i] = col[i]
    elif len(pick[0]) < 20:         # 时间序列还不够长 直接在末尾添加
        pick = np.concatenate((pick, col), axis=1)
    else:                           # 删掉第一列 在末尾添加 模拟滑动效果
        pick = np.delete(pick, 0, axis=1)
        pick = np.concatenate((pick, col), axis=1)
    return pick # 返回添加后的大数组


def read_TXT(lines):
    # 将每六行存储到一个新的二维数组中
    arrays = []
    length = len(lines[0].strip().split())
    for i in range(6):
        arrays.append(np.empty((length, 0)))
    
    for i in range(0, len(lines), 6):
        for j in range(6):
            array = arrays[j] # 那个n*1数组
            values = lines[i+j].strip().split()
            values = np.array(values).reshape(-1,1)
            array = np.concatenate((array, values), axis=1)
            arrays[j]=array
    return arrays


def initdata(path,length):
    data = []

    with open(path, 'r') as file:
        lines = file.readlines()
    # 创建list
    if len(lines)==0:
        for i in range(6):
            data.append([[0] for i in range(length)])
    else:
        data = read_TXT(lines)
    

    for i in range(len(data)):
        data[i] = np.array(data[i])
        data[i] = data[i].astype(float)

    if len(data[0][0])>96:
        for i in range(len(data)):
            data[i] = data[i][:,-96:]
    return data

def testData(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    arr = np.zeros((6, 7))

    for i in range(6):
        row = lines[i].split()
        for j in range(7):
            arr[i, j] = float(row[j])
    arr = arr.T
    return arr

def saveAndWrite(path,dataReal):
    with open(path, 'r') as file:
        lines = file.readlines()
    if len(lines) > 6*96:
        lines = lines[-6*96:]
    lines = np.array(lines)
    lineWrite = []
    for li in lines:
        lin =  np.fromstring(li, sep=' ')
        lineWrite.append(lin)
    lineWrite = np.array(lineWrite)
    np.savetxt(path, lineWrite, delimiter=' ', newline='\n')

    # dataReal的数据保存
    with open(path, 'a') as f:
        for i in range(len(dataReal[0])):
            column = [str(row[i]) for row in dataReal]
            f.write(' '.join(column) + '\n')