"""
温度预测服务 - 主运行文件
从 data.txt 读取数据，进行温度预测和告警分析
"""

import numpy as np
from alert import alertLevel
from predictAnswer import predict, predictLongTime
from dataProcess import initdata, addArr, testData
from predictProcess import Modelpredict, getDone


def main():
    """
    主函数：读取数据并进行温度预测
    """
    print("=" * 60)
    print("温度预测系统启动")
    print("=" * 60)
    
    # ============ 输入参数 ============
    print("\n【步骤 1】读取输入数据...")
    
    # 1. dataReal: n*6 - 温度数据，其中n为开关柜数量，6是6个点位（上ABC、下ABC）
    # 从 input.txt 读取：6 行，每行 n 列（n 个开关柜）
    dataReal = testData('input.txt') 
    print(f"  ✓ dataReal (温度数据): {dataReal.shape} - {dataReal.shape[0]}个开关柜 × 6个点位")
    
    # 2. time: 预测时间长度（0=下一时刻，1=一天，7=一周，15=半个月）
    time = 0  # 可以根据需要修改：0, 1, 7, 15
    print(f"  ✓ time (预测时间长度): {time} ({'下一时刻' if time == 0 else f'{time}天'})")
    
    # 3. instan: 1*n - 当前负载，n为开关柜数量
    n_switches = dataReal.shape[0]  # 开关柜数量
    instant = np.array([7.0, 6.8, 7.6, 5.5, 4.5, 2.4, 2.4][:n_switches])
    print(f"  ✓ instant (当前负载): {instant.shape} - {n_switches}个开关柜")
    
    # 4. rate: 1*n - 额定负载，n为开关柜数量
    current = np.array([7.0, 6.8, 7.6, 5.5, 4.5, 2.4, 2.4][:n_switches])
    print(f"  ✓ rate (额定负载): {current.shape} - {n_switches}个开关柜")
    
    # 5. fan: 1 - 风机属性（1=有风机，0=无风机）
    fan = 0  # 可以根据实际情况修改：0 或 1
    print(f"  ✓ fan (风机属性): {fan} ({'有风机' if fan == 1 else '无风机'})")
    
    # 6. dataHistory: 6*n*(time*96) - 同期历史温度数据（可选，暂时为空）
    dataHistory = []  # 如果需要长时间预测，需要提供历史数据
    print(f"  ✓ dataHistory (历史数据): {'已提供' if len(dataHistory) > 0 else '未提供'}")
    
    print("\n【步骤 2】开始预测计算...")
    
    # 检查历史数据是否足够（时序模型需要至少96个时间点）
    from dataProcess import initdata
    history_data = initdata("record.txt", dataReal.shape[0])
    history_length = len(history_data[0][0]) if len(history_data) > 0 and len(history_data[0]) > 0 else 0
    if history_length < 96:
        print(f"\n⚠️  警告：历史数据不足！")
        print(f"  当前历史数据长度: {history_length} 个时间点")
        print(f"  时序模型需要至少: 96 个时间点")
    
    # ============ 调用预测函数 ============
    dataPredict, isOpen, dataPredictLongTime, level, healthAssessment = getDone(
        [],           # data: 历史数据（由 getDone 内部从 record.txt 读取，这里传空列表即可）
        dataReal,     # dataReal: 当前温度数据 (n*6 或 6*n，函数内部会处理)
        time,         # time: 预测时间长度
        instant,      # instan: 当前负载 (1*n)
        current,      # rate: 额定负载 (1*n)
        fan,          # fan: 风机属性 (0 或 1)
        dataHistory   # dataHistory: 同期历史温度数据 (可选)
    )
    
    print("  ✓ 预测计算完成\n")
    
    # ============ 输出结果 ============
    print("=" * 60)
    print("预测结果输出")
    print("=" * 60)
    
    # 1. dataPredict: n*6 - 预测数据（下一时刻），n个开关柜 × 6个点位
    print("\n【输出 1】dataPredict - 下一时刻预测温度数据")
    print(f"  形状: {dataPredict.shape} ({dataPredict.shape[0]}个开关柜 × {dataPredict.shape[1]}个点位)")
    print("  点位说明: [上A, 上B, 上C, 下A, 下B, 下C]")
    print("  数据（按点位显示）:")
    point_names = ['上A', '上B', '上C', '下A', '下B', '下C']
    # dataPredict 是 (n, 6)，需要转置为 (6, n) 来按点位显示
    dataPredict_transposed = dataPredict.T  # 转置为 (6, n)
    for i, point_name in enumerate(point_names):
        print(f"    {point_name}: {dataPredict_transposed[i, :]}")
    
    # # 2. isOpen: n*6 - 当前开关柜是否为活跃（通电）状态
    # print("\n【输出 2】isOpen - 开关柜活跃状态")
    # print(f"  形状: {isOpen.shape} ({isOpen.shape[0]}个开关柜 × {isOpen.shape[1]}个点位)")
    # print("  说明: True=活跃/通电, False=不活跃/断电")
    # print("  数据（按点位显示）:")
    # # isOpen 是 (n, 6)，需要转置为 (6, n) 来按点位显示
    # isOpen_transposed = isOpen.T  # 转置为 (6, n)
    # for i, point_name in enumerate(point_names):
    #     status_str = [('活跃' if status else '断电') for status in isOpen_transposed[i, :]]
    #     print(f"    {point_name}: {status_str}")
    
    # 3. dataPredictLongTime: 6*n*(time*96) - 长时间预测结果（如果 time > 0）
    if time != 0 and len(dataPredictLongTime) > 0:
        print("\n【输出 3】dataPredictLongTime - 长时间预测结果")
        print(f"  形状: {dataPredictLongTime.shape}")
        print(f"  说明: {time}天的预测数据 ({time}天 × 96个时间点)")
        print(f"  数据预览 (每个点位的前5个时间点):")
        for i, point_name in enumerate(point_names):
            if i < len(dataPredictLongTime):
                preview = dataPredictLongTime[i][0, :5] if dataPredictLongTime[i].shape[0] > 0 else []
                print(f"    {point_name} (开关柜1): {preview}...")
    else:
        print("\n【输出 3】dataPredictLongTime - 长时间预测结果")
        print(f"  当前模式: time={time} (下一时刻预测)")
        print(f"  预测数据 (dataPredict): {dataPredict.shape}")
        print("  数据:")
        # 按开关柜显示预测结果
        for switch_idx in range(dataPredict.shape[0]):
            print(f"    开关柜{switch_idx+1}: {dataPredict[switch_idx, :]}")
    
    # 4. level: 告警等级
    print("\n【输出 4】level - 告警等级")
    if isinstance(level, np.ndarray):
        print(f"  形状: {level.shape}")
        print("  说明: 0=正常, 1=一级告警, 2=二级告警, 3=三级告警")
        if level.ndim == 2:
            # 如果是 (n, 6) 形状，转置为 (6, n) 来按点位显示
            level_transposed = level.T if level.shape[1] == 6 else level
            print("  数据（按点位显示）:")
            level_names = ['正常', '一级', '二级', '三级']
            for i, point_name in enumerate(point_names):
                if i < level_transposed.shape[0]:
                    level_str = [level_names[int(l)] if 0 <= int(l) < 4 else '未知' 
                                for l in level_transposed[i, :]]
                    print(f"    {point_name}: {level_str}")
        else:
            print(f"  数据: {level}")
    else:
        print(f"  全局告警等级: {level}")
        level_names = {0: '正常', 1: '一级告警', 2: '二级告警', 3: '三级告警'}
        print(f"  说明: {level_names.get(level, '未知')}")
    
    # 5. healthAssessment: 健康评估（如果有）
    if healthAssessment is not None and len(healthAssessment) > 0:
        print("\n【输出 5】healthAssessment - 健康评估")
        print(f"  数据: {healthAssessment}")
    
    print("\n" + "=" * 60)
    print("预测完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
