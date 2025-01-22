import pandas as pd
import numpy as np
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import KShape, TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.preprocessing import TimeSeriesScalerMinMax
import matplotlib.pyplot as plt
import tslearn.metrics as metrics
from tslearn.clustering import silhouette_score
import time
from sklearn.metrics import davies_bouldin_score,calinski_harabasz_score
import sklearn
from tslearn.metrics import soft_dtw
from tslearn.utils import to_time_series_dataset


# 先将原始数据降维变成时间跨度为10min，15min，1h的数据，再进行转置后输出文件，
def Tmp_data_preparation():

    # SamplingInterval代表每个多少个点采样，
    # SamplingInterval=12代表每小时，对应下面数据集长度为24，
    # SamplingInterval=3代表每十五分钟，对应PointNumber长度为96
    # SamplingInterval=2代表每十分钟，对应PointNumber长度为144，
    # SamplingInterval=6代表每半小时，对应PointNumber长度为48，

    SamplingInterval = 1
    # 选择数据集长度
    PointNumber = 288 # PointNumber为一天数据有多少个点，下面依据PointNumber的大小进行转置
    Days = 412  # 总天数

    # 读取csv文件
    data = pd.read_csv('./Filtered data/DataSet-412day.csv')

    # 获取需要处理的三列数据
    cols = ['Ta', 'Tb', 'Tc']
    df = data[cols]

    # 每SamplingInterval个数据取一个值
    df = df.iloc[::SamplingInterval, :]

    data_len = PointNumber * Days  # 每天的点数*天数=数据集总长度
    df = df.head(data_len) #取数据集前 data_len长度 个数据

    # 读取某相绕组温度
    TmpA= np.array(df[["Ta"]]).reshape(-1, 1)
    TmpB= np.array(df[["Tb"]]).reshape(-1, 1)
    TmpC= np.array(df[["Tc"]]).reshape(-1, 1)



    # 允许的最大连续重复值
    M = 287
    # outputdataA = np.zeros((data_len // PointNumber, PointNumber))
    outputdataA = np.empty((data_len // PointNumber, PointNumber))
    # n记录当前写到新数组的第几行了
    n = 0
    # 记录哪个i对应的数据被删了
    L = []
    # i每次增加PointNumber，将每天的数据取出，转置后放入outputdata中
    for i in range(0, data_len, PointNumber):
        aday = TmpA[i:i + PointNumber, :]
        aday = aday.transpose()

        # 记录该行有多少值会连续重复
        max = 0
        pre = aday[0][0]
        temp = 0
        for j in range(PointNumber-1):
            if aday[0][j+1]== pre:
                temp = temp + 1
            else:
                if temp > max:
                    max = temp
                    temp = 0
            pre = aday[0][j+1]
        if temp > max:
            max = temp
        # 若连续重复的值小于等于M个则保留
        if max <= M:
            outputdataA[n, :] = aday
            n = n + 1
        else: L.append(i)
    # df_outputdataA = pd.DataFrame(outputdataA[:n,:])

    # 输出温度数据到csv文件,如果不设置index=False会多出一列行索引
    # df_outputdataA.to_csv("./myresult/Ta_{}point_{}day.csv".format(PointNumber, n), index=False, header="", encoding='gb2312')

    # B相
    outputdataB = np.zeros((data_len // PointNumber, PointNumber))
    n = 0
    # i每次增加PointNumber，将每天的数据取出，转置后放入outputdata中
    for i in range(0, data_len, PointNumber):
        if i in L:
            continue
        else:
            aday = TmpB[i:i + PointNumber, :]
            aday = aday.transpose()

            # 记录该行有多少值会连续重复
            max = 0
            pre = aday[0][0]
            temp = 0
            for j in range(PointNumber - 1):
                if aday[0][j + 1] == pre:
                    temp = temp + 1
                else:
                    if temp > max:
                        max = temp
                        temp = 0
                pre = aday[0][j + 1]
            if temp > max:
                max = temp
            # 若连续重复的值小于等于M个则保留
            if max <= M:
                outputdataB[n, :] = aday
                n = n + 1
            else: L.append(i)
    # df_outputdataB = pd.DataFrame(outputdataB[:n, :])
    # 输出温度数据到csv文件,如果不设置index=False会多出一列行索引
    # df_outputdataB.to_csv("./myresult/Tb_{}point_{}day.csv".format(PointNumber, n), index=False, header="", encoding='gb2312')

    # C相
    outputdataC = np.zeros((data_len // PointNumber, PointNumber))
    n = 0
    # i每次增加PointNumber，将每天的数据取出，转置后放入outputdata中
    for i in range(0, data_len, PointNumber):
        if i in L:
            continue
        else:
            aday = TmpC[i:i + PointNumber, :]
            aday = aday.transpose()

            # 记录该行有多少值会连续重复
            max = 0
            pre = aday[0][0]
            temp = 1
            for j in range(PointNumber - 1):
                if aday[0][j + 1] == pre:
                    temp = temp + 1
                else:
                    if temp > max:
                        max = temp
                        temp = 1
                pre = aday[0][j + 1]
            if temp > max:
                max = temp
            # 若连续重复的值小于等于M个则保留
            if max <= M:
                outputdataC[n, :] = aday
                n = n + 1
            else:
                L.append(i)
    # 得到最终的L,再过一遍TA\TB
    # -----------------
    df_outputdataC = pd.DataFrame(outputdataC[:n, :])

    # 输出温度数据到csv文件,如果不设置index=False会多出一列行索引
    df_outputdataC.to_csv("./myresult/Tc_{}point_{}day.csv".format(PointNumber, n), index=False, header="", encoding='gb2312')

    # 利用L再过一遍TA\TB
    # A相
    outputdataA = np.zeros((data_len // PointNumber, PointNumber))
    n = 0
    # i每次增加PointNumber，将每天的数据取出，转置后放入outputdata中
    for i in range(0, data_len, PointNumber):
        if i in L:
            continue
        # else:
        aday = TmpA[i:i + PointNumber, :]
        aday = aday.transpose()
        #
        #     # 记录该行有多少值会连续重复
        #     max = 0
        #     pre = aday[0][0]
        #     temp = 0
        #     for j in range(PointNumber - 1):
        #         if aday[0][j + 1] == pre:
        #             temp = temp + 1
        #         else:
        #             if temp > max:
        #                 max = temp
        #                 temp = 0
        #         pre = aday[0][j + 1]
        #     if temp > max:
        #         max = temp
        #     # 若连续重复的值小于等于M个则保留
        #     if max <= M:
        outputdataA[n, :] = aday
        n = n + 1
    df_outputdataA = pd.DataFrame(outputdataA[:n, :])
    # 输出温度数据到csv文件,如果不设置index=False会多出一列行索引
    df_outputdataA.to_csv("./myresult/Ta_{}point_{}day.csv".format(PointNumber, n), index=False, header="",
                          encoding='gb2312')

   # B相
    outputdataB = np.zeros((data_len // PointNumber, PointNumber))
    n = 0
    # i每次增加PointNumber，将每天的数据取出，转置后放入outputdata中
    for i in range(0, data_len, PointNumber):
        if i in L:
            continue
        # else:
        aday = TmpB[i:i + PointNumber, :]
        aday = aday.transpose()
        #
        #     # 记录该行有多少值会连续重复
        #     max = 0
        #     pre = aday[0][0]
        #     temp = 0
        #     for j in range(PointNumber - 1):
        #         if aday[0][j + 1] == pre:
        #             temp = temp + 1
        #         else:
        #             if temp > max:
        #                 max = temp;
        #                 temp = 0;
        #         pre = aday[0][j + 1]
        #     if temp > max:
        #         max = temp
        #     # 若连续重复的值小于等于M个则保留
        #     if max <= M:
        outputdataB[n, :] = aday
        n = n + 1
    print('最终删除的天数：')
    print(len(L))
    print('*******不符合要求的天数********')
    L.sort()
    for i in L:
        print(i/24)
    df_outputdataB = pd.DataFrame(outputdataB[:n, :])
    # 输出温度数据到csv文件,如果不设置index=False会多出一列行索引
    df_outputdataB.to_csv("./myresult/Tb_{}point_{}day.csv".format(PointNumber, n), index=False, header="", encoding='gb2312')

    return L

def Other_data_preparation(L):

    SamplingInterval = 3
    # 选择数据集长度
    PointNumber = 96  # PointNumber为一天数据有多少个点，下面依据PointNumber的大小进行转置
    Days = 527  # 总天数

    # 读取csv文件
    data = pd.read_csv('DataSet-527day.csv')

    # 获取需要处理的数据
    cols = ['Te', 'Ia', 'Ib', 'Ic', 'Upa', 'Upb', 'Upc']
    df = data[cols]

    # 每SamplingInterval个数据取一个值
    df = df.iloc[::SamplingInterval, :]

    data_len = PointNumber * Days  # 每天的点数*天数=数据集总长度
    df = df.head(data_len)  # 取数据集前 data_len长度 个数据

    #
    Te= np.array(df[["Te"]]).reshape(-1, 1)
    Ia= np.array(df[["Ia"]]).reshape(-1, 1)
    Ib= np.array(df[["Ib"]]).reshape(-1, 1)
    Ic= np.array(df[["Ic"]]).reshape(-1, 1)
    Ua= np.array(df[["Upa"]]).reshape(-1, 1)
    Ub= np.array(df[["Upb"]]).reshape(-1, 1)
    Uc= np.array(df[["Upc"]]).reshape(-1, 1)

    # 构造一个全为0的np矩阵，大小为(data_len//PointNumber,PointNumber)  X，Y
    outputdataTe = np.zeros((data_len // PointNumber, PointNumber))
    n = 0
    # i每次增加PointNumber，将每天的数据取出，转置后放入outputdata中
    for i in range(0, data_len, PointNumber):
        if i in L:
            continue
        else:
            aday = Te[i:i + PointNumber, :]
            outputdataTe[n, :] = aday.transpose()
            n = n + 1

    df_outputdataTe = pd.DataFrame(outputdataTe[:n, :])
    # 输出温度数据到csv文件,如果不设置index=False会多出一列行索引
    df_outputdataTe.to_csv("./myresult/Te_{}point_{}day.csv".format(PointNumber, n), index=False, header="", encoding='gb2312')

    # B相
    outputdataIa = np.zeros((data_len // PointNumber, PointNumber))
    n = 0
    # i每次增加PointNumber，将每天的数据取出，转置后放入outputdata中
    for i in range(0, data_len, PointNumber):
        if i in L:
            continue
        else:
            aday = Ia[i:i + PointNumber, :]
            outputdataIa[n, :] = aday.transpose()
            n = n + 1
    df_outputdataIa = pd.DataFrame(outputdataIa[:n, :])
    # 输出温度数据到csv文件,如果不设置index=False会多出一列行索引
    df_outputdataIa.to_csv("./myresult/Ia_{}point_{}day.csv".format(PointNumber, n), index=False, header="", encoding='gb2312')


    outputdataIb = np.zeros((data_len // PointNumber, PointNumber))
    n = 0
    # i每次增加PointNumber，将每天的数据取出，转置后放入outputdata中
    for i in range(0, data_len, PointNumber):
        if i in L:
            continue
        else:
            aday = Ib[i:i + PointNumber, :]
            outputdataIb[n, :] = aday.transpose()
            n = n + 1
    df_outputdataIb = pd.DataFrame(outputdataIb[:n, :])
    # 输出温度数据到csv文件,如果不设置index=False会多出一列行索引
    df_outputdataIb.to_csv("./myresult/Ib_{}point_{}day.csv".format(PointNumber, n), index=False, header="", encoding='gb2312')

    outputdataIc = np.zeros((data_len // PointNumber, PointNumber))
    n = 0
    # i每次增加PointNumber，将每天的数据取出，转置后放入outputdata中
    for i in range(0, data_len, PointNumber):
        if i in L:
            continue
        else:
            aday = Ic[i:i + PointNumber, :]
            outputdataIc[n, :] = aday.transpose()
            n = n + 1
    df_outputdataIc = pd.DataFrame(outputdataIc[:n, :])
    # 输出温度数据到csv文件,如果不设置index=False会多出一列行索引
    df_outputdataIc.to_csv("./myresult/Ic_{}point_{}day.csv".format(PointNumber, n), index=False, header="", encoding='gb2312')

    outputdataUa = np.zeros((data_len // PointNumber, PointNumber))
    n = 0
    # i每次增加PointNumber，将每天的数据取出，转置后放入outputdata中
    for i in range(0, data_len, PointNumber):
        if i in L:
            continue
        else:
            aday = Ua[i:i + PointNumber, :]
            outputdataUa[n, :] = aday.transpose()
            n = n + 1
    df_outputdataUa = pd.DataFrame(outputdataUa[:n, :])
    # 输出温度数据到csv文件,如果不设置index=False会多出一列行索引
    df_outputdataUa.to_csv("./myresult/Ua_{}point_{}day.csv".format(PointNumber, n), index=False, header="", encoding='gb2312')

    outputdataUb = np.zeros((data_len // PointNumber, PointNumber))
    n = 0
    # i每次增加PointNumber，将每天的数据取出，转置后放入outputdata中
    for i in range(0, data_len, PointNumber):
        if i in L:
            continue
        else:
            aday = Ub[i:i + PointNumber, :]
            outputdataUb[n, :] = aday.transpose()
            n = n + 1
    df_outputdataUb = pd.DataFrame(outputdataUb[:n, :])
    # 输出温度数据到csv文件,如果不设置index=False会多出一列行索引
    df_outputdataUb.to_csv("./myresult/Ub_{}point_{}day.csv".format(PointNumber, n), index=False, header="", encoding='gb2312')

    outputdataUc = np.zeros((data_len // PointNumber, PointNumber))
    n = 0
    # i每次增加PointNumber，将每天的数据取出，转置后放入outputdata中
    for i in range(0, data_len, PointNumber):
        if i in L:
            continue
        else:
            aday = Uc[i:i + PointNumber, :]
            outputdataUc[n, :] = aday.transpose()
            n = n + 1
    df_outputdataUc = pd.DataFrame(outputdataUc[:n, :])
    # 输出温度数据到csv文件,如果不设置index=False会多出一列行索引
    df_outputdataUc.to_csv("./myresult/Uc_{}point_{}day.csv".format(PointNumber, n), index=False, header="", encoding='gb2312')

def Kw_data_preparation():

    SamplingInterval = 1
    # 选择数据集长度
    PointNumber = 288 # PointNumber为一天数据有多少个点，下面依据PointNumber的大小进行转置
    Days = 717  # 总天数

    # 读取csv文件
    data = pd.read_csv('my_py/DataPre/filtered_data_717days.csv')

    # 获取需要处理的三列数据
    cols = ['Kw']
    df = data[cols]

    # 每SamplingInterval个数据取一个值
    df = df.iloc[::SamplingInterval, :]

    data_len = PointNumber * Days  # 每天的点数*天数=数据集总长度
    df = df.head(data_len) #取数据集前 data_len长度 个数据

    # 读取KW并做归一化
    KW = np.array(df[["Kw"]])
    # KW = TimeSeriesScalerMinMax().fit_transform(KW.transpose()).reshape(-1,1)
    KW = KW.transpose().reshape(-1,1)

    # 允许的最大连续重复值
    M = 287
    outputdataKw = np.empty((data_len // PointNumber, PointNumber))
    # n记录当前写到新数组的第几行了
    n = 0
    # 记录哪个i对应的数据被删了
    L = []
    # i每次增加PointNumber，将每天的数据取出，转置后放入outputdata中
    for i in range(0, data_len, PointNumber):
        aday = KW[i:i + PointNumber, :]
        aday = aday.transpose()
        # 记录该行有多少值会连续重复
        max = 0
        pre = aday[0][0]
        temp = 0
        for j in range(PointNumber-1):
            if aday[0][j+1]== pre:
                temp = temp + 1
            else:
                if temp > max:
                    max = temp
                    temp = 0
            pre = aday[0][j+1]
        if temp > max:
            max = temp
        # 若连续重复的值小于等于M个则保留
        if max <= M:
            outputdataKw[n, :] = aday
            n = n + 1
        else: L.append(i)
    df_outputdataKw = pd.DataFrame(outputdataKw[:n,:])
    print('最终删除的天数：')
    print(len(L))
    # print('*******不符合要求的天数********')
    # L.sort()
    # for i in L:
    #     print(i/24)
    # 输出温度数据到csv文件,如果不设置index=False会多出一列行索引
    df_outputdataKw.to_csv("./myresult/Kw_{}point_{}day.csv".format(PointNumber, n), index=False, header="", encoding='gb2312')
    return L

def Te_data_preparation():

    SamplingInterval = 1
    # 选择数据集长度
    PointNumber = 288 # PointNumber为一天数据有多少个点，下面依据PointNumber的大小进行转置
    Days = 412  # 总天数

    # 读取csv文件
    data = pd.read_csv('./Filtered data/DataSet-412day.csv')

    # 获取需要处理的三列数据
    cols = ['Te']
    df = data[cols]

    # 每SamplingInterval个数据取一个值
    df = df.iloc[::SamplingInterval, :]

    data_len = PointNumber * Days  # 每天的点数*天数=数据集总长度
    df = df.head(data_len) #取数据集前 data_len长度 个数据

    # 读取KW并做归一化
    Te = np.array(df[["Te"]])
    Te = TimeSeriesScalerMinMax().fit_transform(Te.transpose()).reshape(-1,1)

    # 允许的最大连续重复值
    M = 287
    outputdataTe = np.empty((data_len // PointNumber, PointNumber))
    # n记录当前写到新数组的第几行了
    n = 0
    # 记录哪个i对应的数据被删了
    L = []
    # i每次增加PointNumber，将每天的数据取出，转置后放入outputdata中
    for i in range(0, data_len, PointNumber):
        aday = Te[i:i + PointNumber, :]
        aday = aday.transpose()
        # 记录该行有多少值会连续重复
        max = 0
        pre = aday[0][0]
        temp = 0
        for j in range(PointNumber-1):
            if aday[0][j+1]== pre:
                temp = temp + 1
            else:
                if temp > max:
                    max = temp
                    temp = 0
            pre = aday[0][j+1]
        if temp > max:
            max = temp
        # 若连续重复的值小于等于M个则保留
        if max <= M:
            outputdataTe[n, :] = aday
            n = n + 1
        else: L.append(i)
    df_outputdataTe = pd.DataFrame(outputdataTe[:n,:])
    print('最终删除的天数：')
    print(len(L))
    # print('*******不符合要求的天数********')
    # L.sort()
    # for i in L:
    #     print(i/24)
    # 输出温度数据到csv文件,如果不设置index=False会多出一列行索引
    df_outputdataTe.to_csv("./myresult/Te_{}point_{}day.csv".format(PointNumber, n), index=False, header="", encoding='gb2312')
    return L

def plot_elbow(filename, mode):
    """
    手肘法判断聚类个数
    :param X: data
    :param mode: 1使用KShape, 2是kmeans+DTW, 3是kmeans+softDTW,
    :return:
    """
    df = pd.read_csv(filename, dtype=np.float64, header=None)
    X = pd.DataFrame(df)  # 转换为dataframe
    print("输入文件名：" + str(filename))

    X_scaled = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X)
    distortions = []
    for i in range(3, 6):
        if mode == 1:
            ks = KShape(n_clusters=i, n_init=1, verbose=True, random_state=1)
            ks.fit(X_scaled)
        elif mode == 2 :
            ks = TimeSeriesKMeans(n_clusters=i, metric="dtw", max_iter=5, max_iter_barycenter=5, random_state=1).fit(X)
            ks.fit(X_scaled)
        elif mode == 3:
            ks = TimeSeriesKMeans(n_clusters=i, metric="softdtw",metric_params={"gamma": 1}, max_iter=5, max_iter_barycenter=5, random_state=1).fit(X)
            ks.fit(X_scaled)
        distortions.append(ks.inertia_)
    plt.plot(range(3, 6), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion Line')
    plt.show()

def get_softdtw_result_and_output(filename, n,gamma):  # mode='dtw' or 'softdtw'
    """
    使用DTW聚类
    :param input_data: 输入
    :param n: 聚类个数
    :return:
    """

    df = pd.read_csv(filename, dtype=np.float64, header=None)
    input_data = pd.DataFrame(df)  # 转换为dataframe
    print("输入文件名：" + str(filename))
    print("待聚类数据维度:" + str(input_data.shape))
    print("聚类数:" + str(n))

    # 这里记录一下聚类个数，下面n会改变
    NumberOfCenter = n
    # 计时器计时开始
    start = time.perf_counter()

    X = input_data.values.reshape((input_data.shape[0], input_data.shape[1], 1))
    # metric_params就是视频中所说的伽马，值越大正则化效果越大，得出的中心曲线越平滑，值越接近于0效果越接近传统的dtw
    km_dba = TimeSeriesKMeans(n_clusters=n, metric="softdtw", max_iter=5, max_iter_barycenter=5,
                                  metric_params={"gamma": gamma}, random_state=1).fit(X)
    y_pred = km_dba.fit_predict(X)
    X_2d = np.squeeze(X)

    end = time.perf_counter()
    print('k-means-softdtw算法计算时间: %s 秒' % (end - start))

    score = silhouette_score(X, y_pred, metric="softdtw")
    print("k-means-softdtw算法轮廓系数：" + str(score))
    score1 = davies_bouldin_score(X_2d, y_pred)
    print("k-means-softdtw算法DBI：" + str(score1))
    # score2 = calinski_harabasz_score(X_2d,y_pred)
    # print("k-means-softdtw算法CHI：" + str(score2))

    # 初始化距离和聚类中心索引列表
    distances = []
    centroid_indices = []

    # 计算每个样本离其聚类中心的距离并保存到距离列表中
    for i in range(X.shape[0]):
        centroid_idx = km_dba.labels_[i]
        centroid = km_dba.cluster_centers_[centroid_idx].reshape(1, -1)
        distance = soft_dtw(X[i].reshape(1, -1), centroid, gamma)
        distances.append(distance)
        centroid_indices.append(centroid_idx)

    # 将距离和聚类中心索引保存到 DataFrame 中
    df_distances = pd.DataFrame({'Distance to centroid': distances, 'Centroid index': centroid_indices})

    # 将 DataFrame 保存到 CSV 文件中
    df_distances.to_csv('./myresultTrans2/DistancesOfTe.csv', index=False)

    for yi in range(n):
        plt.subplot(3, 3, yi + 1)
        for n in range(len(y_pred)):
            if y_pred[n] == yi:
                plt.plot(X[n], "k-", alpha=.3)
        plt.plot(km_dba.cluster_centers_[yi].ravel(), "r-")
        plt.text(0.05, 0.85, 'Cluster %d' % (yi + 1),
                 transform=plt.gca().transAxes)
        if yi == 0:
            plt.title("$k$-means-softdtw_{}type_gamma={}".format(input_data.shape[1] - 1,gamma))
    plt.tight_layout()
    plt.show()


    # 将聚类标签插入到最后一列
    output_data = pd.DataFrame(df)
    output_data.insert(loc=output_data.shape[1], column="NEW", value=y_pred)

    # 将聚类中心增添到最后一行
    for i in range(NumberOfCenter):
        center=pd.DataFrame(km_dba.cluster_centers_[i].transpose())
        output_data=output_data.append(center)

    # input_data.shape[1]为输入数据的列数，即一天有多少点，input_data.shape[0]为行数，即多少天
    output_data.to_csv('./myresultTrans2/Te聚类结果_{}类_{}point_{}days.csv'.format(NumberOfCenter,input_data.shape[1] - 1,input_data.shape[0]), index=False, header="", encoding='gb2312')
    return y_pred

def get_kmeans_softdtw(filename, n,gamma):  # mode='dtw' or 'softdtw'
    """
    使用DTW聚类
    :param input_data: 输入
    :param n: 聚类个数
    :return:
    """

    df = pd.read_csv(filename, dtype=np.float64, header=None)
    input_data = pd.DataFrame(df)  # 转换为dataframe
    print("输入文件名：" + str(filename))
    print("待聚类数据维度:" + str(input_data.shape))
    print("聚类数:" + str(n))
    print("gamma:" + str(gamma))

    # X = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(input_data)
    X = TimeSeriesScalerMinMax().fit_transform(input_data)
    # metric_params就是视频中所说的伽马，值越大正则化效果越大，得出的中心曲线越平滑，值越接近于0效果越接近传统的dtw
    km_dba = TimeSeriesKMeans(n_clusters=n, metric="softdtw", max_iter=5, max_iter_barycenter=5,
                                  metric_params={"gamma": gamma}, random_state=1).fit(X)
    y_pred = km_dba.fit_predict(X)
    lables = km_dba.labels_
    X_2d = np.squeeze(X)

    # score = silhouette_score(X, y_pred, metric="softdtw")
    # print("k-means-softdtw算法轮廓系数：" + str(score))

    score1 = davies_bouldin_score(X_2d, lables)
    print("k-means-softdtw算法DBI：" + str(score1))
    # score2 = calinski_harabasz_score(X_2d,lables)
    # print("k-means-softdtw算法CHI：" + str(score2))

    return y_pred

def generate_labels(KwFilePath, TeFilePath, FilteredDataFilePath):
    # 读取三个CSV文件
    kw_data = pd.read_csv(KwFilePath, header=None)
    te_data = pd.read_csv(TeFilePath, header=None)
    filtered_data = pd.read_csv(FilteredDataFilePath)

    # 根据规定生成标签列
    label_values = kw_data.iloc[:-3, -1] * 3 + te_data.iloc[:-3, -1]
    expanded_labels = np.repeat(label_values, 288)

    # 取消 expanded_labels 的索引，确保与 filtered_data 的索引对齐
    expanded_labels = pd.Series(expanded_labels, name='groupId').reset_index(drop=True)

    # 将扩充后的标签列直接添加到filtered_data的最后一列，并命名为groupId
    filtered_data['groupId'] = expanded_labels

    # 将带有标签列的数据保存为新的CSV文件
    filtered_data.to_csv('./myresultTrans2/Labeled_data.csv', index=False)


# grid search
df = pd.read_csv('./myresultTrans1/Kw_288point_412day.csv', dtype=np.float64, header=None)
input_data = pd.DataFrame(df)  # 转换为dataframe
X = input_data.values.reshape((input_data.shape[0], input_data.shape[1], 1))

# DBIs = []
# SCs = []

for n in range(2,7):
    for gamma in range(1,10,1):
        km_dba = TimeSeriesKMeans(n_clusters=n, metric="softdtw", max_iter=5, max_iter_barycenter=5,  metric_params={"gamma": (gamma/10)}, random_state=1).fit(X)
        lables = km_dba.labels_
        X_2d = np.squeeze(X)
        DBI = davies_bouldin_score(X_2d, lables)
        SC = silhouette_score(X_2d, lables, metric="softdtw")
        print('聚类数:' + str(n),',gamma:' + str(gamma/10),',DBI:'+ str(DBI),'SC:'+ str(SC))

