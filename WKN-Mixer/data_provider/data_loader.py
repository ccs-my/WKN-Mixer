import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        removeList = ['Time', 'Upa', 'Upb', 'Upc', 'Ula', 'Ulb', 'Ulc', 'S', 'PF', 'Hum']
        cols = list(df_raw.columns)
        for item in removeList:
            cols.remove(item)
        df_raw = df_raw[['Time'] + cols]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        df_stamp = df_raw[['Time']]
        df_stamp['Time'] = pd.to_datetime(df_stamp.Time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['Time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # 归一化
        train_data = df_raw.iloc[border1s[0]: border2s[0], 1:-1].values
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)

        ori_time_stamp = df_raw.iloc[:, 0].values
        scaled_data = self.scaler.transform(df_raw.iloc[:, 1:-1].values)

        Minutes = (self.seq_len+self.pred_len) * 5

        seq_x_mark = list()
        seq_y_mark = list()
        data_X = list()
        data_Y = list()
        time_stamp = list()

        n = len(df_raw)
        # 窗口化
        for i in range(n - self.seq_len - self.pred_len):
            l = datetime.strptime(ori_time_stamp[i], '%Y/%m/%d %H:%M')
            r = datetime.strptime(ori_time_stamp[i + self.seq_len + self.pred_len], '%Y/%m/%d %H:%M')

            if (r - l) > timedelta(minutes=Minutes):
                continue

            time_stamp.append(ori_time_stamp[i+self.seq_len])
            seq_x_mark.append(data_stamp[i:i + self.seq_len])
            data_X.append(scaled_data[i:i + self.seq_len])
            seq_y_mark.append(data_stamp[i + self.seq_len:i + self.seq_len + self.pred_len])
            data_Y.append(scaled_data[i + self.seq_len:i + self.seq_len + self.pred_len])

        self.n = np.shape(data_X)[0]

        num_train = int(self.n * 0.7)
        num_test = int(self.n * 0.2)
        num_vali = self.n - num_train - num_test
        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, self.n]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data_X = np.array(data_X[border1:border2])
        self.data_Y = np.array(data_Y[border1:border2])
        self.seq_x_mark = np.array(seq_x_mark[border1:border2])
        self.seq_y_mark = np.array(seq_y_mark[border1:border2])
        self.time_stamp = np.array(time_stamp[border1:border2])

        self.n = self.data_X.shape[0]

    def __getitem__(self, index):
        return self.data_X[index], self.data_Y[index], self.seq_x_mark[index], self.seq_y_mark[index]

    def __len__(self):
        return self.n

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_data(self):
        return self.data_X, self.data_Y, self.time_stamp

    def temp_inverse(self, data):
        bais = np.array(self.scaler.mean_[0:3])
        scale = np.array(self.scaler.scale_[0:3])

        data = data * scale + bais
        return data

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='predict.csv', ori_data_path='train.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.ori_data_path = ori_data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        ori_data = pd.read_csv(os.path.join(self.root_path,self.ori_data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        removeList = ['Time', 'Upa', 'Upb', 'Upc', 'Ula', 'Ulb', 'Ulc', 'S', 'PF', 'Hum']
        cols = list(df_raw.columns)
        for item in removeList:
            cols.remove(item)
        df_raw = df_raw[['Time'] + cols]
        ori_data =  ori_data[['Time'] + cols]

        df_stamp = df_raw[['Time']]
        df_stamp['Time'] = pd.to_datetime(df_stamp.Time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['Time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # 归一化
        train_data = df_raw.iloc[:, 1:-1].values
        self.scaler = StandardScaler()
        l = 0
        r = int(len(ori_data) * 0.7)
        self.scaler.fit(ori_data.iloc[l: r, 1:-1].values)

        ori_time_stamp = df_raw.iloc[:, 0].values
        scaled_data = self.scaler.transform(train_data)

        Minutes = (self.seq_len + self.pred_len) * 5

        seq_x_mark = list()
        seq_y_mark = list()
        data_X = list()
        data_Y = list()
        time_stamp = list()

        n = len(df_raw)
        # 窗口化
        for i in range(n - self.seq_len - self.pred_len):
            l = datetime.strptime(ori_time_stamp[i], '%Y/%m/%d %H:%M')
            r = datetime.strptime(ori_time_stamp[i + self.seq_len + self.pred_len], '%Y/%m/%d %H:%M')

            if (r - l) > timedelta(minutes=Minutes):
                continue

            time_stamp.append(ori_time_stamp[i + self.seq_len])
            seq_x_mark.append(data_stamp[i:i + self.seq_len])
            data_X.append(scaled_data[i:i + self.seq_len])
            seq_y_mark.append(data_stamp[i + self.seq_len:i + self.seq_len + self.pred_len])
            data_Y.append(scaled_data[i + self.seq_len:i + self.seq_len + self.pred_len])

        self.n = np.shape(data_X)[0]

        self.data_X = np.array(data_X)
        self.data_Y = np.array(data_Y)
        self.seq_x_mark = np.array(seq_x_mark)
        self.seq_y_mark = np.array(seq_y_mark)
        self.time_stamp = np.array(time_stamp)

        self.n = self.data_X.shape[0]

    def __getitem__(self, index):
        return self.data_X[index], self.data_Y[index], self.seq_x_mark[index], self.seq_y_mark[index]

    def get_data(self):
        return self.data_X, self.data_Y, self.time_stamp

    def __len__(self):
        return self.n

    def temp_inverse(self, data):
        bais = np.array(self.scaler.mean_[0:3])
        scale = np.array(self.scaler.scale_[0:3])

        data = data * scale + bais
        return data
