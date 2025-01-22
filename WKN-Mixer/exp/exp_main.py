import os
import time
import warnings
from datetime import datetime, timedelta
import pandas
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Transformer, DLinear, SCINet, MTSMixer, MTSMatrix, MTSAttn, FNet, Transformer_lite, MTSD, MyMatrix, MyLinear, MyModel, improveMixer, laplaceMixer, morletMixer, mexhatMixer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, R2

warnings.filterwarnings('ignore')

non_transformer = ['DLinear', 'SCINet', 'MTSMixer', 'MTSMatrix', 'MTSAttn', 'FNet', 'Transformer_lite', 'MTSD', 'MyMatrix', 'MyLinear', 'MyModel', 'improveMixer', "laplaceMixer", "morletMixer", "mexhatMixer"]

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
            'DLinear': DLinear,
            'SCINet': SCINet,
            'MTSMixer': MTSMixer,
            'MTSMatrix': MTSMatrix,
            'MTSAttn': MTSAttn,
            'FNet': FNet,
            'Transformer_lite': Transformer_lite,
            'MTSD': MTSD,
            'MyMatrix': MyMatrix,
            'MyLinear': MyLinear,
            'MyModel': MyModel,
            'improveMixer': improveMixer,
            'laplaceMixer': laplaceMixer,
            'morletMixer': morletMixer,
            'mexhatMixer': mexhatMixer,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        total = sum([param.nelement() for param in model.parameters()])
        print('Number of parameters: %.2fM' % (total / 1e6))

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model in non_transformer:
                            outputs = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0] if self.args.output_attention else \
                                self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model in non_transformer:
                        outputs = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0] if self.args.output_attention else \
                            self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, 0:3]
                batch_y = batch_y[:, -self.args.pred_len:, 0:3].to(self.device)
                loss = criterion(outputs.detach().cpu(), batch_y.detach().cpu())
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            score = []
            pred = []
            true = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model in non_transformer:
                            outputs = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0] if self.args.output_attention else \
                                self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        outputs = outputs[:, -self.args.pred_len:, :]
                        batch_y = batch_y[:, -self.args.pred_len:, 0:3].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.model in non_transformer:
                        outputs = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0] if self.args.output_attention else \
                            self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    # f_dim = -1 if self.args.features == 'MS' else 0
                    # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, 0:3].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                pred.append(outputs.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            pred = train_data.temp_inverse(np.reshape(pred, (-1, 3)))
            true = train_data.temp_inverse(np.reshape(true, (-1, 3)))
            score = np.sum(np.abs(pred-true), axis=1)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        threshold = np.percentile(score, 99)
        print("threshold: {0}".format(threshold))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        data_X, data_Y, time_stamp = test_data.get_data()

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []
        gt = []
        pd = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            last = datetime.strptime(time_stamp[0], '%Y/%m/%d %H:%M')
            pic = 0
            for idx, data in enumerate(data_X):
                now = datetime.strptime(time_stamp[idx], '%Y/%m/%d %H:%M')
                if (now - last) > timedelta(minutes=5):

                    visual(np.array(gt)[:, -2], np.array(pd)[:, -2], os.path.join(folder_path, str(pic) + '.pdf'))
                    pic = pic + 1
                    gt = []
                    pd = []

                _, m = data.shape
                data = torch.Tensor(data.reshape(1, self.args.seq_len, m)).to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                else:
                    outputs = self.model(data)
                
                outputs = outputs[:, -self.args.pred_len:, :]
    
                pred = outputs.detach().cpu().numpy()[0]
        
                true = data_Y[idx]

                pred = test_data.temp_inverse(pred[:, 0:3])
                true = test_data.temp_inverse(true[:, 0:3])
                # pred = pred[:, 0:3]
                # true = true[:, 0:3]

                pd.append(pred[0])
                gt.append(true[0])
                
                preds.append(pred[0])
                trues.append(true[0])

                last = now

        preds = np.array(preds)
        trues = np.array(trues)

        df = pandas.concat([pandas.DataFrame(trues), pandas.DataFrame(preds)], axis=1)
        df.to_csv(folder_path+"testData.csv", index=False)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        rmse, mae, r_squared, mape = metric(preds, trues)
        print('rmse:{:.4f}, mae:{:.4f}, R2:{:.4f}, mape:{:.4f}'.format(rmse, mae, r_squared, mape))
        with open('result.txt', "a") as file:
            file.write(setting+' rmse:{:.4f}, mae:{:.4f}, R2:{:.4f}, mape:{:.4f}\n'.format(rmse, mae, r_squared, mape))
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        data_X, data_Y, time_stamp = pred_data.get_data()

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []
        gt = []
        pd = []
        folder_path = './predict_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():

            last = datetime.strptime(time_stamp[0], '%Y/%m/%d %H:%M')
            pic = 0
            for idx, data in enumerate(data_X):
                now = datetime.strptime(time_stamp[idx], '%Y/%m/%d %H:%M')
                if (now - last) > timedelta(minutes=5):
                    visual(np.array(gt)[:, -2], np.array(pd)[:, -2], os.path.join(folder_path, str(pic) + '.pdf'))
                    pic = pic + 1
                    gt = []
                    pd = []

                _, m = data.shape
                data = torch.Tensor(data.reshape(1, self.args.seq_len, m)).to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                else:
                    outputs = self.model(data)

                outputs = outputs[:, -self.args.pred_len:, :]

                pred = outputs.detach().cpu().numpy()[0]

                true = data_Y[idx]

                pred = pred_data.temp_inverse(pred[:, 0:3])
                true = pred_data.temp_inverse(true[:, 0:3])
                # pred = pred[:, 0:3]
                # true = true[:, 0:3]

                pd.append(pred[0])
                gt.append(true[0])

                preds.append(pred[0])
                trues.append(true[0])

                last = now

        preds = np.array(preds)
        trues = np.array(trues)

        df = pandas.concat([pandas.DataFrame(trues), pandas.DataFrame(preds)], axis=1)
        df.to_csv(folder_path+'preData.csv', index=False)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        rmse, mae, r_squared, mape = metric(preds, trues)
        print('rmse:{:.4f}, mae:{:.4f}, R2:{:.4f}, mape:{:.4f}'.format(rmse, mae, r_squared, mape))
        with open('result.txt', "a") as file:
            file.write('predict ' + setting+ ' rmse:{:.4f}, mae:{:.4f}, R2:{:.4f}, mape:{:.4f}\n'.format(rmse, mae, r_squared, mape))
        return
