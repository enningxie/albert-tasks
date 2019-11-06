# coding=utf-8
import pandas as pd
from keras.layers import Lambda, Dense
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from bert4keras.backend import set_gelu
from bert4keras.bert import build_bert_model
from bert4keras.utils import Tokenizer, load_vocab
from bert4keras.train import PiecewiseLinearLearningRate
# 用load_model加载整个模型
from bert4keras.layers import *
from collections import defaultdict
import numpy as np
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
set_gelu('tanh')  # 切换gelu版本
# albert_tiny_zh_google/ | albert_small_zh_google
MODEL_PATH_PREFIX = '/Data/public/Bert/albert_small_zh_google'


def select_albert():
    # Evaluate
    tmp_models_config = [
        ['full', 'albert_small_g_01.h5', 'LCQMC'],
        ['part', 'albert_small_g_02.h5', 'LCQMC'],
        # ['full', 'albert_tiny_g_01.h5', 'LCQMC'],
        # ['part', 'albert_tiny_g_02.h5', 'LCQMC']
        # ['full', 'test_albert_tiny_489k_04.h5', 'BQ'],
        # ['part', 'test_albert_tiny_489k_05.h5', 'sent_pair'],
        # ['full', 'test_albert_tiny_489k_06.h5', 'sent_pair']
    ]
    # models_config = [
    #     ['part', '01', 'LCQMC'],
    #     ['full', '02', 'LCQMC'],
    #     ['part', '03', 'BQ'],
    #     ['full', '04', 'BQ'],
    #     ['part', '05', 'sent_pair'],
    #     ['full', '06', 'sent_pair'],
    #     ['full', '07', 'LCQMC'],
    #     ['full', '08', 'LCQMC'],
    #     ['full', '09', 'LCQMC'],
    #     ['part', '10', 'sent_pair']
    # ]
    durations = []
    accs = []
    for tmp_model_config in tmp_models_config:
        print('--------albert_tiny_{}----------'.format(tmp_model_config[1]))
        duration, acc = evaluate_albert_tiny('../data/test_data(1).csv', *tmp_model_config)
        durations.append(duration)
        accs.append(acc)

    for model_index, (tmp_duration, tmp_acc) in enumerate(zip(durations, accs)):
        print('--------albert_tiny_{}----------'.format(model_index + 1))
        print('cost time: {}'.format(tmp_duration))
        print('acc: {}'.format(tmp_acc))


def evaluate_albert_tiny(csv_path, mode_, model_name, dataset_name):
    raw_df = pd.read_csv(csv_path)
    sent1_list = list(raw_df.sentence1)
    sent2_list = list(raw_df.sentence2)
    y_label = np.expand_dims(np.asarray(raw_df.label), -1)
    model_albert = Albert(mode='inference',
                          mode_=mode_,
                          model_name=model_name,
                          dataset_name=dataset_name)
    start_time = time.time()
    y_pred = model_albert.predict(sent1_list, sent2_list)
    duration = time.time() - start_time
    return duration, exceed_threshold(y_pred, y_label)


def exceed_threshold(y_pred, y_label, threshold=0.7):
    y_pred_np = np.asarray(y_pred)
    y_pred_len = len(y_pred_np)
    y_pred_label = (y_pred_np > threshold).astype('int')
    accuracy = np.sum(y_pred_label == y_label) / y_pred_len
    return accuracy


def test_albert():
    albert_model = Albert(model_name='albert_small_g_01.h5')
    while True:
        sent1 = input('sent1: ')
        sent2 = input('sent2: ')
        print(albert_model.predict([sent1], [sent2]).item())


class Albert(object):
    def __init__(self, mode='inference', mode_='full', model_name=None, dataset_name='LCQMC'):
        self.maxlen = 32
        # albert_config_tiny_g.json | albert_config_small_google
        self.albert_config_path = os.path.join(MODEL_PATH_PREFIX, 'albert_config_small_google.json')
        self.albert_checkpoint_path = os.path.join(MODEL_PATH_PREFIX, 'albert_model.ckpt')
        self.albert_dict_path = os.path.join(MODEL_PATH_PREFIX, 'vocab.txt')
        self.train_data_path = '../data/train_{}.csv'.format(dataset_name)
        self.dev_data_path = '../data/dev_{}.csv'.format(dataset_name)
        self.test_data_path = '../data/test_{}.csv'.format(dataset_name)
        # albert_tiny_250k.h5 挺好的
        # self.restore_model_path = 'saved_models/test_albert_tiny_{}.h5'.format(model_name)
        self.restore_model_path = '../saved_models/{}'.format(model_name)

        # albert
        self.albert_process_data(mode_)
        if mode == 'train':
            self.model = self._get_model()
            self.train()
        elif mode == 'inference':
            self._init_model()

    # todo keep words 工业场景下需要remove
    def albert_process_data(self, mode='part'):
        _token_dict = load_vocab(self.albert_dict_path)  # 读取字典
        # 只取涉及数据集中出现的字
        if mode == 'part':
            train_df = pd.read_csv(self.train_data_path, names=['seq1', 'seq2', 'label'])
            valid_df = pd.read_csv(self.dev_data_path, names=['seq1', 'seq2', 'label'])
            test_df = pd.read_csv(self.test_data_path, names=['seq1', 'seq2', 'label'])
            # total data
            tmp_df = pd.concat([train_df, valid_df, test_df])
            chars = defaultdict(int)
            for _, tmp_row in tmp_df.iterrows():
                for tmp_char in tmp_row.seq1:
                    chars[tmp_char] += 1
                for tmp_char in tmp_row.seq2:
                    chars[tmp_char] += 1
            # 过滤低频字
            chars = {i: j for i, j in chars.items() if j >= 4}
            self.token_dict, self.keep_words = {}, []  # keep_words是在bert中保留的字表
            # 保留特殊字符
            for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
                self.token_dict[c] = len(self.token_dict)
                self.keep_words.append(_token_dict[c])
            # 字典只保留数据中出现的高频字
            for c in chars:
                if c in _token_dict and c not in self.token_dict:
                    self.token_dict[c] = len(self.token_dict)
                    self.keep_words.append(_token_dict[c])
        elif mode == 'full':
            self.token_dict, self.keep_words = _token_dict, []
            for k in self.token_dict:
                self.keep_words.append(self.token_dict[k])
        self.tokenizer = Tokenizer(self.token_dict)  # 建立分词器

    # data pre-processing operation
    def _data_preprocessing(self, sentence1, sentence2):
        X1, X2 = [], []
        for tmp_sent1, tmp_sent2 in zip(sentence1, sentence2):
            x1, x2 = self.tokenizer.encode(first_text=tmp_sent1[:self.maxlen], second_text=tmp_sent2[:self.maxlen])
            X1.append(x1)
            X2.append(x2)
        X1 = self._seq_padding(X1)
        X2 = self._seq_padding(X2)
        # X1 = pad_sequences(X1, maxlen=67, padding='post', truncating='post')
        # X2 = pad_sequences(X2, maxlen=67, padding='post', truncating='post')
        return X1, X2

    def _seq_padding(self, X, padding=0):
        L = [len(x) for x in X]
        ML = max(L)
        padded_sent = np.array([
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])
        return padded_sent

    # prepare data for training
    def _prepare_data(self, data_path):
        data = pd.read_csv(data_path)
        sent_1 = data['sentence1'].values
        sent_2 = data['sentence2'].values
        label = data['label'].values
        X1_pad, X2_pad = self._data_preprocessing(sent_1, sent_2)
        # X1 = np.vstack((X1_pad, X2_pad))
        # X2 = np.vstack((X2_pad, X1_pad))
        # y_train = np.hstack((label, label))
        return X1_pad, X2_pad, label

    # albert for Semantic matching, model architecture
    def _get_model(self):
        model = build_bert_model(
            self.albert_config_path,
            self.albert_checkpoint_path,
            keep_words=self.keep_words,  # 只保留keep_words中的字，精简原字表
            albert=True
        )
        output = Lambda(lambda x: x[:, 0])(model.output)
        output = Dense(1, activation='sigmoid')(output)
        model = Model(model.input, output)
        return model

    # model training operation
    def train(self):
        # train_data
        train_x1, train_x2, train_label = self._prepare_data(self.train_data_path)
        # dev_data
        dev_x1, dev_x2, dev_label = self._prepare_data(self.dev_data_path)
        checkpoint = ModelCheckpoint(self.restore_model_path, monitor='val_accuracy', verbose=0,
                                     save_best_only=True, save_weights_only=False)
        early_stop = EarlyStopping(monitor='val_accuracy', patience=3, verbose=0, mode='auto', baseline=None,
                                   restore_best_weights=True)
        self.model.compile(
            loss='binary_crossentropy',
            # optimizer=Adam(1e-4),  # 用足够小的学习率
            optimizer=PiecewiseLinearLearningRate(Adam(1e-4), {1000: 1, 2000: 0.1}),
            metrics=['accuracy']
        )
        self.model.summary()
        self.model.fit(x=[train_x1, train_x2],
                       y=train_label,
                       batch_size=64,
                       epochs=10,
                       verbose=1,
                       callbacks=[checkpoint, early_stop],
                       validation_data=([dev_x1, dev_x2], dev_label))

    # model predict operation
    def predict(self, sentence1, sentence2):
        X1, X2 = self._data_preprocessing(sentence1, sentence2)
        y_pred = self.model.predict([X1, X2], batch_size=1024)
        return y_pred

    def test(self):
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(1e-4),  # 用足够小的学习率
            metrics=['accuracy']
        )
        # test_data
        test_x1, test_x2, test_label = self._prepare_data(self.dev_data_path)
        test_loss, test_acc = self.model.evaluate(x=[test_x1, test_x2], y=test_label)
        print('test loss: {}'.format(test_loss))
        print('test acc: {}'.format(test_acc))

    def _init_model(self):
        self.model = load_model(self.restore_model_path)
        sentence1 = '干嘛呢'
        sentence2 = '你是机器人'
        print('model albert loaded succeed. ({})'.format(self.predict([sentence1], [sentence2]).item()))

    #################


if __name__ == '__main__':
    # tmp_albert = Albert(mode='train', mode_='full', model_name='albert_small_g_01.h5')
    # select_albert()
    test_albert()
