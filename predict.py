#!/usr/bin/python
# -*- coding: Shift_JIS -*-
#
#�@�w�K�ς݂̃��f����ǂݍ��ށB
#  �e�X�g�f�[�^��ǂݍ��ށB
#  �T���v���S��������炵�Ȃ���A�\���l�̃��X�g�����B


import os
import tensorflow as tf
import pandas as pd
import numpy as np
from zipfile import ZipFile

######################################################################
# list 15


def rnn_predict(input_dataset):
    # �W����
    previous = TimeSeriesDataSet(input_dataset).tail(SERIES_LENGTH).standardize(mean=train_mean, std=train_std)
    # �\���Ώۂ̎���
    predict_time = previous.times[-1] + np.timedelta64(1, 'h')

    # �\��
    batch_x = previous.as_array()
    predict_data = prediction.eval({x: batch_x})

    # ���ʂ̃f�[�^�t���[�����쐬
    df_standardized = pd.DataFrame(predict_data, columns=input_dataset.columns, index=[predict_time])
    # �W�����̋t����
    return train_mean + train_std * df_standardized

######################################################################
# �N���X��`
class TimeSeriesDataSet:

    def __init__(self, dataframe):
        self.feature_count = len(dataframe.columns)
        self.series_length = len(dataframe)
        self.series_data = dataframe.astype('float32')

    def __getitem__(self, n):
        return TimeSeriesDataSet(self.series_data[n])

    def __len__(self):
        return len(self.series_data)

    @property
    def times(self):
        return self.series_data.index

    def next_batch(self, length, batch_size):
        """
        �A������length���Ԃ̃f�[�^�����1���Ԃ̌덷����p�f�[�^���擾����B
        �Ō��1���Ԃ͍ŏI�o�̓f�[�^�B
        """
        max_start_index = len(self) - length
        design_matrix = []
        expectation = []
        while len(design_matrix) < batch_size:
            start_index = np.random.choice(max_start_index)
            end_index = start_index + length + 1
            values = self.series_data[start_index:end_index]
            if (values.count() == length + 1).all():  # �؂�o�����f�[�^���Ɍ����l���Ȃ�
                train_data = values[:-1]
                true_value = values[-1:]
                design_matrix.append(train_data.as_matrix())
                expectation.append(np.reshape(true_value.as_matrix(), [self.feature_count]))

        return np.stack(design_matrix), np.stack(expectation)

    def append(self, data_point):
        dataframe = pd.DataFrame(data_point, columns=self.series_data.columns)
        self.series_data = self.series_data.append(dataframe)

    def tail(self, n):
        return TimeSeriesDataSet(self.series_data.tail(n))

    def as_array(self):
        return np.stack([self.series_data.as_matrix()])

    def mean(self):
        return self.series_data.mean()

    def std(self):
        return self.series_data.std()

    def standardize(self, mean=None, std=None):
        if mean is None:
            mean = self.mean()
        if std is None:
            std = self.std()
        return TimeSeriesDataSet((self.series_data - mean) / std)


#######################################################################
# list 3
if os.name == 'nt':
    zip_file = "UCI_data\\AirQualityUCI.zip"
else:
    zip_file = "UCI_data/AirQualityUCI.zip"

with ZipFile(zip_file) as z:
    with z.open('AirQualityUCI.xlsx') as f:
        air_quality = pd.read_excel(
            f,
            index_col=0, parse_dates={'DateTime': [0, 1]}, #1
            na_values=[-200.0],                            #2
            convert_float=False                            #3
    )

#############################################################
# list 7
# �s�v��̏���
target_columns = ['T', 'AH', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)']
air_quality = air_quality[target_columns]


# hogehoge
dataset = TimeSeriesDataSet(air_quality)
train_dataset = dataset[dataset.times.year < 2005]
test_dataset = dataset[dataset.times.year >= 2005]


# �p�����[�^�[
# �w�K���Ԓ�
SERIES_LENGTH = 72
# �����ʐ�
FEATURE_COUNT = dataset.feature_count

# ���́iplaceholder���\�b�h�̈����́A�f�[�^�^�A�e���\���̃T�C�Y�j
# �P���f�[�^
x = tf.placeholder(tf.float32, [None, SERIES_LENGTH, FEATURE_COUNT])
# ���t�f�[�^
y = tf.placeholder(tf.float32, [None, FEATURE_COUNT])

# �W����
train_mean = train_dataset.mean()
train_std = train_dataset.std()

#######################################################################
# list 11
# RNN�Z���̍쐬
cell = tf.nn.rnn_cell.BasicRNNCell(20)
initial_state = cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
outputs, last_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)

# �S����
# �d��
w = tf.Variable(tf.zeros([20, FEATURE_COUNT]))
# �o�C�A�X
b = tf.Variable([0.1] * FEATURE_COUNT)
# �ŏI�o�́i�\���j
prediction = tf.matmul(last_state, w) + b

##
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# restore
saver = tf.train.Saver()
cwd = os.getcwd()
if os.name == 'nt':
    saver.restore(sess, cwd + "\\model.ckpt")
else:
    saver.restore(sess, cwd + "/model.ckpt")

predict_air_quality = pd.DataFrame([], columns=air_quality.columns)
for current_time in test_dataset.times:
    predict_result = rnn_predict(air_quality[air_quality.index < current_time])
    predict_air_quality = predict_air_quality.append(predict_result)

print(predict_air_quality)





