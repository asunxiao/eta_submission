# -*- coding: utf-8 -*-
import regex as re
import numpy as np
import pandas as pd
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from sklearn import preprocessing
from sklearn import ensemble
from sklearn import model_selection

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

import joblib


# x,y是否相同
def is_same(x, y):
    if x == y:
        return 1
    else:
        return 0


# x是否是一个IP
def is_ip(x):
    p = re.compile('^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$')
    if p.match(str(x)):
        return 1
    else:
        return 0


# x是否只包含CN
def only_CN(x):
    p = re.compile('^CN=(.*?)/')
    if p.match(str(x)):
        return 1
    else:
        return 0


# x是否是一个.com域
def is_com(x):
    p = re.compile('.*\.com')
    if p.match(str(x)):
        return 1
    else:
        return 0


# 用于计算tlsSubject和tlsSni中的元素个数
def count_elements(x):
    return str(x).count('=')


# 用于计算tlsSubject和tlsSni中的信息的长度
def count_length(x):
    return len(str(x))


# 用于计算字符串的信息熵
import math


def get_entropy(text):
    if text == 'NULL':
        return 0
    else:
        h = 0.0
        sum = 0
        letter = [0] * 100
        text = text.lower()
        for i in range(len(text)):
            if text[i].encode().isalpha():
                letter[ord(text[i]) - ord('a')] += 1
                sum += 1
        if sum == 0:
            return 0
        else:
            for i in range(26):
                p = 1.0 * letter[i] / sum
                if p > 0:
                    h += -(p * math.log(p, 2))
            return h


# 特征工程
def makefeature(data):
    # 对tlsSni的缺失值进行填充
    data['tlsSni'].fillna(data['tlsSni'].mode().iloc[0], inplace=True)

    # 对bytesOut、bytesIn、pktsOut、pktsIn进行Binning处理，分成0、1、2、3、4五个范围等级
    data.loc[data['bytesOut'] <= 2587.8, 'bytesOut_bin_id'] = '0'
    data.loc[(data['bytesOut'] > 2587.8) & (data['bytesOut'] <= 5235), 'bytesOut_bin_id'] = '1'
    data.loc[(data['bytesOut'] > 5235) & (data['bytesOut'] <= 7891), 'bytesOut_bin_id'] = '2'
    data.loc[(data['bytesOut'] > 7891) & (data['bytesOut'] <= 15798), 'bytesOut_bin_id'] = '3'
    data.loc[data['bytesOut'] > 15798, 'bytesOut_bin_id'] = '4'
    bytesOut_dummies = pd.get_dummies(data['bytesOut_bin_id']).rename(columns=lambda x: 'bytesOut_' + str(x))
    data = data.join(bytesOut_dummies)
    data.drop(['bytesOut_bin_id'], axis=1, inplace=True)

    data.loc[data['bytesIn'] <= 1632, 'bytesIn_bin_id'] = '0'
    data.loc[(data['bytesIn'] > 1632) & (data['bytesIn'] <= 2388), 'bytesIn_bin_id'] = '1'
    data.loc[(data['bytesIn'] > 2388) & (data['bytesIn'] <= 3590.4), 'bytesIn_bin_id'] = '2'
    data.loc[(data['bytesIn'] > 3590.4) & (data['bytesIn'] <= 7051.2), 'bytesIn_bin_id'] = '3'
    data.loc[data['bytesIn'] > 7051.2, 'bytesIn_bin_id'] = '4'
    bytesIn_dummies = pd.get_dummies(data['bytesIn_bin_id']).rename(columns=lambda x: 'bytesIn_' + str(x))
    data = data.join(bytesIn_dummies)
    data.drop(['bytesIn_bin_id'], axis=1, inplace=True)

    data.loc[data['pktsOut'] <= 10, 'pktsOut_bin_id'] = '0'
    data.loc[(data['pktsOut'] > 10) & (data['pktsOut'] <= 13), 'pktsOut_bin_id'] = '1'
    data.loc[(data['pktsOut'] > 13) & (data['pktsOut'] <= 18), 'pktsOut_bin_id'] = '2'
    data.loc[(data['pktsOut'] > 18) & (data['pktsOut'] <= 31), 'pktsOut_bin_id'] = '3'
    data.loc[data['pktsOut'] > 31, 'pktsOut_bin_id'] = '4'
    pktsOut_dummies = pd.get_dummies(data['pktsOut_bin_id']).rename(columns=lambda x: 'pktsOut_' + str(x))
    data = data.join(pktsOut_dummies)
    data.drop(['pktsOut_bin_id'], axis=1, inplace=True)

    data.loc[data['pktsIn'] <= 10, 'pktsIn_bin_id'] = '0'
    data.loc[(data['pktsIn'] > 10) & (data['pktsIn'] <= 15), 'pktsIn_bin_id'] = '1'
    data.loc[(data['pktsIn'] > 15) & (data['pktsIn'] <= 22), 'pktsIn_bin_id'] = '2'
    data.loc[(data['pktsIn'] > 22) & (data['pktsIn'] <= 39), 'pktsIn_bin_id'] = '3'
    data.loc[data['pktsIn'] > 39, 'pktsIn_bin_id'] = '4'
    pktsIn_dummies = pd.get_dummies(data['pktsIn_bin_id']).rename(columns=lambda x: 'pktsIn_' + str(x))
    data = data.join(pktsIn_dummies)
    data.drop(['pktsIn_bin_id'], axis=1, inplace=True)

    # dummy处理tlsVersion属性，这里只对四个主要的版本进行提取处理
    data['tlsVersion_TLS 1.2'] = data['tlsVersion'].apply(lambda x: 1 if x == 'TLS 1.2' else 0)
    data['tlsVersion_TLS 1.3'] = data['tlsVersion'].apply(lambda x: 1 if x == 'TLS 1.3' else 0)
    data['tlsVersion_TLSv1'] = data['tlsVersion'].apply(lambda x: 1 if x == 'TLSv1' else 0)
    data['tlsVersion_UNDETERMINED'] = data['tlsVersion'].apply(lambda x: 1 if x == 'UNDETERMINED' else 0)

    # tlsSubject和tlsIssuerDn的缺失值标记为NULL
    data.loc[data.tlsSubject.isnull(), 'tlsSubject'] = 'NULL'
    data.loc[data.tlsIssuerDn.isnull(), 'tlsIssuerDn'] = 'NULL'

    # Has_tls表示tlsSubject是否为空（NULL）
    data['Has_tls'] = data['tlsSubject'].apply(lambda x: 0 if x == 'NULL' else 1)

    # Subject_eq_Issuer表示tlsSubject与tlsIssuerDn是否相同
    data['Subject_eq_Issuer'] = data.apply(lambda x: is_same(x['tlsSubject'], x['tlsIssuerDn']), axis=1)

    # tlsSubject_temp辅助提取出tlsSubject中的C、O、OU、ST、L、CN、CO属性
    data['tlsSubject_temp'] = data['tlsSubject'] + '/'
    data['tlsSubject_C'] = data['tlsSubject_temp'].str.extract(r'C=(.*?)[,/]', expand=True)
    data['tlsSubject_O'] = data['tlsSubject_temp'].str.extract(r'O=(.*?)[,/]', expand=True)
    data['tlsSubject_OU'] = data['tlsSubject_temp'].str.extract(r'OU=(.*?)[,/]', expand=True)
    data['tlsSubject_ST'] = data['tlsSubject_temp'].str.extract(r'ST=(.*?)[,/]', expand=True)
    data['tlsSubject_L'] = data['tlsSubject_temp'].str.extract(r'L=(.*?)[,/]', expand=True)
    data['tlsSubject_CN'] = data['tlsSubject_temp'].str.extract(r'CN=(.*?)[,/]', expand=True)
    data['tlsSubject_CO'] = data['tlsSubject_temp'].str.extract(r'CO=(.*?)[,/]', expand=True)

    # tlsSubject中的C、O、OU、ST、L、CN、CO属性空值标记为NULL
    data.loc[data.tlsSubject_C.isnull(), 'tlsSubject_C'] = 'NULL'
    data.loc[data.tlsSubject_O.isnull(), 'tlsSubject_O'] = 'NULL'
    data.loc[data.tlsSubject_OU.isnull(), 'tlsSubject_OU'] = 'NULL'
    data.loc[data.tlsSubject_ST.isnull(), 'tlsSubject_ST'] = 'NULL'
    data.loc[data.tlsSubject_L.isnull(), 'tlsSubject_L'] = 'NULL'
    data.loc[data.tlsSubject_CN.isnull(), 'tlsSubject_CN'] = 'NULL'
    data.loc[data.tlsSubject_CO.isnull(), 'tlsSubject_CO'] = 'NULL'

    # tlsIssuerDn_temp辅助提取出tlsIssuerIn中的C、O、OU、ST、L、CN、CO属性
    data['tlsIssuerDn_temp'] = data['tlsIssuerDn'] + '/'
    data['tlsIssuerDn_C'] = data['tlsIssuerDn_temp'].str.extract(r'C=(.*?)[,/]', expand=True)
    data['tlsIssuerDn_O'] = data['tlsIssuerDn_temp'].str.extract(r'O=(.*?)[,/]', expand=True)
    data['tlsIssuerDn_OU'] = data['tlsIssuerDn_temp'].str.extract(r'OU=(.*?)[,/]', expand=True)
    data['tlsIssuerDn_ST'] = data['tlsIssuerDn_temp'].str.extract(r'ST=(.*?)[,/]', expand=True)
    data['tlsIssuerDn_L'] = data['tlsIssuerDn_temp'].str.extract(r'L=(.*?)[,/]', expand=True)
    data['tlsIssuerDn_CN'] = data['tlsIssuerDn_temp'].str.extract(r'CN=(.*?)[,/]', expand=True)
    data['tlsIssuerDn_CO'] = data['tlsIssuerDn_temp'].str.extract(r'CO=(.*?)[,/]', expand=True)

    # tlsIssuerIn中的C、O、OU、ST、L、CN、CO属性的空值标记为NULL
    data.loc[data.tlsIssuerDn_C.isnull(), 'tlsIssuerDn_C'] = 'NULL'
    data.loc[data.tlsIssuerDn_O.isnull(), 'tlsIssuerDn_O'] = 'NULL'
    data.loc[data.tlsIssuerDn_OU.isnull(), 'tlsIssuerDn_OU'] = 'NULL'
    data.loc[data.tlsIssuerDn_ST.isnull(), 'tlsIssuerDn_ST'] = 'NULL'
    data.loc[data.tlsIssuerDn_L.isnull(), 'tlsIssuerDn_L'] = 'NULL'
    data.loc[data.tlsIssuerDn_CN.isnull(), 'tlsIssuerDn_CN'] = 'NULL'
    data.loc[data.tlsIssuerDn_CO.isnull(), 'tlsIssuerDn_CO'] = 'NULL'

    # （1）	Subject_onlyCN：表示Subject是否只有CN。
    # （2）	Subject_is_com：表示Subject是否是“.com”域。
    # （3）	Issuer_is_com：表示Issuer是否是“.com”域。
    # （4）	Subject_eq_Issuer：表示是否Subject Principal=Issuer Principal.
    # （5）	SubjectElements：表示Subject中元素的个数。
    # （6）	IssuerElements：表示Issuer中元素的个数。
    # （7）	SubjectLength：表示Subject中字符串的长度。
    # （8）	IssuerLength：表示Issuer中字符串的长度。
    # （9）	SubjectCommonName：表示Subject中的CN字符串的熵值。
    # （10）	SubjectCommonNameIP	：表示Subject中CN是否是一个IP。
    data['SubjectCommonNameIP'] = data.apply(lambda x: is_ip(x['tlsSubject_CN']), axis=1)
    data['SubjectCommonName'] = data.apply(lambda x: get_entropy(x['tlsSubject_CN']), axis=1)
    data['Subject_onlyCN'] = data.apply(lambda x: only_CN(x['tlsSubject_temp']), axis=1)
    data['Subject_is_com'] = data.apply(lambda x: is_com(x['tlsSubject_temp']), axis=1)
    data['SubjectElements'] = data.apply(lambda x: count_elements(x['tlsSubject_temp']), axis=1)
    data['SubjectLength'] = data.apply(lambda x: count_length(x['tlsSubject']), axis=1)

    data['SubjectHasOrganization'] = data['tlsSubject_O'].apply(lambda x: 0 if x == 'NULL' else 1)
    data['SubjectHasState'] = data['tlsSubject_ST'].apply(lambda x: 0 if x == 'NULL' else 1)
    data['SubjectHasLocation'] = data['tlsSubject_L'].apply(lambda x: 0 if x == 'NULL' else 1)
    data['SubjectHasCompany'] = data['tlsSubject_CO'].apply(lambda x: 0 if x == 'NULL' else 1)
    data['SubjectHasCommonName'] = data['tlsSubject_CN'].apply(lambda x: 0 if x == 'NULL' else 1)

    data['Issuer_is_com'] = data.apply(lambda x: is_com(x['tlsIssuerDn_temp']), axis=1)
    data['IssuerElements'] = data.apply(lambda x: count_elements(x['tlsIssuerDn_temp']), axis=1)
    data['IssuerLength'] = data.apply(lambda x: count_length(x['tlsIssuerDn']), axis=1)

    data['IssuerHasOrganization'] = data['tlsIssuerDn_O'].apply(lambda x: 0 if x == 'NULL' else 1)
    data['IssuerHasState'] = data['tlsIssuerDn_ST'].apply(lambda x: 0 if x == 'NULL' else 1)
    data['IssuerHasLocation'] = data['tlsIssuerDn_L'].apply(lambda x: 0 if x == 'NULL' else 1)
    data['IssuerHasCompany'] = data['tlsIssuerDn_CO'].apply(lambda x: 0 if x == 'NULL' else 1)
    data['IssuerHasCommonName'] = data['tlsIssuerDn_CN'].apply(lambda x: 0 if x == 'NULL' else 1)

    # 删除tlsSubject_temp和tlsIssuerDn_temp两列
    data.drop(['tlsSubject_temp'], axis=1, inplace=True)
    data.drop(['tlsIssuerDn_temp'], axis=1, inplace=True)

    # 对'srcPort','destPort','bytesOut','bytesIn','pktsIn','pktsOut','SubjectElements','SubjectLength','IssuerElements','IssuerLength','SubjectCommonName'进行scaling处理
    scale_data = preprocessing.StandardScaler().fit(
        data[['srcPort', 'destPort', 'bytesOut', 'bytesIn', 'pktsIn', 'pktsOut',
              'SubjectElements', 'SubjectLength', 'IssuerElements', 'IssuerLength', 'SubjectCommonName']])
    data[['srcPort', 'destPort', 'bytesOut', 'bytesIn', 'pktsIn', 'pktsOut', 'SubjectElements', 'SubjectLength',
          'IssuerElements', 'IssuerLength', 'SubjectCommonName']] = scale_data.transform(data[['srcPort', 'destPort',
                                                                                               'bytesOut', 'bytesIn',
                                                                                               'pktsIn', 'pktsOut',
                                                                                               'SubjectElements',
                                                                                               'SubjectLength',
                                                                                               'IssuerElements',
                                                                                               'IssuerLength',
                                                                                               'SubjectCommonName']])

    # 过滤提取出需要的特征
    return_data = data.filter(
        regex='label|srcPort|destPort|bytesOut_|bytesIn_|pktsOut_|pktsIn_|Has_tls|^Subject.*|^Issuer.*|tlsVersion_')

    return return_data


def test_func(test_path, save_path):
    # 请填写测试代码
    test = pd.read_csv(test_path)
    eventId = test['eventId']
    test_data = makefeature(test)
    print('[TEST SIZE]: ', test_data.shape)
    test_data_X = test_data
    # 选手不得改变格式，测试代码跑不通分数以零算

    # #####选手填写测试集处理逻辑,在指定文件夹下生成可提交的csv文件
    K = 7
    clfs = joblib.load('clfs.pkl')
    test_predict_feature = np.zeros((test_data_X.shape[0], K))
    test_predict_feature_test = np.zeros((test_data_X.shape[0]))
    # the first layer in Blending
    for j, clf in enumerate(clfs):
        test_predict_feature[:, j] = clf.predict(test_data_X)

    test_predict_feature_test = test_predict_feature.mean(axis=1)
    # the second layer in Blending
    gbm = joblib.load('model.pkl')
    predictions = gbm.predict(test_predict_feature_test.reshape(-1, 1))

    submission = pd.DataFrame({'eventId': eventId, 'label': predictions})
    print(submission.groupby('label')['label'].count())
    submission.to_csv(save_path + 'hn-2020_eta_submission_1030.csv', index=False, encoding='utf-8', sep=',')


if __name__ == '__main__':
    test_path = '../data/test_1.csv'
    sava_path = '../result/'
    test_func(test_path, sava_path)
