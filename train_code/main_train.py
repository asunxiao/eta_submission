# -*- coding: utf-8 -*-
import regex as re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

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


# 提取top n特征
def get_top_n_features(train_data_X, train_data_Y, top_n_features):
    # randomforest
    rf_est = RandomForestClassifier(random_state=0)
    rf_param_grid = {'n_estimators': [500, 400, 300], 'min_samples_split': [2, 3, 4], 'max_depth': [20, 30, 40]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    rf_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score:' + str(rf_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(train_data_X),
                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Features from RF Classifier')
    print(str(features_top_n_rf[:]))

    # AdaBoost
    ada_est = AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(ada_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(train_data_X),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Features from Ada Classifier:')
    print(str(features_top_n_ada[:]))

    # ExtraTree
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Best ET Params:' + str(et_grid.best_params_))
    print('Top N Features Best ET Score:' + str(et_grid.best_score_))
    print('Top N Features ET Train Score:' + str(et_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_et = pd.DataFrame({'feature': list(train_data_X),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:]))

    # GradientBoosting
    gb_est = GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
    gb_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' + str(gb_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(train_data_X),
                                          'importance': gb_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:]))

    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)
    dt_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Bset DT Params:' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score:' + str(dt_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_dt = pd.DataFrame({'feature': list(train_data_X),
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:]))

    # merge the three models
    features_top_n = pd.concat(
        [features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt],
        ignore_index=True).drop_duplicates()
    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et,
                                     feature_imp_sorted_gb, feature_imp_sorted_dt], ignore_index=True)

    return features_top_n, features_importance


def train_func(train_path):
    # 请填写训练代码
    train = pd.read_csv(train_path)
    print("Training is start")

    train_data = makefeature(train)

    print('[TRAIN SIZE]: ', train_data.shape)
    print('[TRAIN INFO]: ')
    print(train_data.info())
    # TRAIN DATA PREPARE
    train_data_Y = train_data['label']
    train_data_X = train_data.drop(['label'], axis=1)
    print('[TRAIN FEATURE SIZE]: ', train_data_X.shape)
    print('[TRAIN LABEL DISTRIBUTION]: ')
    print(train_data_Y.value_counts())
    ##################################################################
    # 要提取的特征的数目
    feature_to_pick = 30
    # feature_top_n,feature_importance = get_top_n_features(train_data_X,train_data_Y,feature_to_pick)

    # 构建不同的基学习器，这里我们使用了RandomForest、AdaBoost、ExtraTrees、GBDT、DecisionTree、KNN、SVM 七个基学习器
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    rf = RandomForestClassifier(n_estimators=400, warm_start=True, max_features='sqrt', max_depth=30,
                                min_samples_split=2, min_samples_leaf=2, n_jobs=-1, verbose=0)
    ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)
    et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)
    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2,
                                    max_depth=5, verbose=0)
    dt = DecisionTreeClassifier(max_depth=8)
    knn = KNeighborsClassifier(n_neighbors=2)
    svm = SVC(kernel='linear', C=0.025)
    ####################################################################
    K = 7
    clfs = [rf, ada, et, gb, dt, knn, svm]
    train_data1, train_data2, train_label1, train_label2 = train_test_split(train_data_X, train_data_Y, test_size=0.2,
                                                                            random_state=2020)
    # train set in the second layer
    train_predict_feature = np.zeros((train_data2.shape[0], K))
    train_predict_feature_test = np.zeros((train_data2.shape[0]))
    trained_model = []

    # the first layer in Blending 第一层Bleding
    for j, clf in enumerate(clfs):
        # train each submodel
        print(j, clf)
        clf.fit(train_data1, train_label1)
        train_predict_feature[:, j] = clf.predict(train_data2)
        # save the trained model in the first layer
        trained_model.append(clf)
        F1_SCORE = f1_score(train_label2, train_predict_feature[:, j], average='binary')
        print("[SCORE]: " + str(F1_SCORE))
    train_predict_feature_test[:] = train_predict_feature.mean(axis=1)

    # gbm模型调参
    cv_params = {'n_estimators': [400, 450, 500, 550, 600, 650]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 400, 'max_depth': 3, 'min_child_weight': 6, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'reg_alpha': 0, 'reg_lambda': 1}
    model = XGBClassifier(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(train_predict_feature_test.reshape(-1, 1), train_label2)
    evalute_result = optimized_GBM.cv_results_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

    # the second layer in Blending 第二层Blending
    gbm = XGBClassifier(learning_rate=0.01, n_estimators=400, max_depth=3, min_child_weight=6, gamma=0.1,
                        reg_alpha=0.05, reg_lambda=3, subsample=0.8,
                        colsample_bytree=0.8, objective='binary:logistic', nthread=-1, scale_pos_weight=1).fit(
        train_predict_feature_test.reshape(-1, 1), train_label2)

    # 将训练好的模型存入pkl文件
    joblib.dump(gbm, "model.pkl")
    joblib.dump(trained_model, "clfs.pkl")
    joblib.dump(gbm, "../predict_code/model.pkl")
    joblib.dump(trained_model, "../predict_code/clfs.pkl")

    print("Training is complete")


if __name__ == '__main__':
    train_path = '../data/train.csv'
    train_func(train_path)
