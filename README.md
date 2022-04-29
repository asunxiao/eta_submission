# eta_submission
2020 West Lake Sword AI Competition -- encrypted traffic detection -- hn_2020

## 背景

近2020年将超过60%的企业将无法有效解密HTTPS流量，从而“无法有效检测出具有针对性的网络恶意软件。”届时加密的流量中将隐藏超过70%的网络恶意软件，而对抗这些威胁的手段将会受制于反解密系统，即便是最大的IT团队也无法忽视这一问题。本方向提供某款安全产品解析且标注的加密流量数据，给定已标注的训练样本，从测试样本中识别出所有具有恶意通讯行为的样本。

## 流程
1. 对数据进行预处理操作；
2. 划分训练集数据、验证集数据；
3. 对流量数据进行特征工程操作；
4. 对构建特征完成的样本集进行特征选择；
5. 建立多个机器学习模型，并进行模型融合；
6. 通过建立的模型，根据流量特征判断是否是恶意流量

## 重要特征


## 运行程序
train
```
python eta_submission/train_code/main_train.py
```
predict
```
python eta_submission/predict_code/main_predict.py
```
