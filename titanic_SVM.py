from math import exp

import math
import tensorflow as tf
import pandas as pd
import numpy as np
import os

# 파일 읽어오기
from sklearn.model_selection import train_test_split

train = pd.read_csv("data/train.csv")

test = pd.read_csv("data/test.csv")

# 성별 변수 수치화
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

# 리스트로 목록 갯수와 맞춰서 넣어야 하지 않나???
train["Embarked_C"] = train["Embarked"] == "C"
train["Embarked_S"] = train["Embarked"] == "S"
train["Embarked_Q"] = train["Embarked"] == "Q"

test["Embarked_C"] = test["Embarked"] == "C"
test["Embarked_S"] = test["Embarked"] == "S"
test["Embarked_Q"] = test["Embarked"] == "Q"

# 데이터 전처리 ########################################################################################################
########################################################################################################################
# fare (요금) 컬럼의 결측치를 평균값으로 채움
mean_fare = train["Fare"].mean()
# print("Fare(Mean) = ${0:.3f}".format(mean_fare))
test.loc[pd.isnull(test["Fare"]), "Fare"] = mean_fare
test[pd.isnull(test["Fare"])]
########################################################################################################################

########################################################################################################################
# 트레이닝 데이터를 가공...
feature_names = ["Pclass", "Sex", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S"]
# (input)
X_train = train[feature_names]
# 살았냐 죽었냐??(output)
Y_train = train["Survived"]

X_test = test[feature_names]
########################################################################################################################
# 러닝 모델 생성
# DT = 기본 예문

from sklearn.model_selection import train_test_split,RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics, preprocessing
from scipy.stats import itemfreq

# X = preprocessing.scale(X)
X_train_cv, X_test_cv, Y_train_cv, Y_test_cv = train_test_split(X_train, Y_train, test_size=0.3, random_state=3)

c_list = list()
g_list = list()

c_x_list = list()
g_x_list = list()

for x in np.arange(-7, 7, 0.5):
    # c_list.append(10**x)
    # g_list.append(10**x)
    c_x_list.append(x)
    g_x_list.append(x)
    c_list.append(exp(x))
    g_list.append(exp(x))

C_grid = c_list #[0.001, 0.01, 0.1, 1, 10]
gamma_grid = g_list #[0.001, 0.01, 0.1, 1]
parameters = {'C': C_grid, 'gamma' : gamma_grid}

model = GridSearchCV(SVC(kernel='rbf'), parameters, cv=10)
model.fit(X_train_cv, Y_train_cv)
# print(X_train_cv, type(X_train_cv), Y_train_cv, type(Y_train_cv))

best_C = model.best_params_['C']
best_gamma = model.best_params_['gamma']

print("SVM best C : " + str(best_C))
print("SVM best gamma : " + str(best_gamma))
#######################################################################################################################
# SVM 적용 ############################################################################################################
# 여기서 부터 볼 것!!!
# 시각화 및 출력데이터 정리
model_SVM = SVC(C=best_C,gamma=best_gamma)
model_SVM.fit(X_train_cv, Y_train_cv)

prediction = model_SVM.predict(X_test)

# # 테스트 (예측)

submission = pd.read_csv("data/gender_submission.csv", index_col="PassengerId")
submission["Survived"] = prediction
# print(submission.shape, type(submission))

result_file = "result/result_SVM.csv"
submission.to_csv(result_file, mode='w')

########################################################################################################################
# 여기서 확인할 수 있는 최대 정확도는 0.77 아직 cv, pca 미적용 #########################################################

for c in c_list:
    for g in g_list:
        model_SVM = SVC(C=c, gamma=best_gamma)
        model_SVM.fit(X_train_cv, Y_train_cv)
        prediction = model_SVM.predict(X_test_cv)

        accuracy = metrics.accuracy_score(Y_test_cv, prediction)
        print('Accuracy    = ' + str(np.round(accuracy, 2)))
########################################################################################################################



# confusion matrix 계산
# 테스트 데이터의 정답이 존재하지 않으므로 의미없다

# def confusion_matrix(pred, gt):
#     cont = np.zeros((2,2))
#     for i in [0, 1]:
#         for j in [0, 1]:
#             cont[i, j] = np.sum((pred == i) & (gt == j))
#     return cont

# survived == prediction

# print(cfm)
# import matplotlib.pyplot as plt
