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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# X = preprocessing.scale(X)
X_train_cv, X_test_cv, Y_train_cv, Y_test_cv = train_test_split(X_train, Y_train, test_size=0.3, random_state=3)

k_grid = np.arange(1,51,1)

# 거리에 따른
weights = ['uniform','distance']
parameters = {'n_neighbors':k_grid, 'weights':weights}

gridCV = GridSearchCV(KNeighborsClassifier(), parameters, cv = 10)                                      # cv : 교차 검증
gridCV.fit(X_train, Y_train)

best_k = gridCV.best_params_['n_neighbors']
best_w = gridCV.best_params_['weights']

print("Best k : " + str(best_k))
print("Best weight : " + best_w)

knn_best = KNeighborsClassifier(n_neighbors=best_k, weights = best_w)
knn_best.fit(X_train, Y_train)
Y_pred = knn_best.predict(X_test)
#######################################################################################################################
# SVM 적용 ############################################################################################################
# 시각화 및 출력데이터 정리
# k-NN + PCA 적용
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.3, random_state=3)
knn_best = KNeighborsClassifier(n_neighbors=best_k, weights = best_w)
knn_best.fit(X_train, Y_train)
Y_pred = knn_best.predict(X_test)
print( "Best Accuracy : " + str(np.round(metrics.accuracy_score(Y_test,Y_pred),3)))

# # 테스트 (예측)

submission = pd.read_csv("data/gender_submission.csv", index_col="PassengerId")
submission["Survived"] = prediction
# print(submission.shape, type(submission))

result_file = "result/result_SVM.csv"
submission.to_csv(result_file, mode='w')