import tensorflow as tf
import pandas as pd
import numpy as np
import os

# 파일 읽어오기
from matplotlib.table import table
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

# 데이터 전처리
train[["Embarked", "Embarked_C", "Embarked_S", "Embarked_Q"]].head()

# X = preprocessing.scale(X)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=3)

##########################################################################################################
# fare (요금) 컬럼의 결측치를 평균값으로 채움
mean_fare = train["Fare"].mean()
print("Fare(Mean) = ${0:.3f}".format(mean_fare))
test.loc[pd.isnull(test["Fare"]), "Fare"] = mean_fare
test[pd.isnull(test["Fare"])]
##########################################################################################################

# 트레이닝 데이터를 가공...
feature_names = ["Pclass", "Sex", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S"]
# (input)
X_train = train[feature_names]
# 살았냐 죽었냐??(output)
y_train = train["Survived"]

#######################################################################################################################
# 러닝 모델 생성
# DT = 기본 예문

from sklearn.tree import DecisionTreeClassifier
seed = 37
model = DecisionTreeClassifier(max_depth=5, random_state=seed)
model.fit(X_train, y_train)

#######################################################################################################################

# graphviz 패키지 :살펴볼 것
# 빼!!!!한번보고 빼 다른모델 쓸때는 에러날거야...
########################################################################################################################
from sklearn.tree import export_graphviz
import graphviz

export_graphviz(model,
                feature_names=feature_names,
                class_names=["Perish", "Survived"],
                out_file="decision-tree.dot")

with open("decision-tree.dot") as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)
# 표현 해주는 plt.show() 같은 기능을 하는 함수 찾아볼 것!!
########################################################################################################################

# 테스트 (예측)
X_test = test[feature_names]
prediction = model.predict(X_test)

submission = pd.read_csv("data/gender_submission.csv", index_col="PassengerId")
submission["Survived"] = prediction
print(submission.shape, type(submission))

model_name = "Dicision_tree"
result_file = "result/result"+model_name+".csv"
submission.to_csv(result_file, mode='w')

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
import matplotlib.pyplot as plt

