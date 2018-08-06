import pandas as pd

# 데이터 로딩 ####################################################################################
# object = dataframe
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
##################################################################################################

##################################################################################################
# 데이터 전처리
# 성별 변수 수치화
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

train["Embarked_C"] = train["Embarked"] == "C"
train["Embarked_S"] = train["Embarked"] == "S"
train["Embarked_Q"] = train["Embarked"] == "Q"

test["Embarked_C"] = test["Embarked"] == "C"
test["Embarked_S"] = test["Embarked"] == "S"
test["Embarked_Q"] = test["Embarked"] == "Q"
##################################################################################################

