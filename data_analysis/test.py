import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

train = pd.read_csv("../data/train.csv", )

test = pd.read_csv("../data/test.csv")

# 성별 변수 수치화
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

# 리스트로 목록 갯수와 맞춰서 넣어야 하지 않나???
# train["Embarked_C"] = train["Embarked"] == "C"
# train["Embarked_S"] = train["Embarked"] == "S"
# train["Embarked_Q"] = train["Embarked"] == "Q"
#
# test["Embarked_C"] = test["Embarked"] == "C"
# test["Embarked_S"] = test["Embarked"] == "S"
# test["Embarked_Q"] = test["Embarked"] == "Q"


train[train["Embarked"] == "C"]["Embarked"] = 0
train[train["Embarked"] == "S"]["Embarked"] = 1
train[train["Embarked"] == "Q"]["Embarked"] = 2
print(train)
#
#
# colors=['red', 'green', 'blue']
# print(colors[1])
# print(train)
# scatter_matrix(train, alpha=0.5)
scatter_matrix(train, alpha=0.5, c=train.Embarked.apply)
# scatter_matrix(train_S, alpha=0.5, colors='blue')

plt.show()

# colors=['red','green']
# scatter_matrix(train,figsize=[20,20],marker='x',c=train.Survived.apply(lambda x:colors[x]))


# plt.scatter()
# plt.imshow(sm)

# matplotlib
# print(type(sm[1]), sm[1])
# load the data from the file
# df = pd.read_csv('../data/train.csv')
#

#
# # define colors list, to be used to plot survived either red (=0) or green (=1)
# colors=['red','green']
#
# # make a scatter plot
#
# for i, matr in enumerate()):
#     # plt.subplot(1, 20, i)
#     plt.scatter()
#     plt.show()
# #
# df.info()
#
# df = pd.DataFrame(np.random.randn(1000, 4), columns=['A','B','C','D'])
# scatter_matrix(df, alpha=0.2)