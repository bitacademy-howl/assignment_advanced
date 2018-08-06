import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 파일 읽어오기
from sklearn.model_selection import train_test_split

train = pd.read_csv("../data/train.csv")

test = pd.read_csv("../data/test.csv")



desc = train.describe()
print(desc)

corr_df = np.round(train.corr(),3)
print(corr_df)

plt_matrix = pd.plotting.scatter_matrix(train,alpha=0.8, diagonal='kde')
#
# plt.subplot(1,5,i+1)
#
# for i, one_img in enumerate(conv2d_img):
# plt.subplot()



# 성별 변수 수치화
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1