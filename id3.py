# khai báo thư viện
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from matplotlib import pyplot as plt


# đọc dữ liệu từ csv
data = pd.read_csv("TTT.csv")
print(data.head())

# chia thuộc tính và kết quả
feature_names = data.columns[:9]
target_names = data['CLASS'].unique().tolist()
X = data.drop('CLASS', axis=1)
y = data['CLASS']

# chia data thành training và testing
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1)  # 70% training and 30% test

# tạo đối tượng cây quyết định loại entropy
clf = DecisionTreeClassifier(criterion="entropy")

# Huấn luyện
clf = clf.fit(X_train, y_train)

# dự đoán
y_pred = clf.predict(X_test)

# tính độ chính xác 
print("Độ chính xác:", metrics.accuracy_score(y_test, y_pred)*100, "%")

# vẽ cây
fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(clf, feature_names=feature_names, class_names=target_names, filled=True)
fig.savefig("decistion_tree.png")

"""
cây quyết định cho trò chơi tic tac toe
kết quả: X thắng or thua
X đánh trước
X = 1, O = 2, trống = 0
dữ liệu có 9 cột tương ứng 9 ô trong tic tac toe
Kết quả: win -> X thắng, lose -> X thua.
TL: Top Left
TM: Top Mid
TR: Top Right
ML: Mid Left
MM: Mid Mid
MR: Mid Right
BL: Bottom Left
BM: bottom Mid
BR: Bottom Right
"""