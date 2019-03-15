from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split as ts
import numpy as np


def discriminative_s(
        negtive_samples,
        negtive_labels,
        positive_samples,
        positive_labels):
    length = positive_labels.shape[0]
    svm_scores = np.ones((length, length))

    for i in range(length):
        data_samples = np.concatenate(
            (positive_samples[i].reshape(
                1, -1), negtive_samples), axis=0)
        data_labels = np.concatenate(
            (positive_labels[i].reshape(
                1, -1), negtive_labels), axis=0)
        #X_train,X_test,y_train,y_test = ts(X,y,test_size=0.3)
        clf_linear = svm.LinearSVC(fit_intercept=False)
        clf_linear.fit(data_samples, data_labels)
        # 将剩下的正包样本 放入训练器 获得分数
        scores_i = clf_linear.decision_function(positive_samples)
        svm_scores[i] = scores_i
        # print(svm_scores)
        # 给svm分数矩阵 根据排序 记录每个节点的名次
        order_matrix = np.ones((length, length))
        index_order = np.argsort(svm_scores, axis=1)

    # 这个循环能优化吗？
    for i in range(length):
        for j in range(length):
            index = index_order[i][j]
            order_matrix[i][index] = j + 1

    # 计算 S
    s_matrix = np.ones((length, length))
    for i in range(length):
        for j in range(length):
            if svm_scores[i][j] > 0 and svm_scores[j][i] > 0:
                s_matrix[i][j] = 1 / (order_matrix[i][j] * order_matrix[j][i])
            else:
                s_matrix[i][j] = 0
    return s_matrix
