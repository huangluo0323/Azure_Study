import numpy as np
import os
import argparse
import itertools
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from azureml.core import Dataset, Run

# 定义参数
parser = argparse.ArgumentParser()
parser.add_argument("--kernel", type=str, default="rbf",
                    help="Kernel type to be use in the algorithm")
parser.add_argument("--penalty", type=float, default=1.0,
                    help="Penalty parameter of the error term")
args = parser.parse_args()
# args:Namespace(kernel='rbf', penalty=1.0)

# 创建输出文件夹
os.makedirs("./outputs", exist_ok=True)

# 参数日志
run = Run.get_context()
run.log("Kernel type", np.str(args.kernel))
run.log("Penalty", np.float(args.penalty))

# 加载数据集
X, y = datasets.load_iris(return_X_y=True)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=223)
data = {"train": {"X": x_train, "y": y_train},
        "test": {"X": x_test, "y": y_test}}

# 训练SVM
svm_model = SVC(kernel=args.kernel, C=args.penalty, gamma="scale").fit(
    data["train"]["X"], data["train"]["y"])

# 预测
svm_predictions = svm_model.predict(data["test"]["X"])

# 模型准确率
accuracy = svm_model.score(data["test"]["X"], data["test"]["y"])
print(f"模型准确率：{round(accuracy,2)}")
run.log("Accuracy", np.float(accuracy))

# 模型精确率（查准率）
precision = precision_score(
    svm_predictions, data["test"]["y"], average='weighted')
print(f"模型精确率：{round(precision,2)}")
run.log('precision', precision)

# 模型召回率（查全率）
recall = recall_score(svm_predictions, data["test"]["y"], average="weighted")
print(f"模型召回率：{round(recall,2)}")
run.log("recall", recall)

# 模型f1值
f1 = f1_score(svm_predictions, data["test"]["y"], average="weighted")
print(f"模型f1值：{round(f1,2)}")
run.log("fi-score", f1)

# 创建混淆矩阵(confusion_matrix)
labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
labels_numbers = [0, 1, 2]
cm = confusion_matrix(y_test, svm_predictions, labels_numbers)
cm_json = {"schema_type": "confusion_matrix",
           "schema_version": "v1",
           "data": {"class_labels": labels, "matrix": cm.tolist()}}

# 画出混淆矩阵图形
# 标准化混淆矩阵
print("标准化混淆矩阵图")
cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Normalized_confusion_matrix")
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

thresh = cm.max() / 2.

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], ".2f"), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
run.log_image("Normalized_confusion_matrix", plot=plt)
plt.savefig(os.path.join("outputs","{0}.png".format("Normalized_confusion_matrix")))