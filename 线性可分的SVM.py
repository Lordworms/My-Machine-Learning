import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, svm


def load_file(file_name):
    fr = open(file_name)
    data = []
    label = []
    for line in fr.readlines():
        arr = line.strip().split('\t')
        data.append([float(arr[0]), float(arr[1])])
        label.append(float(arr[2]))

    return data, label


def trans(data, label):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for i in range(0, len(label) - 1):
        if label[i] == 1:
            x1.append(data[0][i])
            y1.append(data[1][i])
        else:
            x2.append(data[0][i])
            y2.append(data[1][i])
    return x1, x2, y1, y2


if __name__ == '__main__':
    dataMat, labelMat = load_file('text1')
    print(dataMat)
    print(labelMat)
    data_train = dataMat[:-20]
    label_train = labelMat[:-20]
    print(data_train)
    print(label_train)
    data_test = dataMat[-20:]
    label_test = labelMat[-20:]
    data_test = np.array(data_test)
    label_test = np.array(label_test)
    print(data_test)
    print(label_test)
    demo = svm.SVC(kernel='linear')
    demo.fit(data_train, label_train)
    print(demo)
    w = demo.coef_[0]
    print('w=',w)
    b = demo.intercept_[0]
    print('b=',b)
    a = -w[0] / w[1]
    print(a)
    data_test = data_test.transpose()
    label_test = label_test.transpose()
    print(data_test)
    y = a * data_test[0] - b / w[1]
    print(y)
    xa, xb, ya, yb = trans(data_test, label_test)
    plt.scatter(xa, ya, color='red')
    plt.scatter(xb, yb, color='green')
    b1 = demo.support_vectors_[0]
    print(b1)
    plt.plot(data_test[0], y, color='blue', linewidth=3)
    plt.plot(data_test[0],y-1/w[1],linestyle='--',linewidth=2)
    plt.plot(data_test[0], y +1 / w[1], linestyle='--', linewidth=2)
    plt.show()
