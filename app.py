import os

import matplotlib.pyplot as plt
import numpy as np
from flask import Flask
from flask import render_template
from flask import request
from sklearn import linear_model
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/lesson-1')
def lesson1():
    if os.path.exists('D:\\workspace\\project\\ML\\static\\images\\lesson1.png'):
        os.remove('D:\\workspace\\project\\ML\\static\\images\\lesson1.png')

    number = request.args.get('number')

    if number is None:
        return render_template('lesson1.html', showImg=False)

    number = int(number)
    center1X = int(request.args.get('center1X'))
    center1Y = int(request.args.get('center1Y'))
    center2X = int(request.args.get('center2X'))
    center2Y = int(request.args.get('center2Y'))

    np.random.seed(100)
    N = number

    centers = [[center1X, center1Y], [center2X, center2X]]
    data, labels = make_blobs(n_samples=N,
                              centers=np.array(centers),
                              random_state=1)

    group0 = []
    group1 = []

    for i in range(0, N):
        if labels[i] == 0:
            group0.append([data[i, 0], data[i, 1]])
        elif labels[i] == 1:
            group1.append([data[i, 0], data[i, 1]])

    group0 = np.array(group0)
    group1 = np.array(group1)

    res = train_test_split(data, labels, train_size=0.8, test_size=0.2, random_state=1)

    train_data, test_data, train_labels, test_labels = res

    group0 = []
    group1 = []

    for i in range(0, train_data.shape[0]):
        if train_labels[i] == 0:
            group0.append([train_data[i, 0], train_data[i, 1]])
        elif train_labels[i] == 1:
            group1.append([train_data[i, 0], train_data[i, 1]])

    group0 = np.array(group0)
    group1 = np.array(group1)

    svc = LinearSVC(C=100, loss="hinge", random_state=42, max_iter=100000)

    svc.fit(train_data, train_labels)

    he_so = svc.coef_
    intercept = svc.intercept_

    plt.plot(group0[:, 0], group0[:, 1], 'og', markersize=2)
    plt.plot(group1[:, 0], group1[:, 1], 'or', markersize=2)

    w = he_so[0]
    a = -w[0] / w[1]
    xx = np.linspace(2, 7, 100)
    yy = a * xx - intercept[0] / w[1]

    plt.plot(xx, yy, 'b')

    decision_function = svc.decision_function(train_data)
    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    support_vectors = train_data[support_vector_indices]

    ax = plt.gca()

    DecisionBoundaryDisplay.from_estimator(
        svc,
        train_data,
        ax=ax,
        grid_resolution=50,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )
    plt.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )

    plt.legend(['Group 0', 'Group 1'])

    plt.savefig('D:\\workspace\\project\\ML\\static\\images\\lesson1.png')

    return render_template('lesson1.html', showImg=True, center1X=center1X, center1Y=center1Y, center2X=center2X,
                           center2Y=center2Y, number=number)


@app.route('/lesson-2')
def lesson2():
    np.random.seed(100)

    N = 30
    X = np.random.rand(N, 1) * 5
    y = 3 * (X - 2) * (X - 3) * (X - 4) + 10 * np.random.randn(N, 1)

    poly_features = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly_features.fit_transform(X)

    N_test = 20

    X_test = (np.random.rand(N_test, 1) - 1 / 8) * 10
    y_test = 3 * (X_test - 2) * (X_test - 3) * (X_test - 4) + 10 * np.random.randn(N_test, 1)

    X_poly_test = poly_features.fit_transform(X_test)

    lin_reg = linear_model.LinearRegression(fit_intercept=False)  # fit_intercept = False for calculating the bias

    lin_reg.fit(X_poly, y)

    x_ve = np.linspace(-2, 10, 100)
    y_ve = np.zeros(100, dtype=np.float64)
    y_real = np.zeros(100, dtype=np.float64)
    x_ve_poly = poly_features.fit_transform(np.array([x_ve]).T)

    y_ve = np.matmul(x_ve_poly, lin_reg.coef_.T)

    for i in range(0, 100):
        y_real[i] = 3 * (x_ve[i] - 2) * (x_ve[i] - 3) * (x_ve[i] - 4)

    # Tinh sai so cua scikit-learn
    y_train_predict = lin_reg.predict(X_poly)
    # print(y_train_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
    print('sai so binh phuong trung binh - tap training: %.6f' % (sai_so_binh_phuong_trung_binh / 2))

    # Tinh sai so cua scikit-learn
    y_test_predict = lin_reg.predict(X_poly_test)
    # print(y_test_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y_test, y_test_predict)
    print('sai so binh phuong trung binh - tap test: %.6f' % (sai_so_binh_phuong_trung_binh / 2))

    plt.plot(X, y, 'ro')
    plt.plot(X_test, y_test, 's')
    plt.plot(x_ve, y_ve, 'b')
    plt.plot(x_ve, y_real, '--')
    plt.title('Hoi quy da thuc bac 2')

    # plt.show()
    plt.savefig('D:\\workspace\\project\\ML\\static\\images\\lesson2.png')

    return render_template('lesson2.html', showImg=True)


@app.route('/lesson-3')
def lesson3():
    x0 = -5
    eta = 0.1
    (x, it) = myGD1(x0, eta)
    x = np.array(x)
    y = cost(x)

    n = 101
    xx = np.linspace(-6, 6, n)
    yy = xx ** 2 + 5 * np.sin(xx)

    plt.subplot(2, 4, 1)
    plt.plot(xx, yy)
    index = 0
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize=8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2, 4, 2)
    plt.plot(xx, yy)
    index = 1
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize=8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2, 4, 3)
    plt.plot(xx, yy)
    index = 2
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize=8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2, 4, 4)
    plt.plot(xx, yy)
    index = 3
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize=8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2, 4, 5)
    plt.plot(xx, yy)
    index = 4
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize=8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2, 4, 6)
    plt.plot(xx, yy)
    index = 5
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize=8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2, 4, 7)
    plt.plot(xx, yy)
    index = 7
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize=8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2, 4, 8)
    plt.plot(xx, yy)
    index = 11
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize=8)
    plt.axis([-7, 7, -10, 50])

    plt.tight_layout()
    # plt.show()
    plt.savefig('D:\\workspace\\project\\ML\\static\\images\\lesson3.png')

    return render_template('lesson3.html', showImg=True)


@app.route('/lesson-4')
def lesson4():

    plt.savefig('D:\\workspace\\project\\ML\\static\\images\\lesson4.png')

    return render_template('lesson4.html', showImg=True)


@app.route('/lesson-5')
def lesson5():

    plt.savefig('D:\\workspace\\project\\ML\\static\\images\\lesson5.png')

    return render_template('lesson5.html', showImg=True)


@app.route('/lesson-6')
def lesson6():

    plt.savefig('D:\\workspace\\project\\ML\\static\\images\\lesson6.png')

    return render_template('lesson6.html', showImg=True)


@app.route('/lesson-7')
def lesson7():

    plt.savefig('D:\\workspace\\project\\ML\\static\\images\\lesson7.png')

    return render_template('lesson7.html', showImg=True)


@app.route('/lesson-8')
def lesson8():

    plt.savefig('D:\\workspace\\project\\ML\\static\\images\\lesson8.png')

    return render_template('lesson8.html', showImg=True)


@app.route('/lesson-9')
def lesson9():

    plt.savefig('D:\\workspace\\project\\ML\\static\\images\\lesson9.png')

    return render_template('lesson9.html', showImg=True)


@app.route('/lesson-10')
def lesson10():

    plt.savefig('D:\\workspace\\project\\ML\\static\\images\\lesson10.png')

    return render_template('lesson10.html', showImg=True)


if __name__ == '__main__':
    app.run()


def grad(x):
    return 2 * x + 5 * np.cos(x)


def cost(x):
    return x ** 2 + 5 * np.sin(x)


def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        if abs(grad(x_new)) < 1e-3:  # just a small number
            break
        x.append(x_new)
    return (x, it)
