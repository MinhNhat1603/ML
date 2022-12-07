import os
import shutil

import matplotlib.pyplot as pyplot
import numpy as np
from flask import Flask
from flask import render_template
from flask import request
from sklearn import linear_model
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC

app = Flask(__name__)
static_path = r'D:\workspace\project\ML\static\images'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/lesson-1')
def lesson1():
    rm_tree()
    number = request.args.get('number')
    center1_x = request.args.get('center1_x')
    center1_y = request.args.get('center1_y')
    center2_x = request.args.get('center2_x')
    center2_y = request.args.get('center2_y')
    if (number is None) \
            or (int(number) <= 0) \
            or (center1_x is None) \
            or (center1_y is None) \
            or (center2_x is None) \
            or (center2_y is None):
        return render_template('lesson1.html', is_show_image=False)
    number = int(number)
    center1_x = int(center1_x)
    center1_y = int(center1_y)
    center2_x = int(center2_x)
    center2_y = int(center2_y)
    np.random.seed(100)
    centers = [[center1_x, center1_y], [center2_x, center2_x]]
    data, labels = make_blobs(n_samples=number,
                              centers=np.array(centers),
                              random_state=1)
    group0 = []
    group1 = []
    for i in range(0, number):
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
    pyplot.plot(group0[:, 0], group0[:, 1], 'og', markersize=2)
    pyplot.plot(group1[:, 0], group1[:, 1], 'or', markersize=2)
    w = he_so[0]
    a = -w[0] / w[1]
    xx = np.linspace(2, 7, 100)
    yy = a * xx - intercept[0] / w[1]
    pyplot.plot(xx, yy, 'b')
    decision_function = svc.decision_function(train_data)
    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    support_vectors = train_data[support_vector_indices]
    ax = pyplot.gca()
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
    pyplot.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    pyplot.legend(['Group 0', 'Group 1'])
    pyplot.savefig(static_path + r'\lesson1.png')
    pyplot.clf()
    return render_template('lesson1.html',
                           is_show_image=True,
                           center1_x=center1_x,
                           center1_y=center1_y,
                           center2_x=center2_x,
                           center2_y=center2_y,
                           number=number)


@app.route('/lesson-2')
def lesson2():
    rm_tree()
    number = request.args.get('number')
    number_test = request.args.get('number_test')
    if (number is None) \
            or (int(number) <= 0) \
            or (number_test is None) \
            or (int(number_test) <= 0):
        return render_template('lesson2.html', is_show_image=False)
    number = int(number)
    number_test = int(number_test)
    np.random.seed(100)
    x = np.random.rand(number, 1) * 5
    y = 3 * (x - 2) * (x - 3) * (x - 4) + 10 * np.random.randn(number, 1)
    poly_features = PolynomialFeatures(degree=2, include_bias=True)
    x_poly = poly_features.fit_transform(x)
    x_test = (np.random.rand(number_test, 1) - 1 / 8) * 10
    y_test = 3 * (x_test - 2) * (x_test - 3) * (x_test - 4) + 10 * np.random.randn(number_test, 1)
    lin_reg = linear_model.LinearRegression(fit_intercept=False)  # fit_intercept = False for calculating the bias
    lin_reg.fit(x_poly, y)
    x_ve = np.linspace(-2, 10, 100)
    y_real = np.zeros(100, dtype=np.float64)
    x_ve_poly = poly_features.fit_transform(np.array([x_ve]).T)
    y_ve = np.matmul(x_ve_poly, lin_reg.coef_.T)
    for i in range(0, 100):
        y_real[i] = 3 * (x_ve[i] - 2) * (x_ve[i] - 3) * (x_ve[i] - 4)
    pyplot.plot(x, y, 'ro')
    pyplot.plot(x_test, y_test, 's')
    pyplot.plot(x_ve, y_ve, 'b')
    pyplot.plot(x_ve, y_real, '--')
    pyplot.title('Polynomial regression of order 2')
    pyplot.savefig(static_path + r'\lesson2.png')
    pyplot.clf()
    return render_template('lesson2.html',
                           is_show_image=True,
                           number=number,
                           number_test=number_test)


@app.route('/lesson-3')
def lesson3():
    rm_tree()
    x0 = request.args.get('x0')
    number = request.args.get('number')
    if (x0 is None) \
            or (int(x0) <= 0) \
            or (number is None)\
            or (int(number) <= 0):
        return render_template('lesson3.html', is_show_image=False)
    x0 = int(x0)
    number = int(number)
    eta = 0.9
    x9 = [x0]
    for it1 in range(100):
        x_ = x9[-1]
        x_new = x9[-1] - eta * 2 * x_ + 5 * np.cos(x_)
        if abs(2 * x_new + 5 * np.cos(x_new)) < 1e-3:  # just a small number
            break
        x9.append(x_new)
    (x, it) = (x9, it1)
    x = np.array(x)
    y = x ** 2 + 5 * np.sin(x)
    xx = np.linspace(-6, 6, number)
    yy = xx ** 2 + 5 * np.sin(xx)
    pyplot.subplot(2, 4, 1)
    pyplot.plot(xx, yy)
    index = 0
    pyplot.plot(x[index], y[index], 'ro')
    x1 = x[index]
    s = ' iter%d/%d, grad=%.3f ' % (index, it, 2 * x1 + 5 * np.cos(x1))
    pyplot.xlabel(s, fontsize=8)
    pyplot.axis([-7, 7, -10, 50])
    pyplot.subplot(2, 4, 2)
    pyplot.plot(xx, yy)
    index = 1
    pyplot.plot(x[index], y[index], 'ro')
    x2 = x[index]
    s = ' iter%d/%d, grad=%.3f ' % (index, it, 2 * x2 + 5 * np.cos(x2))
    pyplot.xlabel(s, fontsize=8)
    pyplot.axis([-7, 7, -10, 50])
    pyplot.subplot(2, 4, 3)
    pyplot.plot(xx, yy)
    index = 2
    pyplot.plot(x[index], y[index], 'ro')
    x3 = x[index]
    s = ' iter%d/%d, grad=%.3f ' % (index, it, 2 * x3 + 5 * np.cos(x3))
    pyplot.xlabel(s, fontsize=8)
    pyplot.axis([-7, 7, -10, 50])
    pyplot.subplot(2, 4, 4)
    pyplot.plot(xx, yy)
    index = 3
    pyplot.plot(x[index], y[index], 'ro')
    x4 = x[index]
    s = ' iter%d/%d, grad=%.3f ' % (index, it, 2 * x4 + 5 * np.cos(x4))
    pyplot.xlabel(s, fontsize=8)
    pyplot.axis([-7, 7, -10, 50])
    pyplot.subplot(2, 4, 5)
    pyplot.plot(xx, yy)
    index = 4
    pyplot.plot(x[index], y[index], 'ro')
    x5 = x[index]
    s = ' iter%d/%d, grad=%.3f ' % (index, it, 2 * x5 + 5 * np.cos(x5))
    pyplot.xlabel(s, fontsize=8)
    pyplot.axis([-7, 7, -10, 50])
    pyplot.subplot(2, 4, 6)
    pyplot.plot(xx, yy)
    index = 5
    pyplot.plot(x[index], y[index], 'ro')
    x6 = x[index]
    s = ' iter%d/%d, grad=%.3f ' % (index, it, 2 * x6 + 5 * np.cos(x6))
    pyplot.xlabel(s, fontsize=8)
    pyplot.axis([-7, 7, -10, 50])
    pyplot.subplot(2, 4, 7)
    pyplot.plot(xx, yy)
    index = 7
    pyplot.plot(x[index], y[index], 'ro')
    x7 = x[index]
    s = ' iter%d/%d, grad=%.3f ' % (index, it, 2 * x7 + 5 * np.cos(x7))
    pyplot.xlabel(s, fontsize=8)
    pyplot.axis([-7, 7, -10, 50])
    pyplot.subplot(2, 4, 8)
    pyplot.plot(xx, yy)
    index = 11
    pyplot.plot(x[index], y[index], 'ro')
    x8 = x[index]
    s = ' iter%d/%d, grad=%.3f ' % (index, it, 2 * x8 + 5 * np.cos(x8))
    pyplot.xlabel(s, fontsize=8)
    pyplot.axis([-7, 7, -10, 50])
    pyplot.tight_layout()
    pyplot.savefig(static_path + r'\lesson3.png')
    pyplot.clf()
    return render_template('lesson3.html',
                           is_show_image=True,
                           x0=x0,
                           number=number)


@app.route('/lesson-4')
def lesson4():
    rm_tree()
    number = request.args.get('number')
    number_test = request.args.get('number_test')
    if (number is None) \
            or (int(number) <= 0) \
            or (number_test is None) \
            or (int(number_test) <= 0):
        return render_template('lesson4.html', is_show_image=False)
    number = int(number)
    number_test = int(number_test)
    np.random.seed(100)
    X = np.random.rand(number, 1) * 5
    y = 3 * (X - 2) * (X - 3) * (X - 4) + 10 * np.random.randn(number, 1)
    poly_features = PolynomialFeatures(degree=8, include_bias=True)
    X_poly = poly_features.fit_transform(X)
    X_test = (np.random.rand(number_test, 1) - 1 / 8) * 10
    y_test = 3 * (X_test - 2) * (X_test - 3) * (X_test - 4) + 10 * np.random.randn(number_test, 1)
    X_poly_test = poly_features.fit_transform(X_test)
    lin_reg = linear_model.LinearRegression(fit_intercept=False)  # fit_intercept = False for calculating the bias
    lin_reg.fit(X_poly, y)
    x_ve = np.linspace(-2, 10, 100)
    y_real = np.zeros(100, dtype=np.float64)
    x_ve_poly = poly_features.fit_transform(np.array([x_ve]).T)
    y_ve = np.matmul(x_ve_poly, lin_reg.coef_.T)
    for i in range(0, 100):
        y_real[i] = 3 * (x_ve[i] - 2) * (x_ve[i] - 3) * (x_ve[i] - 4)

    pyplot.plot(X, y, 'ro')
    pyplot.plot(X_test, y_test, 's')
    pyplot.plot(x_ve, y_ve, 'b')
    pyplot.plot(x_ve, y_real, '--')
    pyplot.title('Polynomial regression of order 16')
    pyplot.savefig(static_path + r'\lesson4.png')
    pyplot.clf()
    return render_template('lesson4.html',
                           is_show_image=True,
                           number=number,
                           number_test=number_test)


if __name__ == '__main__':
    app.run()


def rm_tree():
    shutil.rmtree(static_path)
    os.mkdir(static_path)
