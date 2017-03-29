def foo(a):
    return a

#
# # coding: utf-8
#
# # # Logistic regression
# #
# # In this exercise you will train a logistic regression model via gradient descent in two simple scenarios.
# #
# # The general setup is as follows:
# # * we are given a set of pairs $(x, y)$, where $x \in R^D$ is a vector of real numbers representing the features, and $y \in \{0,1\}$ is the target,
# # * for a given $x$ we model the probability of $y=1$ by $h(x):=g(w^Tx)$, where $g$ is the sigmoid function: $g(z) = \frac{1}{1+e^{-z}}$,
# # * to find the right $w$ we will optimize the so called logarithmic loss: $J(w) = -\frac{1}{n}\sum_{i=1}^n y_i \log{h(x_i)} + (1-y_i) \log{(1-h(x_i))}$,
# # * with the loss function in hand we can improve our guesses iteratively:
# #     * $w_j^{t+1} = w_j^t - \text{step_size} \cdot \frac{\partial J(w)}{\partial w_j}$,
# # * we can end the process after some predefined number of epochs (or when the changes are no longer meaningful).
#
# # Let's start with the simplest example - linear separated points on a plane.
#
# # In[2]:
#
# get_ipython().magic(u'matplotlib inline')
#
# import numpy as np
#
# np.random.seed(123)
#
# # these parametrize the line
# a = 0.3
# b = -0.2
# c = 0.001
#
# # True/False mapping
# def lin_rule(x, noise=0.):
#     return a * x[0] + b * x[1] + c + noise < 0.
#
# # Just for plotting
# def get_y_fun(a, b, c):
#     def y(x):
#         return - x * a / b - c / b
#     return y
#
# lin_fun = get_y_fun(a, b, c)
#
#
# # In[3]:
#
# # Training data
#
# n = 500
# range_points = 1
# sigma = 0.05
#
# X = range_points * 2 * (np.random.rand(n, 2) - 0.5)
# y = [lin_rule(x, sigma * np.random.normal()) for x in X]
#
# print X[:10]
# print y[:10]
#
#
# # Let's plot the data.
#
# # In[4]:
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# range_plot = 1.1
# h = .002
#
# plt.figure(figsize=(11,11))
#
# plt.scatter(X[:, 0], X[: , 1], c=y)
#
# _x = np.linspace(-range_plot, range_plot, 1000)
# _y = lin_fun(_x)
#
# plt.plot(_x, _y)
#
#
# # Now, let's implement and train a logistic regression model.
#
# # In[23]:
#
# a = np.array([[1, 2], [3, 4]])
# b = np.array([5, 6])
# # print a.shape
# for v in a:
#     print b.dot(v); print len(v)
# #     print v; print v.shape
#
#
# # In[74]:
#
# help(np.r_)
#
#
# # In[82]:
#
# import math
# import sys
#
# """Init param values"""
# w = np.array([0, 0, 0])
#
# """Add a column with 1s to data"""
# X2 = np.append(X, np.ones((len(X), 1)), axis=1)
#
# """Logistic function"""
# g = lambda z: 1 / (1 + np.exp(-z))
#
# """Takes (w and) x and returns the prob"""
# h = lambda x: g(w.dot(x))
#
# """Partial derivative of the loss function I(w) wrt. w_i"""
# partial_derivative = lambda i: sum((h(x) - y) * x[i] for (x, y) in zip(X2, y))
# # partial_derivative = lambda i: sum((h(x) - y) * (x[i] if i < x.size else 1) for (x, y) in zip(X, y))
#
# """Gradient vector of the loss function"""
# gradient = lambda w: np.array([partial_derivative(i) for i in xrange(w.size)])
#
# """Loss function"""
# loss_function = lambda w: (-1) * sum(yi * math.log(h(x) + sys.float_info.min) + (1 - yi) * math.log(1 - h(x) + sys.float_info.min) for (x, yi) in zip(X2, y))
#
# """Gradient descent params"""
# iters = 50
# alfa = 0.1
#
# # gradient descent
# logging = []
# for _ in xrange(iters):
#     loss = loss_function(w)
#     print loss
#     logging.append(loss)
#     w = w - (alfa * gradient(w))
#
#
# # Let's visually asses our model. We can do this by using our estimates for $a,b,c$.
#
# # In[83]:
#
# plt.figure(figsize=(11,11))
#
# #################################################################
# # TODO: Pass your estimates for a,b,c to the get_y_fun function #
# #################################################################
#
# lin_fun2 = get_y_fun(w[0], w[1], w[2])
#
# _y2 = lin_fun2(_x)
#
# plt.figure(figsize=(11,11))
# plt.scatter(X[:, 0], X[: , 1], c=y)
# plt.plot(_x, _y, _x, _y2)
#
#
# # Let's now complicate the things a little bit and make our next problem nonlinear.
#
# # In[ ]:
#
# # Parameters of the ellipse
# s1 = 1.
# s2 = 2.
# r = 0.75
# m1 = 0.15
# m2 = 0.125
#
# # True/False mapping, checks whether we are inside the ellipse
# def circle_rule(x, noise=0.):
#     return s1 * (x[0] - m1) ** 2 + s2 * (x[1] - m2) ** 2 + noise < r ** 2.
#
#
# # In[ ]:
#
# # Training data
#
# n = 500
# range_points = 1
#
# sigma = 0.1
#
# X = range_points * 2 * (np.random.rand(n, 2) - 0.5)
#
# y = [circle_rule(x, sigma * np.random.normal()) for x in X]
#
# print X[:10]
# print y[:10]
#
#
# # Let's plot the data.
#
# # In[ ]:
#
# range_plot = 1.1
# h = .005
#
# plt.figure(figsize=(11,11))
#
# xx, yy = np.meshgrid(np.arange(-range_plot, range_plot, h), np.arange(-range_plot, range_plot, h))
# Z = np.array(map(circle_rule, np.c_[xx.ravel(), yy.ravel()]))
#
# Z = Z.reshape(xx.shape)
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
#
# plt.scatter(X[:, 0], X[: , 1], c=y)
#
#
# # Now, let's train a logistic regression model to tackle this problem. Note that we now need a nonlinear decision boundary.
#
# # Hint:
# # <sub><sup><sub><sup><sub><sup>
# # Use feature engineering.
# # </sup></sub></sup></sub></sup></sub>
#
# # In[ ]:
#
# ################################################################
# # TODO: Implement logistic regression and compute its accuracy #
# ################################################################
#
#
# # Let's visually asses our model.
# #
# # Contrary to the previous scenario, converting our weights to parameters of the ground truth curve may not be straightforward. It's easier to just provide predictions for a set of points in $R^2$.
#
# # In[ ]:
#
# range_plot = 1.1
# h = .005
#
# xx, yy = np.meshgrid(np.arange(-range_plot, range_plot, h), np.arange(-range_plot, range_plot, h))
# X_plot = np.c_[xx.ravel(), yy.ravel()]
#
# print X_plot
# print X_plot.shape
#
# ############################################################
# # TODO: Compute true/false predictions for the X_plot data #
# ############################################################
#
# # preds = ...
#
#
# # In[ ]:
#
# plt.figure(figsize=(11,11))
#
# Z = preds
# Z = np.array(Z).reshape(xx.shape)
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
#
# Z = np.array(map(circle_rule, X_plot))
# Z = Z.reshape(xx.shape)
#
# plt.pcolormesh(xx, yy, Z, alpha=0.1, cmap=plt.cm.Paired)
#
# plt.scatter(X[:, 0], X[: , 1], c=y)
#
