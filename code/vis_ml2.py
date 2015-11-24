# -*- coding: utf-8 -*-
"""
This code uses "MLP" class defined in:
http://deeplearning.net/tutorial/mlp.html#tips-and-tricks-for-training-mlps
""" 

import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from mlp import MLP

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=500,
             batch_size=20, n_hidden=3):

    numpy.random.seed(1)
    rng = numpy.random.RandomState(1234)

    # 集団内の要素数 (散布図の通り、同じ色の2集団で 1クラスを形成)
    N = 100

    # 説明変数
    x = numpy.matrix([[0] * N + [1] * N + [0] * N + [1] * N,
                      [0] * N + [1] * N + [1] * N + [0] * N], dtype=numpy.float32).T
    x += numpy.random.rand(N * 4, 2) / 2
    # 目的変数
    y = numpy.array([0] * N * 2 + [1] * N * 2, dtype=numpy.int32)

    # 2 次元にプロット
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ['red'] * N * 2 + ['blue'] * N * 2
    ax.scatter(x[:, 0], x[:, 1], color=colors)
    plt.show()

    # Theano の共有変数として宣言
    x_data = theano.shared(value=x, name='x', borrow=True)
    y_data = theano.shared(value=y, name='y', borrow=True)

    n_train_batches = x_data.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    # MLPインスタンスを生成
    classifier = MLP(rng=rng, input=x, n_in=2, n_hidden=n_hidden, n_out=2)

    # 損失関数
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # 各係数行列、バイアスの更新処理
    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: x_data[index * batch_size: (index + 1) * batch_size],
            y: y_data[index * batch_size: (index + 1) * batch_size]
        }
    )

    # 隠れ層の出力を取得
    apply_hidden = theano.function(inputs=[x], outputs=classifier.hiddenLayer.output)
    labels = y_data.eval()

    # 3 次元にプロット
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 表示領域 / カメラアングルを指定
    ax.set_xlabel('x0')
    ax.set_xlim(-1, 1.5)
    ax.set_ylabel('x1')
    ax.set_ylim(-0.5, 1.5)
    ax.set_zlabel('z')
    ax.set_zlim(-1, 1)
    ax.view_init(azim=30, elev=30)

    # 座標 x0, x1 について 分離平面の z 座標を計算
    def calc_z(classifier, x0, x1):
        w = classifier.logRegressionLayer.W.get_value()
        b = classifier.logRegressionLayer.b.get_value()
        z = ((w[0, 0] - w[0, 1]) * x0 + (w[1, 0] - w[1, 1]) * x1 + b[0] - b[1]) / (w[2, 1] - w[2, 0])
        return z

    objs = []
    colors = ['red'] * N * 2 + ['blue'] * N * 2

    for epoch in range(n_epochs):
        for minibatch_index in xrange(n_train_batches):
            train_model(minibatch_index)

        # 10 エポックごとに描画
        if epoch % 10 == 0:
            z_data = apply_hidden(x_data.get_value())

            s = ax.scatter(z_data[:, 0], z_data[:, 1], z_data[:, 2], color=colors)
            zx0_min = z_data[:, 0].min()
            zx0_max = z_data[:, 0].max()
            zx1_min = z_data[:, 1].min()
            zx1_max = z_data[:, 1].max()
            bx0 = numpy.array([zx0_min, zx0_min, zx0_max, zx0_max])
            bx1 = numpy.array([zx1_min, zx1_max, zx1_max, zx0_min])
            bz = calc_z(classifier, bx0, bx1)
            # 分離平面
            tri = mplot3d.art3d.Poly3DCollection([zip(bx0, bx1, bz)], facecolor='gray', alpha=0.5)
            area = ax.add_collection3d(tri)
            objs.append((s, tri))

    # アニメーション開始
    ani = animation.ArtistAnimation(fig, objs, interval=40, repeat=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('im.mp4', writer=writer)
    #plt.show()


if __name__ == '__main__':
    test_mlp()
