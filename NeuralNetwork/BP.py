import numpy as np
import matplotlib.pyplot as plt


def show(x, y, label):
    plt.plot(x, y, label=label)
    plt.legend()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def init_parameter(shape):
    np.random.seed(1)
    return 2 * np.random.random(shape) - 1


def BP(X, y, eta, loop=1000, step=10):
    w = init_parameter((X.shape[1], 1))
    theta = init_parameter((1, 1))

    sample_x = range(1, loop + 1, step)
    sample_y = list()

    for iter in xrange(loop):
        l0 = X  # 4 x 3
        l1 = sigmoid(np.dot(l0, w) - theta)  # 4 x 1

        delta_w = np.dot(X.T, eta * l1 * (1 - l1) * (y - l1))  # 3 * 1
        delta_theta = np.sum(-eta * l1 * (1 - l1) * (y - l1))

        w += delta_w
        theta += delta_theta

        if iter % step == 0:
            sample_y.append(1.0 / 2.0 * np.sum((y - l1) ** 2))

    return sample_x, sample_y


if __name__ == "__main__":
    X = np.array([[0, 0, 1],
                  [1, 1, 1],
                  [1, 0, 1],
                  [0, 1, 1]])

    y = np.array([[0, 1, 1, 0]]).T

    plt.xlabel('loop num')

    plt.ylabel('E')

    s_x, s_y = BP(X=X, y=y, eta=0.1)
    show(s_x, s_y, "eta=0.1")
    s_x, s_y = BP(X=X, y=y, eta=0.2)
    show(s_x, s_y, "eta=0.2")
    s_x, s_y = BP(X=X, y=y, eta=1)
    show(s_x, s_y, "eta=1")
    plt.show()
