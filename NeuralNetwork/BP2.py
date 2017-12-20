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


def BP(X, y, eta, loop=60000, step=100):
    input_n = X.shape[1]
    output_n = 1

    w = init_parameter((input_n + 2, 1))  # 5 x 1
    theta = init_parameter((1, 1))  # 1 x 1
    gamma = init_parameter((input_n + 2, 1))  # 5 x 1
    v = init_parameter((input_n, input_n + 2))  # 3 x 5

    sample_x = range(1, loop + 1, step)
    sample_y = list()

    for iter in xrange(loop):
        l0 = X  # 4 x 3
        l1 = sigmoid(np.dot(l0, v) - gamma.T)  # 4 x 5
        l2 = sigmoid(np.dot(l1, w) - theta.T)  # 4 x 1

        delta_w = eta * np.dot(l1.T, l2 * (1 - l2) * (y - l2))  # 5 x 1
        delta_theta = -eta * np.sum(l2 * (1 - l2) * (y - l2))  # 1 x 1
        delta_v = eta * np.dot(l0.T, (l1 * (1 - l1) * np.dot(l2 * (1 - l2) * (y - l2), w.T)))
        delta_gamma = -eta * np.dot((l1 * (1 - l1)).T * w, l2 * (1 - l2) * (y - l2))

        w += delta_w
        theta += delta_theta
        v += delta_v
        gamma += delta_gamma

        if iter % step == 0:
            sample_y.append(1.0 / 2.0 * np.sum((y - l2) ** 2))

    print(l2)
    return sample_x, sample_y


if __name__ == "__main__":
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    y = np.array([[0, 1, 1, 0]]).T

    plt.xlabel('loop num')

    plt.ylabel('E')

    s_x, s_y = BP(X=X, y=y, eta=0.1)
    show(s_x, s_y, "eta=0.1")
    s_x, s_y = BP(X=X, y=y, eta=10)
    show(s_x, s_y, "eta=10")
    s_x, s_y = BP(X=X, y=y, eta=1)
    show(s_x, s_y, "eta=1")
    plt.show()
