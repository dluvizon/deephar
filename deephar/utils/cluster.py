import numpy as np
# import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans

# def plothist(x):
    # vmin = x.min()-1
    # vmax = x.max()+1
    # bins = np.arange(vmin, vmax, (vmax - vmin)/50)
    # plt.hist(x, bins=bins)
    # plt.show()

# def scatterpred(pred):
    # plt.scatter(pred[:,0], pred[:,1])
    # plt.show()

# def scatter_kmeans(pred):
    # plt.scatter(pred[:,0], pred[:,1], color='b')
    # c,v = kmeans(pred, 8)
    # plt.scatter(c[:,0], c[:,1], color='r')
    # plt.show()

def most_assigned(x, c):
    nb_c = len(c)
    assign = np.zeros(nb_c)
    for i in range(len(x)):
        y = x[i].reshape((1,2))
        d = np.sqrt(np.sum(np.power(y.repeat(nb_c, axis=0) - c, 2), axis=1))
        assign[d.argmin()] += 1
    return assign.argmax()

def mean_on_most_assigned(x, c):
    nb_c = len(c)
    assign = np.zeros(nb_c)
    mean = np.zeros(c.shape)
    for i in range(len(x)):
        y = x[i].reshape((1,2))
        d = np.sqrt(np.sum(np.power(y.repeat(nb_c, axis=0) - c, 2), axis=1))
        idx = d.argmin()
        assign[idx] += 1
        mean[idx,:] += x[i]
    idx = assign.argmax()
    return mean[idx,:] / assign[idx]

# def best_kmeans(pred):
    # plt.scatter(pred[:,0], pred[:,1], color='b')
    # c,v = kmeans(pred, 3)
    # plt.scatter(c[:,0], c[:,1], color='g')
    # n = most_assigned(pred, c)
    # plt.scatter(c[n,0], c[n,1], color='r')
    # plt.show()

def clustering_joints(y_pred, k=3):
    _,nb_spl,nb_joints,dim = y_pred.shape
    y = np.zeros((nb_spl, nb_joints, dim))
    for s in range(nb_spl):
        for j in range(nb_joints):
            d = y_pred[:,s,j]
            c,v = kmeans(d, k)
            n = most_assigned(d, c)
            y[s,j,:] = c[n]
    return y

def clustering_grid(y_pred, size=10):
    _, nb_spl, nb_joints, dim = y_pred.shape
    assert dim == 2
    yp = np.zeros((nb_spl, nb_joints, dim))
    for s in range(nb_spl):
        for j in range(nb_joints):
            d = y_pred[:,s,j,:]
            xmin = d[:,0].min()
            ymin = d[:,1].min()
            xmax = d[:,0].max()
            ymax = d[:,1].max()
            xstep = (xmax - xmin) / size
            ystep = (ymax - ymin) / size
            c = np.zeros((size * size, dim))
            for x in range(size):
                for y in range(size):
                    c[x + size*y, 0] = xmin + (x + 0.5) * xstep
                    c[x + size*y, 1] = ymin + (y + 0.5) * ystep
            yp[s,j,:] = mean_on_most_assigned(d, c)
    return yp

def mean_joints(y_pred):
    _, nb_spl, dim, nb_joints = y_pred.shape
    assert dim == 2
    yp = np.zeros((nb_spl, dim, nb_joints))
    for s in range(nb_spl):
        for j in range(nb_joints):
            d = y_pred[:,s,:,j]
            yp[s, 0, j] = d[:,0].mean()
            yp[s, 1, j] = d[:,1].mean()
    return yp

