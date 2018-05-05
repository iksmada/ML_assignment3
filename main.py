from os import listdir, path, makedirs
import csv

import numpy as np

from sklearn import cluster, model_selection, metrics
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import multiprocessing


def rescale(x, min_new=0., max_new =255., min_original=-1, max_original=-1):
    """
        x numpy array like matrix
        if min_original or  max_original are not explict given,
        this funcition tries to use x.min() and x.max() to fit values
        """
    if min_original == -1 or max_original == -1:
        min_original = x.min()
        max_original = x.max()

    return ((max_new - min_new) / (max_original - min_original)) * (x - min_original) + min_new


def feature_extract(sentence):

    return np.zeros(10)


dates = []
headlines = []
with open('news_headlines.csv') as csvfile:
    rows = csv.reader(csvfile)
    # jump header
    next(rows, None)
    for row in rows:
        dates.append(row[0])
        headlines.append(row[1])

num_cores = multiprocessing.cpu_count()

trainSamples = Parallel(n_jobs=num_cores)(
            delayed(feature_extract)(i) for i in headlines)


#X_train, X_test = model_selection.train_test_split(trainSamples, train_size=0.8)

scores = []
clusters = range(1, 10)
for n in clusters:
    # Mini batch is faster
    # model = cluster.KMeans(n_clusters=n, n_jobs=-1)
    model = cluster.MiniBatchKMeans(n_clusters=n)

    model.fit(trainSamples)
    # could score test set
    scores.append(model.score(trainSamples))

# plot
plt.scatter(clusters, scores)
plt.plot(clusters, scores)
plt.show()

# print(metrics.accuracy_score(y_test, pred))
