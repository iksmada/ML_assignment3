import argparse
import operator
import csv
from time import time

from sklearn import cluster, metrics
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import multiprocessing


def rescale(x, min_new=0., max_new=255., min_original=-1, max_original=-1):
    """
        x numpy array like matrix
        if min_original or  max_original are not explict given,
        this funcition tries to use x.min() and x.max() to fit values
        """
    if min_original == -1 or max_original == -1:
        min_original = x.min()
        max_original = x.max()

    return ((max_new - min_new) / (max_original - min_original)) * (x - min_original) + min_new

def bigram_extract(sentence):
    words = sentence.split(" ")
    bi_grams = []
    for i in range(len(words) + 1):
        if i == 0:
            bi = "<s> " + words[i]
        elif i == len(words):
            bi = words[i - 1] + " <s>"
        else:
            bi = words[i - 1] + " " + words[i]
        bi_grams.append(bi)

    return bi_grams


def feature_extract(sentence, mydict):
    features = []
    bi_grams = bigram_extract(sentence)
    for key in mydict.keys():
        if key in bi_grams:
            features.append(1)
        else:
            features.append(0)

    return features


def create_dict(sentences):
    # count words freq
    bi_count = dict()
    for sentence in sentences:
        bi_grams = bigram_extract(sentence)
        for bi in bi_grams:
            if bi in bi_count.keys():
                bi_count[bi] = bi_count[bi] + 1
            else:
                bi_count[bi] = 1

    # select n most common words
    # sort by frequency
    sorted_words = sorted(bi_count.items(), key=operator.itemgetter(1), reverse=True)
    selected_words = dict(sorted_words[:FEATURES])

    return selected_words


parser = argparse.ArgumentParser(description='K-Means with headlines')
parser.add_argument('-c', '--clusters', type=int, help='Max number of clusters to test', default=10)
parser.add_argument('-f', '--features', type=int, help='Number of features per sample', default=10)
args = vars(parser.parse_args())
CLUSTERS = args["clusters"]
FEATURES = args["features"]

dates = []
headlines = []
with open('news_headlines.csv') as csvfile:
    rows = csv.reader(csvfile)
    # jump header
    next(rows, None)
    for row in rows:
        dates.append(row[0])
        headlines.append(row[1])

myDict = create_dict(headlines)

num_cores = multiprocessing.cpu_count()

trainSamples = Parallel(n_jobs=num_cores)(
    delayed(feature_extract)(headline, myDict) for headline in headlines)

# X_train, X_test = model_selection.train_test_split(trainSamples, train_size=0.8)

print(82 * '_')
print('N Clusters\ttime\tinertia\tvariance\tsilhouette')
costs = []
silhouette_scores = []
variances = []
clusters = range(2, CLUSTERS)
for n in clusters:
    t0 = time()
    # Mini batch is faster
    # model = cluster.KMeans(n_clusters=n, n_jobs=-1)
    model = cluster.MiniBatchKMeans(n_clusters=n)

    distances = model.fit_transform(trainSamples)

    # cost
    cost = model.inertia_
    #silloute score
    silhouette = metrics.silhouette_score(trainSamples, model.labels_, sample_size=10000)
    # variance
    variance = 0
    i = 0
    for label in model.labels_:
        variance = variance + distances[i][label]
        i = i + 1

    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f'
          % (str(n), (time() - t0), cost, variance, silhouette))
    costs.append(cost)
    silhouette_scores.append(silhouette)
    variances.append(variance)

    # plot
plt.scatter(clusters, costs)
plt.plot(clusters, costs)
plt.show()

plt.scatter(clusters, silhouette_scores)
plt.plot(clusters, silhouette_scores)
plt.show()

plt.scatter(clusters, variances)
plt.plot(clusters, variances)
plt.show()


# print(metrics.accuracy_score(y_test, pred))
