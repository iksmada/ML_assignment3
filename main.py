import argparse
import operator
import csv
import pickle
import re
from time import time
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cluster, metrics
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
import matplotlib.pyplot as plt
from matplotlib import offsetbox

from joblib import Parallel, delayed
import multiprocessing

from nltk.downloader import download
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud

def remove_stopwords(sentence):
    cleaned = []
    for word in sentence.split():
        if word not in stop:
            cleaned.append(word)
    return " ".join(cleaned)


def extract_stemmer(sentence):
    cleaned = []
    for word in sentence.split():
        cleaned.append(stemmer.stem(word))
    # remove space in the begging
    return " ".join(cleaned)


def extract_lemma(sentence):
    cleaned = []
    for word in sentence.split():
        cleaned.append(wlem.lemmatize(word))
    # remove space in the begging
    return " ".join(cleaned)


parser = argparse.ArgumentParser(description='K-Means with headlines')
parser.add_argument('-s', '--size', type=int, help='Size of dataset to use', default=-1)
parser.add_argument('-c', '--clusters', type=int, help='Max number of clusters to test', default=10)
parser.add_argument('-f', '--features', type=int, help='Number of features per sample', default=10)
parser.add_argument('-n1', '--mingram', type=int, help='Min number of word per gram', default=2)
parser.add_argument('-n2', '--maxgram', type=int, help='Max number of word per gram', default=2)
parser.add_argument('-a', '--analyzer', type=str, help='Analyser of Ngram as word or char',
                    default='word', choices=("word", "char"))
parser.add_argument('-i', '--filename', type=str, help='Name of the file containing the dataset',
                    default='news_headlines.csv')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--no-lemma', help='Disable lemmatize', action='store_false')
group.add_argument('--no-stemmer', help='Disable stemmer', action='store_false')
parser.add_argument('--no-stop', help='Disable stop words', action='store_false')
parser.add_argument('--no-tfidf', help='Disable tfidf - only freq', action='store_false')
parser.add_argument('--no-norm', help='Disable number normalization', action='store_false')
parser.add_argument('-v', '--verbose', help='Show more info for clustering', action='store_true')

args = vars(parser.parse_args())
print(args)
SIZE = args["size"]
CLUSTERS = args["clusters"]
FEATURES = args["features"]
MINGRAM = args["mingram"]
MAXGRAM = args["maxgram"]
ANALYZER = args["analyzer"]
FILENAME = args["filename"]
STEMMER = args["no_stemmer"]
STOPWORDS = args["no_stop"]
TFIDF = args["no_tfidf"]
NORMALIZE = args["no_norm"]
LEMMA = args["no_lemma"]
VERBOSE = args["verbose"]


download('stopwords')
download('wordnet')
stop = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
wlem = WordNetLemmatizer()
num_cores = multiprocessing.cpu_count()


def parse_csv(row):
    sentence = row
    if STOPWORDS:
        sentence = remove_stopwords(sentence)
    if STEMMER:
        sentence = extract_stemmer(sentence)
    if LEMMA:
        sentence = extract_lemma(sentence)
    if NORMALIZE:
        sentence = re.sub("[0-9]+(\.[0-9]+)?[^\s]*", "", sentence)
    return sentence


with open(FILENAME) as csvfile:
    rows = csv.reader(csvfile)
    # jump header
    next(rows, None)
    dates, original = zip(*rows)
    #original = original[:SIZE]
    if not STOPWORDS and not STEMMER:
        headlines = original
    else:
        headlines = Parallel(n_jobs=num_cores)(
            delayed(parse_csv)(row) for row in original)

# to avoid outliers
unique_headlines = list(set(headlines))
print("Unique reduced from " + str(len(headlines)) + " to " + str(len(unique_headlines)) + " samples")

if TFIDF:
    # tf-idf
    tf_transformer = TfidfVectorizer(
        max_features=FEATURES, 
        ngram_range=(MINGRAM, MAXGRAM), 
        analyzer=ANALYZER,
        #max_df=0.95, min_df=0.001
    )
    trainSamples = tf_transformer.fit_transform(unique_headlines)
    myDict = tf_transformer.vocabulary_
else:
    print("Deu pau!")
    """
    dictName = "dict-n" + str(NGRAM) + "f" + str(FEATURES)
    try:
        with open('obj/' + dictName + '.pkl', 'rb') as f:
            myDict = pickle.load(f)
            print("Loaded: " + dictName)
    except EnvironmentError: # parent of IOError, OSError *and* WindowsError where available
        myDict = create_dict(unique_headlines)
        with open('obj/' + dictName + '.pkl', 'w+b') as f:
            pickle.dump(myDict, f, pickle.HIGHEST_PROTOCOL)
            print("Saved: " + dictName)

    trainSamples = Parallel(n_jobs=num_cores)(
        delayed(feature_extract)(headline, myDict) for headline in unique_headlines)
    """

print("Most common n grams used as features, total " + str(len(myDict)))
print(myDict)
print("Most common n grams used as features, total " + str(len(myDict)))

myDictStats = dict()
for key in myDict.keys():
    n_words = len(key.split())
    if n_words in myDictStats.keys():
        myDictStats[n_words] = myDictStats[n_words] + 1
    else:
        myDictStats[n_words] = 1
print(myDictStats)


# X_train, X_test = model_selection.train_test_split(trainSamples, train_size=0.8)
"""
print(82 * '_')
print('N Clusters\ttime\tinertia\tvariance\tsilhouette')
clusters = range(2, CLUSTERS + 1)
#clusters = range(int(CLUSTERS/2), CLUSTERS + 1)

def run_Kmeans(n):
    t0 = time()
    # Mini batch is faster
    # model = cluster.KMeans(n_clusters=n, n_jobs=-1)
    model = cluster.KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=0, n_jobs=1)

    labels = model.fit_predict(trainSamples)

    # cost
    cost = model.inertia_
    # silhoutte score
    try:
        silhouette = metrics.silhouette_score(trainSamples, model.labels_, sample_size=5000)
    except ValueError:
        silhouette = 1
    # variance
    count = []
    for cluss in range(n):
        count.append(np.count_nonzero(labels == cluss))
    variance = np.std(count)
    if VERBOSE:
        clusters = np.zeros((n, trainSamples.shape[1]))
        i=0
        for cluss in labels:
            clusters[cluss] = clusters[cluss] + trainSamples[i]
            i = i + 1
        print(count)
        i = 0
        print("means:"),
        for cluss in clusters:
            print(" %f" % (sum(cluss/count[i]))),
            i = i + 1
        print("")

    print('%-9s\t%.2fs\t%7i\t%8i\t%10.3f'
          % (str(n), (time() - t0), cost, int(variance), silhouette))

    return [cost, silhouette, variance]


stats = Parallel(n_jobs=num_cores)(
    delayed(run_Kmeans)(n) for n in clusters)

costs, silhouette_scores, variances = zip(*stats)

# plot
plt.scatter(clusters, costs)
plt.plot(clusters, costs)
plt.title("Cost (Inertia)")
plt.xlabel("Number of Clusters")
plt.savefig('cost')
plt.show()

plt.scatter(clusters, silhouette_scores)
plt.plot(clusters, silhouette_scores)
plt.title("Silhouette Score")
axes = plt.gca()
axes.set_ylim([max(0, axes.get_ylim()[0]), min(1.0, axes.get_ylim()[1])])
plt.xlabel("Number of Clusters")
plt.savefig('silhouette')
plt.show()

plt.scatter(clusters, variances)
plt.plot(clusters, variances)
plt.title("Standard Deviation")
plt.xlabel("Number of Clusters")
plt.savefig('variance')
plt.show()
"""
# print word cloud
n = int(input("For which number of clusters do you want to print the word cloud? From 2 to " + str(CLUSTERS)))
while (n != -1):
    print("Re training for %d clusters" % n)
    model = cluster.KMeans(n_clusters=n, init='k-means++', max_iter=500, n_init=10, random_state=0, n_jobs=-1)
    labels = model.fit_predict(trainSamples)
    count = []
    for label in range(n):
        count.append(np.count_nonzero(labels == label))
    print("Distribution of samples per cluster")
    print(count)

    # separate clusters
    clusters = dict()
    i = 0
    for label in labels:
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
        i = i + 1

    """
    # print clustered headlines
    for label in clusters:
        raw_input("...")
        print("CLUSTER " + str(label))
        for idx in clusters[label]:
            print original[idx]
    """
    """
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    visual = tsne.fit_transform(clusters)

    plot_embedding(visual,
                   "t-SNE embedding of the headlines (time %.2fs)" %
                   (time() - t0))

    plt.show()
    """

    # make word clouds out of each cluster
    wordclouds = []
    for idx in clusters:
        wordclouds.append(WordCloud().generate(" ".join(operator.itemgetter(*clusters[idx])(unique_headlines))))
        plt.imshow(wordclouds[idx], interpolation='bilinear')
        plt.axis("off")
        plt.title("Cluster " + str(idx))
        plt.savefig("wordcloud-" + str(n) + "-" + str(idx))
        #plt.show()

    # print(metrics.accuracy_score(y_test, pred))
    n = int(input("For which number of clusters do you want to print the word cloud? From 2 to " + str(CLUSTERS)))