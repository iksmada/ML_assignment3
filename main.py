import argparse
import operator
import csv
import pickle
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cluster, metrics
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import multiprocessing

from nltk.downloader import download
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from wordcloud import WordCloud


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


def ngram_extract(sentence, n=3):
    # <s> marks start and end of sentence - WORSE
    # sentence = "<s> " + sentence + " <s>"
    words = sentence.split()
    n_grams = []
    for i in range(len(words) - n):
        entry = ""
        for j in range(n):
            entry = entry + words[i + j] + " "
        # bi[:-1] -> delete space in the end
        n_grams.append(entry[:-1])

    return n_grams


def feature_extract(sentence, mydict):
    features = []
    bi_grams = ngram_extract(sentence, NGRAM)
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
        bi_grams = ngram_extract(sentence, NGRAM)
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


parser = argparse.ArgumentParser(description='K-Means with headlines')
parser.add_argument('-c', '--clusters', type=int, help='Max number of clusters to test', default=10)
parser.add_argument('-f', '--features', type=int, help='Number of features per sample', default=10)
parser.add_argument('-n', '--ngram', type=int, help='Number of word per gram', default=2)
parser.add_argument('-a', '--analyzer', type=str, help='Analyser of Ngram as word or char',
                    default='word', choices=("word", "char"))
parser.add_argument('--no-stemmer', help='Disable stemmer', action='store_false')
parser.add_argument('--no-stop', help='Disable stop words', action='store_false')
parser.add_argument('--no-tfidf', help='Disable tfidf - only freq', action='store_false')

args = vars(parser.parse_args())
CLUSTERS = args["clusters"]
FEATURES = args["features"]
NGRAM = args["ngram"]
ANALYZER = args["analyzer"]
STEMMER = args["no_stemmer"]
STOPWORDS = args["no_stop"]
TFIDF = args["no_tfidf"]


download('stopwords')
stop = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
num_cores = multiprocessing.cpu_count()


def parse_csv(row):
    sentence = row
    if STOPWORDS:
        sentence = remove_stopwords(sentence)
    if STEMMER:
        sentence = extract_stemmer(sentence)

    return sentence


with open('news_headlines.csv') as csvfile:
    rows = csv.reader(csvfile)
    # jump header
    next(rows, None)
    dates, original = zip(*rows)
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
    tf_transformer = TfidfVectorizer(max_features=FEATURES, ngram_range=(NGRAM, NGRAM), analyzer=ANALYZER)
    trainSamples = tf_transformer.fit_transform(unique_headlines)
    myDict = tf_transformer.vocabulary_
else:

    dictName = "dict-n" + str(NGRAM) + "f" + str(FEATURES)
    try:
        with open('obj/' + dictName + '.pkl', 'rb') as f:
            myDict = pickle.load(f)
    except EnvironmentError: # parent of IOError, OSError *and* WindowsError where available
        myDict = create_dict(unique_headlines)
        with open('obj/' + dictName + '.pkl', 'w+b') as f:
            pickle.dump(myDict, f, pickle.HIGHEST_PROTOCOL)

    trainSamples = Parallel(n_jobs=num_cores)(
        delayed(feature_extract)(headline, myDict) for headline in unique_headlines)

print("Most common n grams used as features, total " + str(len(myDict)))
print(myDict.keys())

# X_train, X_test = model_selection.train_test_split(trainSamples, train_size=0.8)

print(82 * '_')
print('N Clusters\ttime\tinertia\tvariance\tsilhouette')
clusters = range(max(2, NGRAM), CLUSTERS + 1)


def run_Kmeans(n):
    t0 = time()
    # Mini batch is faster
    # model = cluster.KMeans(n_clusters=n, n_jobs=-1)
    model = cluster.MiniBatchKMeans(n_clusters=n)

    distances = model.fit_transform(trainSamples)

    # cost
    cost = model.inertia_
    # silhoutte score
    try:
        silhouette = metrics.silhouette_score(trainSamples, model.labels_, sample_size=5000)
    except ValueError:
        silhouette = 1
    # variance
    variance = 0
    i = 0
    for label in model.labels_:
        variance = variance + distances[i][label]
        i = i + 1

    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f'
          % (str(n), (time() - t0), cost, variance, silhouette))

    return [cost, silhouette, variance]


stats = Parallel(n_jobs=num_cores)(
    delayed(run_Kmeans)(n) for n in clusters)

costs, silhouette_scores, variances = zip(*stats)

# plot
plt.scatter(clusters, costs)
plt.plot(clusters, costs)
plt.title("Cost")
plt.xlabel("Clusters")
plt.show()

plt.scatter(clusters, silhouette_scores)
plt.plot(clusters, silhouette_scores)
plt.title("Silhouette Score")
axes = plt.gca()
axes.set_ylim([axes.get_ylim()[0], 1.0])
plt.xlabel("Clusters")
plt.show()

plt.scatter(clusters, variances)
plt.plot(clusters, variances)
plt.title("Variance")
plt.xlabel("Clusters")
plt.show()

# print word cloud
n = int(input("For which number of clusters do you want to print the word cloud? From 2 to " + str(CLUSTERS+1)))
model = cluster.MiniBatchKMeans(n_clusters=n)
labels = model.fit_predict(trainSamples)

# separate clusters
clusters = dict()
i = 0
for label in labels:
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(headlines.index(unique_headlines[i]))
    i = i + 1


# make word clouds out of each cluster
wordclouds = []
for idx in clusters:
    wordclouds.append(WordCloud().generate(" ".join(operator.itemgetter(*clusters[idx])(original))))
    plt.imshow(wordclouds[idx], interpolation='bilinear')
    plt.axis("off")
    plt.title("Cluster " + str(idx))
    # plt.savefig("graphs/wordcloud-(2,2) " + str(i) + "-" + str(idx))
    plt.show()

# print(metrics.accuracy_score(y_test, pred))
