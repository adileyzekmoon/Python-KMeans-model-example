import numpy
import sklearn
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.cluster import KMeans

# load up dataset
digits = load_digits()
print(digits.data)
print(digits.target)

# scale down data to range of -1 and 1 for easier/faster computation
data = scale(digits.data)
y = digits.target

# declare K (number of sets)
k = 10
# Obtain sample size(number of input) and features(from each sample) so this example is 1797 samples x 64 features
sample, features = data.shape
print(type(sample), type(features))
print(sample, features)


# function to score k means pulled from sklearn site
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


# n_clusters is the number of clusters, init is random(can be sped up with 'k-means++'
# n_init is number of times algo is run
kmeans_model = KMeans(n_clusters=k, init="random", n_init=10)

bench_k_means(kmeans_model, "1", data)
