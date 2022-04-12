import os
import datetime
import numpy as np
import time
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import (
    load_digits,
    load_iris,
    make_blobs,
    make_circles,
    make_classification,
    make_moons,
    make_multilabel_classification
)
from os.path import join


def create_logdir():
    dt = datetime.datetime.fromtimestamp(time.time())
    logdir = os.path.join("./outputs/", dt.strftime("%Y-%m-%d"))

    print(f"Logging to {logdir}")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir

def plot_hist(ax, data, label, title, alpha=None, xlabel=None):
    ax.hist(data, alpha=alpha or 1, label=label)
    if label:
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    
    


def construct_datasets(n_samples, noise=0.1):
    # linearly separable
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1,
        class_sep=2
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)
    
    # high dim classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=100,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1,
        class_sep=2
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    high_dim_class = (X, y)
    
    # high dim multiclass
    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=100,
        random_state=1,
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    y_int_label = []
    for i, item in enumerate(y):
        y_int_label.append( int("".join(str(x) for x in item), 2))
        
    high_dim_multiclass = (X, np.array(y_int_label))
    
    # independent high dimensional guassian
    guassian_one = rng.normal(5, 1, (n_samples, 10))
    guassian_two = rng.normal(-5, 1, (n_samples, 10))
    X = np.concatenate((guassian_one, guassian_two))
    y = np.concatenate(([0] * n_samples, [1] * n_samples))
    ind_high_dim_guassian = (X, y)

    # non overlapping blobs
    centers = [(-5, -5), (5, 5)]
    X, y = make_blobs(
        n_samples=n_samples, centers=centers, shuffle=False, random_state=42
    )
    y[: n_samples // 2] = 0
    y[n_samples // 2 :] = 1
    separated_blobs = (X, y)

    # overlapping blobs
    centers = [(1, 1), (1, 5)]
    X, y = make_blobs(
        n_samples=n_samples, centers=centers, shuffle=False, random_state=42
    )
    y[: n_samples // 2] = 0
    y[n_samples // 2 :] = 1
    overlapping_blobs = (X, y)

    # iris data (2 features)
    iris = load_iris()
    X, y = iris.data[:, :2], iris.target  # we only take the first two features.
    iris_first_two_feats = (X, y)

    # mnist (2 pcs)
    X, y = load_digits(return_X_y=True)
    pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=42))
    X = pca.fit_transform(X)
    pca_mnist = (X, y)

    ds = [
        make_moons(n_samples=n_samples, noise=noise, random_state=42),
        make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42),
        linearly_separable,
        separated_blobs,
        overlapping_blobs,
        iris_first_two_feats,
        pca_mnist,
#         ind_high_dim_guassian,
        high_dim_class,
        high_dim_multiclass
    ]

    names = [
        "moons",
        "circles",
        "linearly_separable",
        "separated_blobs",
        "overlapping_blobs",
        "iris_first_two_feats",
        "pca_mnist",
#         "ind_high_dim_guassian",
        "high_dim_class",
        "high_dim_multiclass"
    ]
    
    return ds, names


def get_binned_acc(data, X, y, clf, quantile_step: float = 0.10, thresholds: list = None, cumulative: bool = False):
    """Returns (pecentile, acc) to plot"""
    acc = []
    percentiles = []

    if not thresholds:
        quantiles = np.arange(0, 1, 0.10)
        quantiles = np.append(quantiles, 1)
        # find binned accuracy for data
        thresholds = np.quantile(data, quantiles)
    else:
        quantiles = thresholds

    for i in range(1, len(thresholds)):
        curr_thresh = thresholds[i - 1]
        next_thresh = thresholds[i]
        # handle edge case where quantiles include a repeated number
        if next_thresh == curr_thresh:
            idx_between_thresh = np.where(data <= curr_thresh)[0]
        else:
            if cumulative:
                idx_between_thresh = np.where(data <= next_thresh)[0]
            else:
                idx_between_thresh = np.where(
                    np.logical_and(data > curr_thresh, data <= next_thresh)
                )[0]
        if len(idx_between_thresh): # there are points between these thresholds
            acc.append(clf.score(X[idx_between_thresh], y[idx_between_thresh]))
            percentiles.append(quantiles[i])
    return percentiles, acc

def plot_scores_vs_acc(ax, clf, bin_data, X_test, y_test, label, cumulative: bool = False):
    x, acc = get_binned_acc(bin_data, X_test, y_test, clf, cumulative=cumulative)
    ax.plot(x, acc, alpha=0.5, marker='o', label=label)
    ax.set_title("score vs acc (cumulative)" if cumulative else 'score vs acc')
    ax.set_xlabel("score (quantiles)")
    ax.set_ylabel("acc")
    ax.legend()
    
    
def hamming(s1: str, s2: str):
    """Calculate the Hamming distance between two bit strings"""
    if isinstance(s1, (int, np.integer)):
        s1 = format(s1, 'b')
    if isinstance(s2, (int, np.integer)):
        s2 = format(s2, 'b')
    if len(s1) < len(s2):
        s1 = '0' * (len(s2) - len(s1)) + s1
    elif len(s2) < len(s1):
        s2 = '0' * (len(s1) - len(s2)) + s2
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))