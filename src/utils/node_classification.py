from typing import Dict, Sequence
import dataclasses as dc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
from sklearn.model_selection import RepeatedStratifiedKFold


@dc.dataclass(frozen=True)
class ClassificationScores:
    accuracy: float
    macro_f1: float
    micro_f1: float
    cm: np.ndarray


@dc.dataclass(frozen=True)
class ClusteringScores:
    fmi: float
    mi: float
    normalised_mi: float
    adjusted_mi: float
    rand_score: float
    adjusted_rand_score: float
    davies_bouldin_score: float
    calinski_harabasz_score: float


def evaluate_emb_logistic_regression(embeddings: np.ndarray, labels_train: pd.Series, labels_test: pd.Series,
                                     unique_labels: Sequence):
    embeddings = StandardScaler().fit_transform(embeddings)

    log_reg = LogisticRegression(class_weight='balanced').fit(embeddings[labels_train.index, :], labels_train)
    preds = log_reg.predict(embeddings[labels_test.index, :])

    # pipe = Pipeline([('scaler', scaler), ('log-reg', log_reg)])
    # pipe.fit(embeddings[labels_train.index, :], labels_train)

    acc = metrics.accuracy_score(y_true=labels_test, y_pred=preds)
    macro_f1 = metrics.f1_score(y_true=labels_test, y_pred=preds, average='macro')
    micro_f1 = metrics.f1_score(y_true=labels_test, y_pred=preds, average='micro')
    cm = metrics.confusion_matrix(y_true=labels_test, y_pred=preds, labels=unique_labels)
    scores = ClassificationScores(accuracy=acc, macro_f1=macro_f1, micro_f1=micro_f1, cm=cm)
    return scores


def evaluate_emb_kmeans(embeddings: np.ndarray, labels: pd.Series,
                        num_clusters: int = None, include_non_labelled_in_clustering=True):
    # labels = pd.concat((labels_train, labels_test), axis=0)
    if num_clusters is None:
        num_clusters = labels.nunique()

    embeddings = StandardScaler().fit_transform(embeddings)

    if include_non_labelled_in_clustering:
        kmeans_clf = KMeans(n_clusters=num_clusters).fit(embeddings)
        cluster_labels = kmeans_clf.labels_[labels.index]
    else:
        kmeans_clf = KMeans(n_clusters=num_clusters).fit(embeddings[labels.index, :])
        cluster_labels = kmeans_clf.labels_

    # embeddings = embeddings[labels.index, :]
    # pipe = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=num_clusters))])
    # pipe.fit(embeddings)

    fmi = metrics.fowlkes_mallows_score(labels, cluster_labels)
    mi = metrics.mutual_info_score(labels, cluster_labels)
    normalised_mi = metrics.normalized_mutual_info_score(labels, cluster_labels)
    adjusted_mi = metrics.adjusted_mutual_info_score(labels, cluster_labels)
    rand_score = metrics.rand_score(labels, cluster_labels)
    adjusted_rand_score = metrics.adjusted_rand_score(labels, cluster_labels)

    davies_bouldin_score = metrics.davies_bouldin_score(embeddings[labels.index, :], labels)
    calinski_harabasz_score = metrics.calinski_harabasz_score(embeddings[labels.index, :], labels)
    scores = ClusteringScores(
        fmi=fmi,
        mi=mi,
        normalised_mi=normalised_mi,
        adjusted_mi=adjusted_mi,
        rand_score=rand_score,
        adjusted_rand_score=adjusted_rand_score,
        davies_bouldin_score=davies_bouldin_score,
        calinski_harabasz_score=calinski_harabasz_score

    )
    return scores


def evaluate_multiple_embeddings(embeddings: np.ndarray, labels: pd.Series, unique_labels: Sequence,
                                 rskf_random_state: int = 113):
    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=rskf_random_state)
    embeddings = StandardScaler().fit_transform(embeddings)
    y_ = labels.loc[~labels.isnull()]

    all_scores = []

    for train_index, test_index in rskf.split(np.empty_like(y_), y_):
        node_train_index = y_.index[train_index]
        node_test_index = y_.index[test_index]

        X_train, X_test = embeddings[node_train_index, :], embeddings[node_test_index, :]
        y_train, y_test = labels.loc[node_train_index], labels.loc[node_test_index]

        log_reg = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
        preds = log_reg.predict(X_test)

        acc = metrics.accuracy_score(y_true=y_test, y_pred=preds)
        macro_f1 = metrics.f1_score(y_true=y_test, y_pred=preds, average='macro')
        micro_f1 = metrics.f1_score(y_true=y_test, y_pred=preds, average='micro')
        cm = metrics.confusion_matrix(y_true=y_test, y_pred=preds, labels=unique_labels)
        scores = ClassificationScores(accuracy=acc, macro_f1=macro_f1, micro_f1=micro_f1, cm=cm)
        all_scores.append(scores)
    return all_scores
