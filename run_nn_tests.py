"""Run supervised learning tests before and after DR and DR+clustering"""

from datetime import datetime
import sys

import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.random_projection import SparseRandomProjection

from mllib.loaders import load_adult
from mllib.helpers import balanced_accuracy_scorer, build_keras_clf, save_search_result

# Neural network parameters
try:
    N_GPUS = sys.argv[1]
except IndexError:
    print('Usage: %s <n_gpus>' % sys.argv[0])
    exit()
try:
    N_GPUS = int(N_GPUS)
except ValueError:
    print('n_gpus must be an integer')

#Optimal NN parameters from A1
HIDDEN_LAYER_SIZES = [(100,100)]
EPOCHS = [100]

dataset = 'adult'
learner_type = 'ANN'
loader_func = load_adult

df = loader_func(preprocess=True)

X = df[[c for c in df.columns if c != 'target']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('Start time: %s' % datetime.now())
# Phase 0: Baseline performance
dr_type = 'BASE'
print('Fitting %s' % dr_type)
n_samples, n_features = X.shape
BASE_PARAM_GRID = {
    'n_input_features': [n_features],
    'n_gpus': [N_GPUS],
    'hidden_layer_sizes': HIDDEN_LAYER_SIZES,
    'epochs': EPOCHS,
}
clf = KerasClassifier(build_fn=build_keras_clf, verbose=0)
grid_search = GridSearchCV(
    estimator=clf,
    param_grid=BASE_PARAM_GRID,
    scoring=balanced_accuracy_scorer,
    return_train_score=True,
    cv=4,
    verbose=1,
    n_jobs=1,
)
grid_search.fit(X_train_scaled, y_train)
test_score = grid_search.score(X_test_scaled, y_test)
save_search_result(grid_search, dataset, learner_type, extras='BASE_%.3f' % test_score)
print('Done fitting %s' % dr_type)

# Parameter sets by algorithm
PCA_COMPONENT_COUNTS = [1, 2, 10, 20, 30]
ICA_COMPONENT_COUNTS = [1, 2, 10, 20, 30]
RP_COMPONENT_COUNTS = [1, 2, 10, 20, 30, 80]
LDA_COMPONENT_COUNTS = [1]
N_CLUSTER_COUNTS = [2, 10, 20, 30]

# Phase 1: Dimensionality reduction (à la carte)
dr_types = ['PCA', 'ICA', 'RP', 'LDA']
dim_reducers = [
    PCA(random_state=0),
    FastICA(random_state=0),
    SparseRandomProjection(random_state=0),
    LDA()
]
component_count_lists = [
    PCA_COMPONENT_COUNTS,
    ICA_COMPONENT_COUNTS,
    RP_COMPONENT_COUNTS,
    LDA_COMPONENT_COUNTS
]

for dr_type, dr, dr_cc_list in zip(dr_types, dim_reducers, component_count_lists):
    for component_count in dr_cc_list:
        model = '%s_%i' % (dr_type, component_count)
        print('Fitting %s' % model)
        pipe = Pipeline([
            ('reduce_dim', dr),
            ('kerasclassifier', clf)
        ])
        param_grid = {
            'reduce_dim__n_components': [component_count],
            'kerasclassifier__n_input_features': [component_count],
            'kerasclassifier__hidden_layer_sizes': HIDDEN_LAYER_SIZES,
            'kerasclassifier__epochs': EPOCHS,
            'kerasclassifier__n_gpus': [N_GPUS],
        }
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=balanced_accuracy_scorer,
            return_train_score=True,
            cv=4,
            verbose=1,
            n_jobs=1,
        )
        grid_search.fit(X_train_scaled, y_train)
        test_score = grid_search.score(X_test_scaled, y_test)
        save_search_result(grid_search, dataset, model, extras='%.3f' % test_score)
    print('Done fitting %s' % dr_type)

# Phase 2: Dimensionality reduction plus clustering
dr_cluster_types = ['PCA+KM', 'ICA+KM', 'RP+KM', 'LDA+KM', 'PCA+EM', 'ICA+EM', 'RP+EM', 'LDA+EM']
dim_reducers = dim_reducers * 2
clusterers = [KMeans(random_state=0, n_jobs=-1)] * 4
clusterers.extend([GaussianMixture(random_state=0, n_init=1, init_params='kmeans')] * 4)
for dr_cluster_type, dr, clusterer, dr_cc_list in zip(dr_cluster_types, dim_reducers, clusterers, component_count_lists):
    for component_count in dr_cc_list:
        for cluster_count in N_CLUSTER_COUNTS:
            model = '%s_%i_%i' % (dr_cluster_type, component_count, cluster_count)
            print('Fitting %s' % model)

            # Apply DR
            dr.set_params(n_components=component_count)
            X_train_dr = dr.fit_transform(X_train_scaled)
            X_test_dr = dr.transform(X_test_scaled)

            # Apply clustering
            # Kwargs are different for KM and GMM
            if isinstance(clusterer, KMeans):
                kwargs = {'n_clusters': cluster_count}
            else:
                kwargs = {'n_components': cluster_count}
            clusterer.set_params(**kwargs)
            train_clusters = clusterer.fit_predict(X_train_dr)
            test_clusters = clusterer.predict(X_test_dr)

            # Add clusters to tranformed features
            X_train_dr_clust = np.hstack((X_train_dr, train_clusters[:, None]))
            X_test_dr_clust = np.hstack((X_test_dr, test_clusters[:, None]))

            pipe = Pipeline([
                ('kerasclassifier', clf)
            ])
            param_grid = {
                'kerasclassifier__n_input_features': [component_count + 1],
                'kerasclassifier__hidden_layer_sizes': HIDDEN_LAYER_SIZES,
                'kerasclassifier__epochs': EPOCHS,
                'kerasclassifier__n_gpus': [N_GPUS],
            }
            grid_search = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring=balanced_accuracy_scorer,
                return_train_score=True,
                cv=4,
                verbose=1,
                n_jobs=1,
            )
            grid_search.fit(X_train_dr_clust, y_train)
            test_score = grid_search.score(X_test_dr_clust, y_test)
            save_search_result(grid_search, dataset, model, extras='%.3f' % test_score)
    print('Done fitting %s' % dr_cluster_type)

print('End time: %s' % datetime.now())

