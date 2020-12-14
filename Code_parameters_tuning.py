# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as sps
import time

RM_train = pd.read_csv('./input/data_train.csv')
R_test = pd.read_csv('./input/data_target_users_test.csv')
URM = pd.read_csv('./input/data_train.csv')
ICM = pd.read_csv('./input/data_ICM_title_abstract.csv')

##### URM
URM_tuples = [tuple(x) for x in URM.to_numpy()]

userList, itemList, ratingList = zip(*URM_tuples)

userList = list(userList)
userList = np.array(userList, dtype=np.int64)
itemList = list(itemList)
itemList = np.array(itemList, dtype=np.int64)

ratingList = list(ratingList)  # not needed
ratingList = np.array(ratingList, dtype=np.float)  # not needed

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all = URM_all.tocsr()

#### ICM

ICM_tuples = [tuple(x) for x in ICM.to_numpy()]
itemList_icm, featureList_icm, scoreList_icm = zip(*ICM_tuples)

itemList_icm = list(itemList_icm)
itemList_icm = np.array(itemList_icm, dtype=np.int64)

featureList_icm = list(featureList_icm)
featureList_icm = np.array(featureList_icm, dtype=np.int64)

scoreList_icm = list(scoreList_icm)
scoreList_icm = np.array(scoreList_icm, dtype=np.float64)

ICM_all = sps.coo_matrix((scoreList_icm, (itemList_icm, featureList_icm)))

#### Test

userTestList = [x for x in R_test.to_numpy()]
userTestList = zip(*userTestList)
userTestList = [list(a) for a in userTestList][0]

#### make validation and test
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

### ralph3 0.032
# from GraphBased import P3alphaRecommender
# recommender = P3alphaRecommender.P3alphaRecommender(URM_train)
# recommender.fit(normalize_similarity=True, topK=250)
# print(evaluator_test.evaluateRecommender(recommender)[0][10]['MAP'])

### PURE SVD 0.023
# from MatrixFactorization import PureSVDRecommender
# recommender = PureSVDRecommender.PureSVDRecommender(URM_train)
# recommender.fit(num_factors=400)

### SLIM ELASTIC NET not yet
# from SLIM_ElasticNet.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
# recommender = MultiThreadSLIM_ElasticNet(URM_train.tocsr())
# recommender.fit(topK=400)

### RP3beta  0.0329
# from GraphBased.RP3betaRecommender import RP3betaRecommender
# recommender = RP3betaRecommender(URM_train)
# recommender.fit(beta=-0.1,alpha=1.,topK=200)

### SLIM BPR 0.0375
# from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
# recommender = SLIM_BPR_Cython(URM_train, recompile_cython=False)
# recommender.fit(epochs=50, batch_size=100, sgd_mode='sgd', learning_rate=1e-2, positive_threshold_BPR=1)
# recommender.get_S_incremental_and_set_W()

### CBF KNN
### Usinng TF IDF
# from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
# recommender = ItemKNNCBFRecommender(URM_train, ICM_idf)
#
# recommender.fit(shrink=10, topK=800)

# from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
# recommender = SLIMElasticNetRecommender(URM_train)
##############
# from MatrixFactorization.IALSRecommender import IALSRecommender
# recommender = IALSRecommender(URM_train)

### Hybrid
##top pop
item_popularity = np.ediff1d(URM_all.tocsc().indptr)

popular_items = np.argsort(item_popularity)
popular_items = np.flip(popular_items, axis=0)
popular_items = popular_items[0:10]

# ## TF_DF
ICM_all = ICM_all.tocsr()
num_tot_items = ICM_all.shape[0]

# let's count how many items have a certain feature
items_per_feature = np.ediff1d(ICM_all.indptr) + 1
# print(items_per_feature)

IDF = np.array(np.log(num_tot_items / items_per_feature))

from scipy.sparse import diags
diags(IDF)

ICM_idf = ICM_all.copy()
ICM_idf = diags(IDF)*ICM_idf

from HybridRecommender2 import HybridRecommender2
recommender2 = HybridRecommender2(URM_train)

from HybridRecommender3 import HybridRecommender3
recommender = HybridRecommender3(URM_train)


checker = [0.0, "TF-IDF","BM25"]
MAP_list = []
for k in checker:
    print(f"Doing {k}")
    if k == 0.0:
        recommender2.fit(ICM=ICM_all)
        MAP_list.append(evaluator_test.evaluateRecommender(recommender2)[0][10]['MAP'])
        print(f"MAP for {k} added {MAP_list[-1]}")
        continue
    recommender.fit(ICM=ICM_all, k=k)
    MAP_list.append(evaluator_test.evaluateRecommender(recommender)[0][10]['MAP'])
    print(f"MAP for {k} added {MAP_list[-1]}")



import matplotlib.pyplot as pyplot

pyplot.plot(checker ,MAP_list, 'r-')
pyplot.ylabel('MAP')
pyplot.xlabel('k')
pyplot.show()
