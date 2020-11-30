# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as sps
import time

RM_train=pd.read_csv('./input/data_train.csv')
R_test=pd.read_csv('./input/data_target_users_test.csv')
URM=pd.read_csv('./input/data_train.csv')
ICM = pd.read_csv('./input/data_ICM_title_abstract.csv')

##### URM
URM_tuples = [tuple(x) for x in URM.to_numpy()]

userList, itemList, ratingList = zip(*URM_tuples)

userList = list(userList)
userList=np.array(userList,dtype=np.int64)
itemList = list(itemList)
itemList=np.array(itemList,dtype=np.int64)

ratingList = list(ratingList)                        #not needed
ratingList=np.array(ratingList,dtype=np.int64)       #not needed

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all = URM_all.tocsr()

#### ICM

ICM_tuples = [tuple(x) for x in ICM.to_numpy()]
itemList_icm, featureList_icm, scoreList_icm = zip(*ICM_tuples)

itemList_icm = list(itemList_icm)
itemList_icm = np.array(itemList_icm,dtype=np.int64)

featureList_icm = list(featureList_icm)
featureList_icm = np.array(featureList_icm,dtype=np.int64)

scoreList_icm = list(scoreList_icm)
scoreList_icm = np.array(scoreList_icm,dtype=np.float64)

ICM_all = sps.coo_matrix((scoreList_icm, (itemList_icm, featureList_icm)))

#### Test

userTestList = [x for x in R_test.to_numpy()]
userTestList = zip(*userTestList)
userTestList = [list(a) for a in userTestList][0]

#### make validation and test
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.80)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.80)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


### SLIM BRP train

from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
#
# recommender = SLIM_BPR_Cython(URM_train, recompile_cython=False)
#
# recommender.fit(epochs=10, batch_size=1, sgd_mode='sgd', learning_rate=1e-4, positive_threshold_BPR=1)


recommender = SLIM_BPR_Cython(URM_all, recompile_cython=False)
recommender.fit(epochs=20, batch_size=1, sgd_mode='sgd', learning_rate=1e-4, positive_threshold_BPR=1)
slim_recoms = recommender.recommend(userTestList, cutoff=10)


### content base filtering

from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

recommender = ItemKNNCBFRecommender(URM_all, ICM_all)
recommender.fit(shrink=10, topK=100)

cbf_recoms = recommender.recommend(userTestList, cutoff=10)


recomList = []

for i in range(len(slim_recoms)):
    temp = []
    k = 0
    for j in range(10):
        if slim_recoms[i][j] not in temp:
            temp.append(slim_recoms[i][j])
            while k < 10 and cbf_recoms[i][k] in temp:
                k += 1
            if k < 10:
                temp.append(cbf_recoms[i][k])

    recomList.append(' '.join(str(e) for e in temp))

print(recomList)

res = {"user_id": userTestList, "item_list": recomList}
result = pd.DataFrame(res, columns= ['user_id', 'item_list'])

result.to_csv('outputs/hybrid_slim_cbfv1.csv', index = False, header=True)

