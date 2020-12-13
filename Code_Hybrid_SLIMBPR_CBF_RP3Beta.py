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


### hybrid recommender

### Usinng TF IDF
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
############## top pop

item_popularity = np.ediff1d(URM_all.tocsc().indptr)

popular_items = np.argsort(item_popularity)
popular_items = np.flip(popular_items, axis=0)
popular_items = popular_items[0:10]
###########

from HybridRecommender import HybridRecommender
recommender = HybridRecommender(URM_all)
recommender.fit([0.2, 0.3, 0.2], ICM_idf)

recoms = recommender.recommend(userTestList, cutoff=10)

recomList = []

for i in range(len(recoms)):
    user_id = userTestList[i]
    start_pos = URM_train.indptr[user_id]
    end_pos = URM_train.indptr[user_id + 1]
    if start_pos == end_pos:
        recomList.append(' '.join(str(e) for e in popular_items))
    else:
        recomList.append(' '.join(str(e) for e in recoms[i]))

# print(recomList)

res = {"user_id": userTestList, "item_list": recomList}
result = pd.DataFrame(res, columns= ['user_id', 'item_list'])

result.to_csv('outputs/hybrid_slim_cbf_rp3v1.csv', index = False, header=True)

