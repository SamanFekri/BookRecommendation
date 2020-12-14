from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from KNN import ItemKNNCFRecommender as ICF
import numpy as np


class HybridRecommender3(BaseItemSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "Hybrid_Recommender3"

    def fit(self, ICM):
        slim_recommender = SLIM_BPR_Cython(self.URM_train, recompile_cython=False)
        slim_recommender.fit(epochs=50, batch_size=100, sgd_mode='sgd', learning_rate=1e-2, positive_threshold_BPR=1)

        rp3_recommender = RP3betaRecommender(self.URM_train)
        rp3_recommender.fit(beta=-0.1,alpha=1., topK=200)

        cbf_recommender = ItemKNNCBFRecommender(URM_train=self.URM_train, ICM_train=ICM)
        cbf_recommender.fit(800, 10, feature_weighting="TF-IDF")
        

        best_parameters = {'topK': 220, 'shrink': 56, 'similarity':"jaccard", "feature_weighting": "BM25"}

        itemKNNCF = ICF.ItemKNNCFRecommender(self.URM_train)
        itemKNNCF.fit(**best_parameters)

        P3alpha = P3alphaRecommender(self.URM_train)
        P3alpha.fit(topK = 280, alpha = .5, implicit=True)

        self.W_sparse = 0.1 * rp3_recommender.W_sparse + 0.3 * cbf_recommender.W_sparse + 0.1*itemKNNCF.W_sparse + .6*P3alpha.W_sparse + 0.1 * slim_recommender.W_sparse