from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

import numpy as np


class HybridRecommender(BaseItemSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "Hybrid_Recommender"

    def fit(self, weights, ICM):
        slim_recommender = SLIM_BPR_Cython(self.URM_train, recompile_cython=False)
        slim_recommender.fit(epochs=50, batch_size=100, sgd_mode='sgd', learning_rate=1e-2, positive_threshold_BPR=1)

        rp3_recommender = RP3betaRecommender(self.URM_train)
        rp3_recommender.fit(beta=-0.1,alpha=1., topK=200)

        cbf_recommender = ItemKNNCBFRecommender(self.URM_train, ICM)
        cbf_recommender.fit(800, 10)

        self.W_sparse = 0.2 * slim_recommender.W_sparse + 0.3 * rp3_recommender.W_sparse + 0.2 * cbf_recommender.W_sparse
