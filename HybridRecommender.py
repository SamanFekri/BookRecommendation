from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

import numpy as np


class HybridRecommender(BaseItemSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "Hybrid_Recommender"

    def fit(self, weights, ICM, top_popular, use_top_pop):
        self.top_popular = top_popular
        self.use_top_pop = use_top_pop
        slim_recommender = SLIM_BPR_Cython(self.URM_train, recompile_cython=False)
        slim_recommender.fit(epochs=50, batch_size=100, sgd_mode='sgd', learning_rate=1e-2, positive_threshold_BPR=1)

        rp3_recommender = RP3betaRecommender(self.URM_train)
        rp3_recommender.fit(beta=-0.1,alpha=1., topK=200)

        cbf_recommender = ItemKNNCBFRecommender(self.URM_train, ICM)
        cbf_recommender.fit(800, 10)

        self.W_sparse = 0.2 * slim_recommender.W_sparse + 0.3 * rp3_recommender.W_sparse + 0.2 * cbf_recommender.W_sparse

    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False, return_scores = False):
        recoms = super().recommend(user_id_array, cutoff, remove_seen_flag, items_to_compute,
                  remove_top_pop_flag, remove_custom_items_flag, return_scores)

        recoms = list(recoms)
        if self.use_top_pop:
            if np.isscalar(user_id_array):
                user_id_array = np.atleast_1d(user_id_array)
                single_user = True
            else:
                single_user = False

            for i in range(len(recoms)):
                user_id = user_id_array[i]
                start_pos = self.URM_train.indptr[user_id]
                end_pos = self.URM_train.indptr[user_id + 1]
                if start_pos == end_pos:
                    recoms[i] = self.top_popular
        recoms = tuple(recoms)

        return recoms