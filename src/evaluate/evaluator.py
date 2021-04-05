"""
This code is based on these codebases associated with Yuta Saito's research.
- Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback: https://github.com/usaito/unbiased-implicit-rec-real
- Unbiased Pairwise Learning from Biased Implicit Feedback: https://github.com/usaito/unbiased-pairwise-rec
- Asymmetric Tri-training for Debiasing Missing-Not-At-Random Explicit Feedback: https://github.com/usaito/asymmetric-tri-rec-real
"""
from typing import List
import numpy as np
import pandas as pd
from evaluate.metrics import average_precision_at_k, dcg_at_k, recall_at_k
round_digit_ranking_metrics=4
at_k_list = [1, 3, 5]
metrics = {'nDCG': dcg_at_k,
           'Recall': recall_at_k,
           'MAP': average_precision_at_k}


class PredictRankingsAllBiases:
    """Predict rankings by trained recommendations"""

    def __init__(self,
                user_embed: np.ndarray,
                item_embed: np.ndarray,
                user_bias: np.ndarray,
                item_bias: np.ndarray,
                global_bias: np.ndarray
                ) -> None:
        """Initialize Class."""
        self.user_embed = user_embed
        self.item_embed = item_embed
        self.user_bias = user_bias
        self.item_bias = item_bias
        self.global_bias = global_bias

    def predict(self, users: np.array, items: np.array) -> np.ndarray:
        """Predict scores for each user-item pairs"""
        # Predict ranking score for each user
        # print("self.user_embed[users]:", self.user_embed[users].shape)
        # print("self.user_embed.shape[1]:", self.user_embed.shape[1])
        user_emb = self.user_embed[users].reshape(1, self.user_embed.shape[1])
        item_emb = self.item_embed[items]
        scores = (user_emb @ item_emb.T).flatten() + self.item_bias[items] + self.user_bias[users] + self.global_bias
        return scores


def aoa_evaluator_all_biases(
                user_embed: np.ndarray,
                item_embed: np.ndarray,
                user_bias: np.ndarray,
                item_bias: np.ndarray,
                global_bias: np.ndarray,
                test: np.ndarray,
                model_name: str,
                at_k: List[int] = at_k_list) -> dict:
    """Calculate ranking metrics with average-over-all evaluator."""
    # Extract records from test data
    users = test[:, 0]
    items = test[:, 1]
    relevances = test[:, 2]

    # Define a predictive model
    model = PredictRankingsAllBiases(
        user_embed=user_embed,
        item_embed=item_embed,
        user_bias=user_bias,
        item_bias=item_bias,
        global_bias=global_bias
    )

    # Prepare ranking metrics
    results = {}
    for k in at_k:
        for metric in metrics:
            results[f'{metric}@{k}'] = []

    # Calculate ranking metrics
    ranking_results_dic = {}
    np.random.seed(12345)
    for user in set(users):
        indices = users == user
        pos_items = items[indices]
        rel = relevances[indices]

        # Predict ranking score for each user
        scores = model.predict(users=user, items=pos_items)
        for k in at_k:
            for metric, metric_func in metrics.items():
                results[f'{metric}@{k}'].append(metric_func(rel, scores, k))

        # Aggregate results
        results_df = pd.DataFrame(index=results.keys())
        results_df[model_name] = list(map(np.mean, list(results.values())))

        ranking_results_dic = {}
        for _idx in range(len(results_df.index)):
        #     print(_idx)
        #     print(results_df.iloc[_idx][0])
            ranking_results_dic[results_df.index[_idx]] = round(results_df.iloc[_idx][0], round_digit_ranking_metrics)
    return ranking_results_dic
