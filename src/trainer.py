"""
This code is based on these codebases associated with Yuta Saito's research.
- Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback: https://github.com/usaito/unbiased-implicit-rec-real
- Unbiased Pairwise Learning from Biased Implicit Feedback: https://github.com/usaito/unbiased-pairwise-rec
- Asymmetric Tri-training for Debiasing Missing-Not-At-Random Explicit Feedback: https://github.com/usaito/asymmetric-tri-rec-real
"""
import time
import yaml
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.trial import Trial
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.python.framework import ops

from evaluate.evaluator import aoa_evaluator_all_biases
from utils.preprocessor import preprocess_datasets
from models.models import PointwiseRecommender as MFMODEL

rand_seed_val=12345
rand_seed_val_tpe=456
round_digit=4


def estimate_pscore(train: np.ndarray, train_mcar: np.ndarray,
                    val: np.ndarray, model_name: str) -> Tuple:
    """Estimate propensity score"""
    # Cast to integer to avoid an error
    train = train.astype(int)
    val = val.astype(int)

    # pscore = No. of items (I think it's equivalent to Relative item click popularity)
    if 'item' in model_name:
        pscore = np.unique(train[:, 1], return_counts=True)[1]
        # print(pscore.shape)
        # print(train.shape)
        pscore = pscore / pscore.max()
        pscore_train = pscore[train[:, 1]]
        pscore_val = pscore[val[:, 1]]

    # pscore = No. of users
    elif 'user' in model_name:
        pscore = np.unique(train[:, 0], return_counts=True)[1]
        pscore = pscore / pscore.max()
        pscore_train = pscore[train[:, 0]]
        pscore_val = pscore[val[:, 0]]

    # user-item propensity
    elif 'both' in model_name:
        user_pscore = np.unique(train[:, 0], return_counts=True)[1]
        user_pscore = user_pscore / user_pscore.max()
        item_pscore = np.unique(train[:, 1], return_counts=True)[1]
        item_pscore = item_pscore / item_pscore.max()
        pscore_train = user_pscore[train[:, 0]] * item_pscore[train[:, 1]]
        pscore_val = user_pscore[val[:, 0]] * item_pscore[val[:, 1]]

    elif 'true' in model_name:
        pscore = np.unique(train[:, 2], return_counts=True)[1] / np.unique(train_mcar[:, 2], return_counts=True)[1]
        pscore = pscore / pscore.max()
        pscore_train = pscore[train[:, 2] - 1]
        pscore_val = pscore[val[:, 2] - 1]

    elif 'nb' in model_name:
        pscore = np.unique(train[:, 2], return_counts=True)[1]
        pscore = pscore / pscore.max()
        pscore_train = pscore[train[:, 2] - 1]
        pscore_val = pscore[val[:, 2] - 1]

    # uniform propensity
    elif 'uniform' in model_name:
        pscore_train = np.ones(train.shape[0])
        pscore_val = np.ones(val.shape[0])

    # To be tested another way to estimate propensity score
    else:
        pass

    pscore_train = np.expand_dims(pscore_train, 1)
    pscore_val = np.expand_dims(pscore_val, 1)
    return pscore_train, pscore_val


# Train MF model without IPW in order to use it as a baseline result
def train_mfmodel_without_ipw(sess: tf.Session, model: MFMODEL, data: str,
                train: np.ndarray, val: np.ndarray, test: np.ndarray,
                max_iters: int = 500, batch_size: int = 2**9,
                model_name: str = 'mf', seed: int = 0) -> Tuple:
    """Train and evaluate the MF-IPS model."""
    train_loss_list = []
    val_loss_list = []
    test_mse_list = []
    test_mae_list = []

    # Initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Count the num of training data and estimate the propensity scores
    num_train = train.shape[0]
    train_mcar, test = train_test_split(test, test_size=0.95, random_state=rand_seed_val)

    labels_train = np.expand_dims(train[:, 2], 1)
    labels_val = np.expand_dims(val[:, 2], 1)
    labels_test = np.expand_dims(test[:, 2], 1)

    # Start training a recommender
    np.random.seed(rand_seed_val)
    for iter_ in np.arange(max_iters):
        # Sample mini-batch
        idx = np.random.choice(np.arange(num_train), size=batch_size)
        train_batch, labels_batch = train[idx], labels_train[idx]
        # Update user-item latent factors
        _, loss, wmse = sess.run([model.apply_grads, model.loss, model.weighted_mse],
                           feed_dict={model.users: train_batch[:, 0],
                                      model.items: train_batch[:, 1],
                                      model.labels: labels_batch,
                                      model.scores: np.ones((np.int(batch_size), 1)) # We just use 1 as propensity score for all records
                                      }
                                )
        # print('train_loss:', loss, wmse)
        train_loss_list.append(loss)
        # Calculate validation loss
        val_loss = sess.run(model.loss,
                            feed_dict={model.users: val[:, 0],
                                       model.items: val[:, 1],
                                       model.labels: labels_val,
                                       model.scores: np.ones((np.int(len(labels_val)), 1)) # We just use 1 as propensity score for all records
                                       }
                            )
        # print('val_loss:', val_loss)
        val_loss_list.append(val_loss)
        # Calculate test loss
        mse_score, mae_score = sess.run([model.mse, model.mae],
                                        feed_dict={model.users: test[:, 0],
                                                   model.items: test[:, 1],
                                                   model.labels: labels_test})
        # mse_score = round(mse_score, round_digit)
        # mae_score = round(mae_score, round_digit)
        # print('mse_score:', mse_score)
        # print('mae_score:', mae_score)
        test_mse_list.append(mse_score)
        test_mae_list.append(mae_score)

    u_emb, i_emb, u_bias, i_bias, g_bias = sess.run([
                                                model.user_embeddings,
                                                model.item_embeddings,
                                                model.user_bias,
                                                model.item_bias,
                                                model.global_bias
                                                ])

    sess.close()

    return (np.min(val_loss_list),
            test_mse_list[np.argmin(val_loss_list)],
            test_mae_list[np.argmin(val_loss_list)],
            u_emb, i_emb, u_bias, i_bias, g_bias)


# Train MF model with IPW (and wihtout Asymmetric Tri-training) in order to use as a baseline result
def train_mfmodel(sess: tf.Session, model: MFMODEL, data: str,
                train: np.ndarray, val: np.ndarray, test: np.ndarray,
                max_iters: int = 500, batch_size: int = 2**9,
                model_name: str = 'mf', seed: int = 0) -> Tuple:
    """Train and evaluate the MF-IPS model"""
    train_loss_list = []
    val_loss_list = []
    test_mse_list = []
    test_mae_list = []

    # Initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Count the num of training data and estimate the propensity scores
    num_train = train.shape[0]
    train_mcar, test = train_test_split(test, test_size=0.95, random_state=rand_seed_val)
    pscore_train, pscore_val = estimate_pscore(train=train, train_mcar=train_mcar,
                                               val=val, model_name=model_name)
    labels_train = np.expand_dims(train[:, 2], 1)
    labels_val = np.expand_dims(val[:, 2], 1)
    labels_test = np.expand_dims(test[:, 2], 1)

    # Start training a recommender
    np.random.seed(rand_seed_val)
    for iter_ in np.arange(max_iters):
        # Sample mini-batch
        idx = np.random.choice(np.arange(num_train), size=batch_size)
        train_batch, pscore_batch, labels_batch = train[idx], pscore_train[idx], labels_train[idx]
        # print('pscore_batch', pscore_batch)
        # Update user-item latent factors
        _, loss, wmse = sess.run([model.apply_grads, model.loss, model.weighted_mse],
                           feed_dict={model.users: train_batch[:, 0],
                                      model.items: train_batch[:, 1],
                                      model.labels: labels_batch,
                                      model.scores: pscore_batch})
        # print('train_loss:', loss, wmse)
        train_loss_list.append(loss)
        # Calculate validation loss
        val_loss = sess.run(model.loss,
                            feed_dict={model.users: val[:, 0],
                                       model.items: val[:, 1],
                                       model.labels: labels_val,
                                       model.scores: pscore_val})
        # print('val_loss:', val_loss)
        val_loss_list.append(val_loss)
        # Calculate test loss
        mse_score, mae_score = sess.run([model.mse, model.mae],
                                        feed_dict={model.users: test[:, 0],
                                                   model.items: test[:, 1],
                                                   model.labels: labels_test})
        # mse_score = round(mse_score, round_digit)
        # mae_score = round(mae_score, round_digit)
        # print('mse_score:', mse_score)
        # print('mae_score:', mae_score)
        test_mse_list.append(mse_score)
        test_mae_list.append(mae_score)

    u_emb, i_emb, u_bias, i_bias, g_bias = sess.run([
                                                model.user_embeddings,
                                                model.item_embeddings,
                                                model.user_bias,
                                                model.item_bias,
                                                model.global_bias
                                                ])

    sess.close()

    return (np.min(val_loss_list),
            test_mse_list[np.argmin(val_loss_list)],
            test_mae_list[np.argmin(val_loss_list)],
             u_emb, i_emb, u_bias, i_bias, g_bias)


# Train MF model with IPW and Asymmetric Tri-training in order to use as a result of a proposed method
def train_mfmodel_with_at(sess: tf.Session,
                        model: MFMODEL, mfmodel1: MFMODEL, mfmodel2: MFMODEL, data: str,
                        train: np.ndarray, val: np.ndarray, test: np.ndarray,
                        epsilon: float,
                        pre_iters: int = 500,
                        post_iters: int = 50,
                        post_steps: int = 5,
                        batch_size: int = 2**9,
                        model_name: str = 'naive-at',
                        seed: int = 0) -> Tuple:
    """Train and evaluate the MF-IPS model with asymmetric tri-training"""
    train_loss_list = []
    val_loss_list = []
    test_mse_list = []
    test_mae_list = []

    # Initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Count the num of training data and estimate the propensity scores
    num_train = train.shape[0]
    train_mcar, test = train_test_split(test, test_size=0.95, random_state=rand_seed_val)
    pscore_train, pscore_val = estimate_pscore(train=train, train_mcar=train_mcar,
                                               val=val, model_name=model_name)
    labels_train = np.expand_dims(train[:, 2], 1)
    labels_val = np.expand_dims(val[:, 2], 1)
    labels_test = np.expand_dims(test[:, 2], 1)
    pscore_model_all_1 = np.ones((batch_size, 1))

    ### Start training a recommender
    np.random.seed(rand_seed_val)
    ## Start pre-training step
    for i in np.arange(pre_iters):
        # Sample mini-batch
        idx = np.random.choice(np.arange(num_train), size=batch_size)
        idx1 = np.random.choice(np.arange(num_train), size=batch_size)
        idx2 = np.random.choice(np.arange(num_train), size=batch_size)
        train_batch, train_batch1, train_batch2 = train[idx], train[idx1], train[idx2]
        labels_batch, labels_batch1, labels_batch2 = labels_train[idx], labels_train[idx1], labels_train[idx2]
        pscore_batch1, pscore_batch2 = pscore_train[idx1], pscore_train[idx2]
        # print('pscore_batch1', pscore_batch1)
        # print('pscore_batch2', pscore_batch2)
        # Update user-item latent factors
        _, train_loss, train_wmse = sess.run([model.apply_grads, model.loss, model.weighted_mse],
                                 feed_dict={model.users: train_batch[:, 0],
                                            model.items: train_batch[:, 1],
                                            model.labels: labels_batch,
                                            model.scores: pscore_model_all_1})
        _, mfmodel1_loss, mfmodel1_wmse = sess.run(
                    [mfmodel1.apply_grads, mfmodel1.loss, mfmodel1.weighted_mse],
                     feed_dict={mfmodel1.users: train_batch1[:, 0],
                                mfmodel1.items: train_batch1[:, 1],
                                mfmodel1.labels: labels_batch1,
                                mfmodel1.scores: pscore_batch1})
        _, mfmodel2_loss, mfmodel2_wmse = sess.run(
                    [mfmodel2.apply_grads, mfmodel2.loss, mfmodel2.weighted_mse],
                     feed_dict={mfmodel2.users: train_batch2[:, 0],
                                mfmodel2.items: train_batch2[:, 1],
                                mfmodel2.labels: labels_batch2,
                                mfmodel2.scores: pscore_batch2})
        # print('train_loss:', train_loss, train_wmse)
        # print('mfmodel1_loss:', mfmodel1_loss, mfmodel1_wmse)
        # print('mfmodel2_loss:', mfmodel2_loss, mfmodel2_wmse)
        # print()

    ## Start psuedo-labeling and final prediction steps
    # Cast to integer to avoid an error
    train = train.astype(int)
    val = val.astype(int)

    all_data = pd.DataFrame(np.zeros((train[:, 0].max() + 1, train[:, 1].max() + 1)))
    all_data = all_data.stack().reset_index().values[:, :2]
    for k in np.arange(post_iters):
        for j in np.arange(post_steps):
            idx = np.random.choice(np.arange(all_data.shape[0]), size=num_train * 5)
            batch_data = all_data[idx]
            # Create psuedo-labeled dataset
            preds1 = sess.run(mfmodel1.preds,
                             feed_dict={mfmodel1.users: batch_data[:, 0],
                                        mfmodel1.items: batch_data[:, 1]})
            preds2 = sess.run(mfmodel2.preds,
                             feed_dict={mfmodel2.users: batch_data[:, 0],
                                        mfmodel2.items: batch_data[:, 1]})

            # Extract records whose prediction difference between model1 and model2 are less than or equal to epsilon
            idx = np.array(np.abs(preds1 - preds2) <= epsilon).flatten()
            # print(idx.sum())
            target_users, target_items, pseudo_labels = batch_data[idx, 0], batch_data[idx, 1], preds1[idx]
            target_data = np.c_[target_users, target_items, pseudo_labels]
            # Store information during the pseudo-labeleing step
            num_target = target_data.shape[0]
            # Sample mini-batch for the pseudo-labeleing step
            idx = np.random.choice(np.arange(num_target), size=batch_size)
            idx1 = np.random.choice(np.arange(num_target), size=batch_size)
            idx2 = np.random.choice(np.arange(num_target), size=batch_size)
            pseudo_train_batch, pseudo_train_batch1, pseudo_train_batch2 = target_data[idx], target_data[idx1], target_data[idx2]
            # Update user-item latent factors of the final prediction model
            _, train_loss = sess.run([model.apply_grads, model.loss],
                                     feed_dict={model.users: pseudo_train_batch[:, 0],
                                                model.items: pseudo_train_batch[:, 1],
                                                model.labels: np.expand_dims(pseudo_train_batch[:, 2], 1),
                                                model.scores: np.ones((np.int(batch_size), 1))
                                                })
            # print('train_loss:', train_loss)
            # Calculate validation loss during the psuedo-labeleing step
            val_loss = sess.run(model.loss,##model.weighted_mse,
                                feed_dict={model.users: val[:, 0],
                                           model.items: val[:, 1],
                                           model.scores: pscore_val,
                                           model.labels: labels_val})
            # print('val_loss:', val_loss)
            # Calculate test losses during the psuedo-labeleing step
            mse_score, mae_score = sess.run([model.mse, model.mae],
                                            feed_dict={model.users: test[:, 0],
                                                       model.items: test[:, 1],
                                                       model.labels: labels_test})
            # mse_score = round(mse_score, round_digit)
            # mae_score = round(mae_score, round_digit)
            # print('mse_score:', mse_score)
            # print('mae_score:', mae_score)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            test_mse_list.append(mse_score)
            test_mae_list.append(mae_score)
            # Re-update the model parameters of pre-trained models using pseudo-labeled data
            _ = sess.run(mfmodel1.apply_grads,
                         feed_dict={mfmodel1.users: pseudo_train_batch1[:, 0],
                                    mfmodel1.items: pseudo_train_batch1[:, 1],
                                    mfmodel1.labels: np.expand_dims(pseudo_train_batch1[:, 2], 1),
                                    mfmodel1.scores: np.ones((batch_size, 1))
                                    })
            _ = sess.run(mfmodel2.apply_grads,
                         feed_dict={mfmodel2.users: pseudo_train_batch2[:, 0],
                                    mfmodel2.items: pseudo_train_batch2[:, 1],
                                    mfmodel2.labels: np.expand_dims(pseudo_train_batch2[:, 2], 1),
                                    mfmodel2.scores: np.ones((batch_size, 1))
                                    })

    # Obtain user-item embeddings
    u_emb, i_emb, u_bias, i_bias, g_bias = sess.run([
                                                model.user_embeddings,
                                                model.item_embeddings,
                                                model.user_bias,
                                                model.item_bias,
                                                model.global_bias
                                                ])

    sess.close()

    return (np.min(val_loss_list),
            test_mse_list[np.argmin(val_loss_list)],
            test_mae_list[np.argmin(val_loss_list)],
            u_emb, i_emb, u_bias, i_bias, g_bias)


class Objective:

    def __init__(self, data: str, model_name: str = 'mf') -> None:
        """Initialize Class"""
        self.data = data
        self.model_name = model_name

    def __call__(self, trial: Trial) -> float:
        """Calculate an objective value."""
        train, val, test, num_users, num_items =\
            preprocess_datasets(data=self.data, seed=rand_seed_val)

        # sample a set of hyperparameters.
        config = yaml.safe_load(open('../config.yaml', 'r'))
        eta = config['eta']
        max_iters = config['max_iters']
        batch_size = config['batch_size']
        pre_iters = config['pre_iters']
        post_iters = config['post_iters']
        post_steps = config['post_steps']
        dim = trial.suggest_discrete_uniform('dim', 5, 50, 5)
        lam = trial.suggest_loguniform('lam', 1e-6, 1)
        if '-at' in self.model_name:
            epsilon = trial.suggest_loguniform('epsilon', 1e-3, 1)

        ops.reset_default_graph()
        tf.set_random_seed(rand_seed_val)
        sess = tf.Session()
        if '-without_ipw' in self.model_name:
            print('*** Without IPW ***')
            model = MFMODEL(num_users=num_users, num_items=num_items, dim=dim, lam=lam, eta=eta)
            score, _, _, _, _, _, _, _ = train_mfmodel_without_ipw(
                sess, model=model, data=self.data, train=train, val=val, test=test,
                max_iters=max_iters, batch_size=batch_size, model_name=self.model_name)
        elif '-at' not in self.model_name:
            print('*** With IPW and without Asymmetric Tri-training ***')
            model = MFMODEL(num_users=num_users, num_items=num_items, dim=dim, lam=lam, eta=eta)
            score, _, _, _, _, _, _, _ = train_mfmodel(
                sess, model=model, data=self.data, train=train, val=val, test=test,
                max_iters=max_iters, batch_size=batch_size, model_name=self.model_name)
        else:
            print('*** With IPW and Asymmetric Tri-training ***')
            model = MFMODEL(num_users=num_users, num_items=num_items, dim=dim, lam=lam, eta=eta, num=0)
            model1 = MFMODEL(num_users=num_users, num_items=num_items, dim=dim, lam=lam, eta=eta, num=1)
            model2 = MFMODEL(num_users=num_users, num_items=num_items, dim=dim, lam=lam, eta=eta, num=2)
            score, _, _, _, _, _, _, _ = train_mfmodel_with_at(
                sess, model=model, mfmodel1=model1, mfmodel2=model2, data=self.data,
                train=train, val=val, test=test, epsilon=epsilon,
                pre_iters=pre_iters, post_iters=post_iters, post_steps=post_steps,
                batch_size=batch_size, model_name=self.model_name)

        return score


class Tuner:
    """Class for tuning hyperparameter of MF models."""

    def __init__(self, data: str, model_name: str) -> None:
        """Initialize Class."""
        self.data = data
        self.model_name = model_name

    def tune(self, n_trials: int = 30) -> None:
        """Hyperparameter Tuning by TPE."""
        path = Path(f'../logs/{self.data}')
        path.mkdir(exist_ok=True, parents=True)
        path_model = Path(f'../logs/{self.data}/{self.model_name}')
        path_model.mkdir(exist_ok=True, parents=True)
        # Tune hyperparameters by Optuna
        objective = Objective(data=self.data, model_name=self.model_name)
        study = optuna.create_study(sampler=TPESampler(seed=rand_seed_val_tpe))
        study.optimize(objective, n_trials=n_trials)
        # Save tuning results as a YAML file
        study.trials_dataframe().to_csv(str(path_model / f'tuning_results.csv'))
        if Path('../hyper_params.yaml').exists():
            pass
        else:
            yaml.dump(dict(yahoo=dict(), coat=dict()),
                      open('../hyper_params.yaml', 'w'), default_flow_style=False)
        time.sleep(np.random.rand())
        hyper_params_dict = yaml.safe_load(open('../hyper_params.yaml', 'r'))
        hyper_params_dict[self.data][self.model_name] = study.best_params
        yaml.dump(hyper_params_dict, open('../hyper_params.yaml', 'w'), default_flow_style=False)


class Trainer:

    def __init__(self, data: str, batch_size: int = 10, eta: float = 0.01,
                 max_iters: int = 500, pre_iters: int = 150, post_iters: int = 10,
                 post_steps: int = 10, model_name: str = 'mf') -> None:
        """Initialize class"""
        self.data = data
        hyper_params = yaml.safe_load(open(f'../hyper_params.yaml', 'r'))[data][model_name]
        self.dim = hyper_params['dim']
        self.lam = hyper_params['lam']
        # If model name includes "-at", it's Asymmetric Tri-training approach
        if '-at' in model_name:
            self.epsilon = hyper_params['epsilon']
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.pre_iters = pre_iters
        self.post_iters = post_iters
        self.post_steps = post_steps
        self.eta = eta
        self.model_name = model_name

    def run_simulations(self, num_sims: int = 5) -> None:
        """Train mf"""
        results_mse = []
        results_mae = []
        results_ndcg_at1, results_ndcg_at3, results_ndcg_at5 = [], [], []
        results_recall_at1, results_recall_at3, results_recall_at5 = [], [], []
        results_map_at1, results_map_at3, results_map_at5 = [], [], []

        # start running simulations
        start = time.time()
        for seed in np.arange(num_sims):
            train, val, test, num_users, num_items =\
                preprocess_datasets(data=self.data, seed=seed)

            ops.reset_default_graph()
            tf.set_random_seed(seed)
            sess = tf.Session()
            if '-without_ipw' in self.model_name:
                print('*** Without IPW ***')
                model = MFMODEL(num_users=num_users, num_items=num_items,
                              dim=self.dim, lam=self.lam, eta=self.eta)
                _, mse, mae, u_emb, i_emb, u_bias, i_bias, g_bias = train_mfmodel_without_ipw(
                    sess, model=model, data=self.data, train=train, val=val, test=test,
                    max_iters=self.max_iters, batch_size=self.batch_size,
                    model_name=self.model_name, seed=seed)
            elif '-at' not in self.model_name:
                print('*** With IPW and without Asymmetric Tri-training ***')
                model = MFMODEL(num_users=num_users, num_items=num_items,
                              dim=self.dim, lam=self.lam, eta=self.eta)
                _, mse, mae, u_emb, i_emb, u_bias, i_bias, g_bias = train_mfmodel(
                    sess, model=model, data=self.data, train=train, val=val, test=test,
                    max_iters=self.max_iters, batch_size=self.batch_size,
                    model_name=self.model_name, seed=seed)
            else:
                print('*** With IPW and Asymmetric Tri-training ***')
                model = MFMODEL(num_users=num_users, num_items=num_items,
                              dim=self.dim, lam=self.lam, eta=self.eta, num=0)
                model1 = MFMODEL(num_users=num_users, num_items=num_items,
                               dim=self.dim, lam=self.lam, eta=self.eta, num=1)
                model2 = MFMODEL(num_users=num_users, num_items=num_items,
                               dim=self.dim, lam=self.lam, eta=self.eta, num=2)
                _, mse, mae, u_emb, i_emb, u_bias, i_bias, g_bias = train_mfmodel_with_at(
                    sess, model=model, mfmodel1=model1, mfmodel2=model2, data=self.data,
                    train=train, val=val, test=test, epsilon=self.epsilon,
                    pre_iters=self.pre_iters, post_iters=self.post_iters, post_steps=self.post_steps,
                    batch_size=self.batch_size, model_name=self.model_name, seed=seed)

            # After building a model, summarize metrics
            results_mae.append(mae)
            results_mse.append(mse)
            ranking_results_dic = aoa_evaluator_all_biases(
                                    user_embed=u_emb,
                                    item_embed=i_emb,
                                    user_bias=u_bias,
                                    item_bias=i_bias,
                                    global_bias=g_bias,
                                    test=test,
                                    model_name=self.model_name
                                    )

            results_ndcg_at1.append(ranking_results_dic['nDCG@1'])
            results_ndcg_at3.append(ranking_results_dic['nDCG@3'])
            results_ndcg_at5.append(ranking_results_dic['nDCG@5'])
            results_recall_at1.append(ranking_results_dic['Recall@1'])
            results_recall_at3.append(ranking_results_dic['Recall@3'])
            results_recall_at5.append(ranking_results_dic['Recall@5'])
            results_map_at1.append(ranking_results_dic['MAP@1'])
            results_map_at3.append(ranking_results_dic['MAP@3'])
            results_map_at5.append(ranking_results_dic['MAP@5'])
            print(f'#{seed+1} {self.model_name}: {np.round((time.time() - start) / 60, 2)} min')
        # Aggregate and save the final results
        result_path = Path(f'../logs/{self.data}/{self.model_name}')
        result_path.mkdir(parents=True, exist_ok=True)
        pd.concat([pd.DataFrame(results_mae, columns=['MAE']),
                   pd.DataFrame(results_mse, columns=['MSE']),
                   pd.DataFrame(results_ndcg_at1, columns=['nDCG@1']),
                   pd.DataFrame(results_ndcg_at3, columns=['nDCG@3']),
                   pd.DataFrame(results_ndcg_at5, columns=['nDCG@5']),
                   pd.DataFrame(results_recall_at1, columns=['Recall@1']),
                   pd.DataFrame(results_recall_at3, columns=['Recall@3']),
                   pd.DataFrame(results_recall_at5, columns=['Recall@5']),
                   pd.DataFrame(results_map_at1, columns=['MAP@1']),
                   pd.DataFrame(results_map_at3, columns=['MAP@3']),
                   pd.DataFrame(results_map_at5, columns=['MAP@5']),
                   ], 1
                   )\
            .to_csv(str(result_path / 'results.csv'))
