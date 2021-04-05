"""
This code is based on these codebases associated with Yuta Saito's research.
- Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback: https://github.com/usaito/unbiased-implicit-rec-real
- Unbiased Pairwise Learning from Biased Implicit Feedback: https://github.com/usaito/unbiased-pairwise-rec
- Asymmetric Tri-training for Debiasing Missing-Not-At-Random Explicit Feedback: https://github.com/usaito/asymmetric-tri-rec-real
"""
import argparse
import yaml
import warnings
import tensorflow as tf
from trainer import Trainer, Tuner

possible_model_names = ['uniform-without_ipw', 'uniform', 'uniform-at',
                        'user-without_ipw', 'user', 'user-at',
                        'item-without_ipw', 'item', 'item-at',
                        'both-without_ipw', 'both', 'both-at',
                        'nb-without_ipw', 'nb', 'nb-at',
                        'nb_true-without_ipw', 'nb_true', 'nb_true-at'
                        ]

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', type=str, required=True)
parser.add_argument('--model_name', '-m', type=str,
                    choices=possible_model_names, required=True)
parser.add_argument('--tuning', '-t', action='store_true')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()

    # Set hyperparameters
    config = yaml.safe_load(open('../config.yaml', 'r'))
    eta = config['eta']
    batch_size = config['batch_size']
    max_iters = config['max_iters']
    pre_iters = config['pre_iters']
    post_steps = config['post_steps']
    post_iters = config['post_iters']
    num_sims = config['num_sims']
    n_trials = config['n_trials']
    model_name = args.model_name
    tuning = args.tuning
    data = args.data

    if tuning:
        tuner = Tuner(data=data, model_name=model_name)
        tuner.tune(n_trials=n_trials)

        print('\n', '=' * 25, '\n')
        print(f'Finished Tuning of {model_name}!')
        print('\n', '=' * 25, '\n')

    trainer = Trainer(data=data, batch_size=batch_size, max_iters=max_iters,
                      pre_iters=pre_iters, post_steps=post_steps,
                      post_iters=post_iters, eta=eta, model_name=model_name)
    trainer.run_simulations(num_sims=num_sims)

    print('\n', '=' * 25, '\n')
    print(f'Finished Running {model_name}!')
    print('\n', '=' * 25, '\n')
