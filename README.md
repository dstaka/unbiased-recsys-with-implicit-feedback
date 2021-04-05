## Unbiased Recommendation System with implicit feedback by Causal Inference approach
---

### About
This repository is to manage code of my private project which is presented at the 18th School of Computing Term Project Showcase (STePS) of National University of Singapore (NUS).  
The project is inspired by these academic researches published by [Yuta Saito](https://usaito.github.io/). Also, experimentation code in the repository is based on his code in GitHub.
- "**Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback**", which has been accepted at [WSDM'20](http://www.wsdm-conference.org/2020/index.php). ([_code in GitHub_](https://github.com/usaito/unbiased-implicit-rec-real))
- "**Unbiased Pairwise Learning from Biased Implicit Feedback**", which has been accepted by [ICTIR'20](https://ictir2020.org/). ([_code in GitHub_](https://github.com/usaito/unbiased-pairwise-rec))
- "**Asymmetric Tri-training for Debiasing Missing-Not-At-Random Explicit Feedback**", which has been accepted at [SIGIR2020](https://sigir.org/sigir2020/) as a full paper. ([_code in GitHub_](https://github.com/usaito/asymmetric-tri-rec-real))
If you find this project interesting, please refer his original papers and codes in his GitHub.

### Dependencies

- numpy==1.17.2
- pandas==0.25.1
- scikit-learn==0.22.1
- tensorflow==1.15.2
- optuna==0.17.0
- mlflow==1.7.0
- pyyaml==5.1.2

### Running the code

To run the simulation with real-world datasets,

1. download the Coat dataset from [https://www.cs.cornell.edu/~schnabts/mnar/](https://www.cs.cornell.edu/~schnabts/mnar/) and put `train.ascii` and `test.ascii` files into `./data/coat/` directory.
2. download the Yahoo! R3 dataset from [https://webscope.sandbox.yahoo.com/catalog.php?datatype=r](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r) and put `train.txt` and `test.txt` files into `./data/yahoo/` directory.

Then, run the following commands in the `./src/` directory:
- for Matrix Factorization models **without** *Inverse Propensity Scoring* and *asymmetric tri-training*
```bash
for data in yahoo coat
do
  for model in uniform-without_ipw user-without_ipw item-without_ipw both-without_ipw nb-without_ipw nb_true-without_ipw
  do
    python main.py -d $data -m $model
  done
done
```

- for Matrix Factorization with *Inverse Propensity Scoring* models **without** *asymmetric tri-training*
```bash
for data in yahoo coat
do
  for model in uniform user item both nb nb_true
  do
    python main.py -d $data -m $model
  done
done
```

- for Matrix Factorization with *Inverse Propensity Scoring* models **with** *asymmetric tri-training*
```bash
for data in coat yahoo
do
  for model in uniform-at user-at item-at both-at nb-at nb_true-at
  do
    python main.py -d $data -m $model
  done
done
```
where (uniform, user, item, both, nb, nb_true) correspond to (*uniform propenisty*, *user propensity*, *item propensity*, *user-item propensity*, *NB (uniform)*, *NB (true)*), respectively.

These commands will run simulations with real-world datasets conducted in Section 5.
The tuned hyperparameters for all models can be found in `./hyper_params.yaml`. <br>
(By adding the `-t` option to the above code, you can re-run the hyperparameter tuning procedure by *Optuna*.)
