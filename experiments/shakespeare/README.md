# Evaluation: Shakespeare

This sub-project contains the Shakespeare experiments presented in the paper. 

## Requirements

### install requirements:

```setup
pip install -r requirements.txt
```

### Initialize git submodule

Run the following to make sure that the remote Google's [federated research repo](https://github.com/google-research/federated) is cloned as a submodule:

```setup
git submodule update --init --recursive
```

### Update PYTHONPATH

Add `experiments/federated/google_tff_research` to `PYTHONPATH`.

## Training

In order to reproduce the paper's results, execute `experiments/shakespeare/run_experiment.py` (the current working directory should be the repo's root).

You can view the documentation for every command line parameter using `experiments/shakespeare/run_experiment.py --help`.

We use the shared command line parameters and each combination of the preprocessing, aggregation, and attack parameters:

### Shared command line parameters 

```shell
--task=shakespeare --clients_per_round=10 --client_datasets_random_seed=1 --client_epochs_per_round=1 --total_rounds=1200 --client_batch_size=4 --shakespeare_sequence_length=80 --client_optimizer=sgd --client_learning_rate=1 --server_optimizer=sgd --server_learning_rate=1 --server_sgd_momentum=0.0
```

Additionally, set the shared output directory for all experiments using:
```shell
--root_output_dir=<your_output_directory>
```

### Preprocessing command line parameter

The `--weight_preproc` parameter determines the type of the preprocessing we use and expects either `passthrough`, `ignore`, or `truncate`.

### Aggregation command line parameter

The `--aggregation` parameter determines the type of the aggregation we use and expects either `mean`, `trimmed_mean`, or `median`.

### Attack command line parameters

We have execute all previous experiment three time:
1. Without attack: no additional parameter needed.
2. 10% precent attackers: add the following `--attack=delta_to_zero --num_byzantine=10_percent --byzantine_client_weight=1_000_000`.
3. A single attacker: add the following `--attack=delta_to_zero --num_byzantine=single --byzantine_client_weight=10_000_000`. 

### Set each experiment name

In order to use our plotting script without change, make sure to name each experiment using the following pattern `--experiment_name=shakespeare_{aggregation}_{weight_preproc}{attack}`, where:
* `weight_preproc` corresponds to `passthrough`, `ignore`, or `truncate`.
* `aggregation` corresponds to `mean`, `tmean` (trimmed mean), or `median`.
* `attack` corresponds to `​` (empty string, no attack), `_byz_d0` (10% attackers), or `_byz_d0_single` (a single attacker).

## Results

You can monitor the progress using TensorBoard:

```setup
tensorboard --logdir <root_output_dir>/logdir
```

After the experiments are done execute `plots.ipynb` using [Jupyter](https://jupyter.org/) to re-create the Shakespeare experiment figures from the paper. 

Run `plot_ds_hist.py` to recreate the sample distribution histogram.