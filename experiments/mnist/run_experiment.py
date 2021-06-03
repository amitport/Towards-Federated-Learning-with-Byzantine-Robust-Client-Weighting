import experiments.mnist.experiment_runner as experiment_runner

experiment_runner.run_no_attacks('expr_no_attacks', seed=1, cpr='all', rounds=100, mu=1.5, sigma=3.45, alpha=0.1,
                                 t_mean_beta=0.1)

experiment_runner.run_all('expr_to_zero_10_precent', seed=1, cpr='all', rounds=100, mu=1.5, sigma=3.45, real_alpha=0.1,
                          num_samples_per_attacker=
                          1_000_000, attack_type='delta_to_zero', alpha=0.1, t_mean_beta=0.1)

experiment_runner.run_all('expr_to_y_flip_10_precent', seed=1, cpr='all', rounds=100, mu=1.5, sigma=3.45,
                          real_alpha=0.1,
                          num_samples_per_attacker=1_000_000, attack_type='y_flip', alpha=0.1, t_mean_beta=0.1)

experiment_runner.run_all('expr_to_zero_single', seed=1, cpr='all', rounds=100, mu=1.5, sigma=3.45, real_alpha=0.1,
                          num_samples_per_attacker=10_000_000, attack_type='delta_to_zero', alpha=1, t_mean_beta=0.1,
                          real_alpha_as_f=True)

experiment_runner.run_all('expr_y_flip_single', seed=1, cpr='all', rounds=100, mu=1.5, sigma=3.45,
                          real_alpha=0.1,
                          num_samples_per_attacker=10_000_000, attack_type='y_flip', alpha=1, t_mean_beta=0.1,
                          real_alpha_as_f=True)
