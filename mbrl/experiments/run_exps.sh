env="mountain_car"
n_seeds=8

python -m mbrl.experiments.main --multirun hydra/launcher=joblib seed="range($n_seeds)" strategy=imed overrides=$env
python -m mbrl.experiments.main --multirun hydra/launcher=joblib seed="range($n_seeds)" strategy=count_based overrides=$env
python -m mbrl.experiments.main --multirun hydra/launcher=joblib seed="range($n_seeds)" strategy=thompson overrides=$env
python -m mbrl.experiments.main --multirun hydra/launcher=joblib seed="range($n_seeds)" strategy=epsilon_greedy overrides=$env