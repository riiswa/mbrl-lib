# IMED-ARL: Indexed Empirical Minimum Divergence in Approximated Reinforcement Learning setting

## How to install

```bash
python3 -m venv venv
source venv/bin/activate 
pip install pip==24.0  # Necessary for the omegaconf package that have invalid metadata in the latest version of pip
pip install -e ".[dev]"
```

## Tasks

### Discrete actions 

- [ ] Discrete actions support: Create `mbrl/algorithms/mbpo-dqn.py` by replacing the SAC implementation (`mbrl/third_party/pytorch_sac_pranz24`) by DQN with extensions.
- [ ] Implement discrete actions exploration strategies: Count-Based, epsilon-greedy and IMED.
- [ ] Benchmark Classic Control environments: Mountain Car, Acrobot, Cart Pole.
- [ ] Benchmark Highway environments with discrete meta-actions: Highway, Merge, Roundabout, Intersection.

### Continous actions

- [ ] Implement MEEE action generation: https://arxiv.org/pdf/2107.01825
- [ ] Implement continous actions exploration strategies: MEEE, IMED
- [ ] Benchmark on 3 Point Maze environment with increasing gaussian noise.
