# Safe Control under Input Limits with Neural Control Barrier Functions

**Simin Liu, Sicun Gao, Koushil Sreenath, Calin Belta** | CoRL 2022

[Paper](https://proceedings.mlr.press/v205/liu23e.html)

<p align="center">
  <img src="assets/teaser.png" width="80%">
</p>

---

## Quickstart

**Clone the repository**
```bash
git clone https://github.com/liusimin/cbf_synthesis_small.git
cd cbf_synthesis_small
```

**Create and activate the Conda environment**
```bash
conda create -n cbf python=3.9
conda activate cbf
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run training**
```bash
python main.py --affix example
```

Logs are written to `log/quad_pend_example/` and checkpoints to `checkpoint/quad_pend_example/`.

Key training arguments:

| Argument | Default | Description |
|---|---|---|
| `--affix` | `default` | Suffix for experiment folder names |
| `--learner_n_steps` | 3000 | Training iterations |
| `--reg_weight` | 150.0 | Safe-set volume regularization weight |
| `--critic_n_samples` | 50 | Counterexample batch size per critic step |
| `--critic_max_n_steps` | 20 | Gradient ascent steps per critic call |
| `--learner_lr` | 1e-3 | Adam learning rate |
| `--gpu` | 0 | CUDA device index |

---

## Repository Structure

```
src/
├── neural_phi.py        # Neural CBF φ*(x): architecture and forward pass
├── critic.py            # Counterexample search via projected gradient ascent
├── learner.py           # Training loop (Algorithm 1)
├── reg_sampler.py       # Rejection sampler for volume regularization
├── utils.py             # Logging, checkpointing, early stopping, coord transforms
├── create_arg_parser.py # All hyperparameter definitions and defaults
└── problems/
    └── quad_pend.py     # Quadcopter-pendulum dynamics, ρ(x), and control polytope
main.py                  # Entry point: sets up all modules and calls learner.train()
quad_pend_analysis/      # Post-hoc evaluation: CBF slices, rollouts, volume estimates
```

---

## Citation

If you found this useful, please cite:
```bibtex
@inproceedings{liu2022safe,
  title={Safe control under input limits with neural control barrier functions},
  author={Liu, Simin and Zeng, Sicun and Sreenath, Koushil and Belta, Calin},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2022}
}
```
