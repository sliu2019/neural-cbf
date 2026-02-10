# Neural CBF Synthesis with Input Saturation

Code for **"Safe control under input limits with neural control barrier functions"**
(Liu et al., CoRL 2022).

## Overview

This repository synthesizes neural Control Barrier Functions (CBFs) for a
quadcopter-pendulum system subject to control input saturation. The core idea
is a modified CBF construction φ*(x) that certifies safety even when actuators
saturate, trained via a learner-critic loop.

**Key contribution:** Standard CBF synthesis ignores input limits. This work reformulates
the forward invariance condition to explicitly account for saturation, then parameterizes
φ*(x) using a neural network trained to eliminate worst-case violations.

## Method

Training alternates between two phases (Algorithm 1):

- **Critic:** Finds states x on the CBF boundary (φ=0) where input saturation causes
  safety violations, using projected gradient ascent constrained to the boundary manifold
- **Learner:** Updates the neural CBF to minimize saturation risk at those states while
  maximizing safe set volume via regularization (Eq. 4)

The CBF has the form:

```
φ*(x) = [Π_{i=0}^{r-1} (1 + c_i d^i/dt^i)] ρ(x) + (nn(x) - nn(x_e))² + h·ρ(x)
```

where ρ(x) is the base safety specification and nn(x) is a learned 64-64 tanh MLP.

## Requirements

```
torch
numpy
```

## Usage

```bash
python main.py --problem quad_pend
```

Key hyperparameters:

| Argument | Default | Description |
|---|---|---|
| `--learner_n_steps` | 3000 | Training iterations |
| `--reg_weight` | 150.0 | Safe set volume regularization weight |
| `--critic_n_samples` | 50 | Counterexample batch size |
| `--critic_max_n_steps` | 20 | Gradient ascent steps per critic call |
| `--learner_lr` | 1e-3 | Adam learning rate |
| `--gpu` | 0 | CUDA device index |

Training logs are written to `log/<problem>_<affix>/` and model checkpoints to
`checkpoint/<problem>_<affix>/`.

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
```

## Citation

```bibtex
@inproceedings{liu2022safe,
  title={Safe control under input limits with neural control barrier functions},
  author={Liu, Simin and Zeng, Sicun and Sreenath, Koushil and Belta, Calin},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2022}
}
```
