"""Utilities for loading trained neural CBF checkpoints and their associated param_dicts."""
import json
import pickle
import types

import torch

from src.neural_phi import NeuralPhi
from src.utils import load_model, TransformEucNNInput


def load_phi_and_params(exp_name: str, checkpoint_number: int):
    """Loads a trained NeuralPhi and its param_dict from a checkpoint.

    If exp_name is given, reads args and param_dict from the corresponding log
    directory; otherwise falls back to default values from the argument parser.

    Args:
        exp_name: Experiment name (subdirectory of ./log/).
        checkpoint_number: Which saved checkpoint to load weights from.

    Returns:
        phi_star_fn: NeuralPhi module on CPU with loaded weights.
        param_dict: Dictionary of problem parameters (x_dim, u_dim, x_lim, …).
    """
    device = torch.device("cpu")

    # Load args and param_dict from log directory
    args_path = "./log/%s/args.txt" % exp_name
    with open(args_path, "r") as f:
        json_data = json.load(f)
    args = types.SimpleNamespace(**json_data)
    param_dict = pickle.load(open("./log/%s/param_dict.pkl" % exp_name, "rb"))

    r = param_dict["r"]
    x_dim = param_dict["x_dim"]
    u_dim = param_dict["u_dim"]

    # Build dynamics and constraint modules
    from src.problems.quad_pend import Rho, XDot, ULimitSetVertices
    h_fn = Rho(param_dict)
    xdot_fn = XDot(param_dict, device)
    uvertices_fn = ULimitSetVertices(param_dict, device)

    h_fn = h_fn.to(device)
    xdot_fn = xdot_fn.to(device)
    uvertices_fn = uvertices_fn.to(device)

    # Optional: input transformation for Euclidean representation
    x_e = torch.zeros(1, x_dim).to(device)
    state_index_dict = param_dict["state_index_dict"]
    if getattr(args, "phi_nn_inputs", "spherical") == "euc":
        nn_input_modifier = TransformEucNNInput(state_index_dict)
    else:
        nn_input_modifier = None

    # Build the neural CBF
    phi_star_fn = NeuralPhi(
        h_fn, xdot_fn, r, x_dim, u_dim, device, args,
        x_e=x_e, nn_input_modifier=nn_input_modifier,
    )
    phi_star_fn = phi_star_fn.to(device)

    # Load weights
    phi_load_path = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
    load_model(phi_star_fn, phi_load_path)
    print("Loaded checkpoint: %s" % phi_load_path)
    print("  h = %s, ci = %s" % (phi_star_fn.h, phi_star_fn.ci))

    return phi_star_fn, param_dict
