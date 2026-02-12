import json
import pickle
import types

import torch

from src.neural_phi import NeuralPhi
from src.problems.quad_pend import Rho, XDot, ULimitSetVertices
from src.utils import load_model, TransformEucNNInput


def load_phi_and_params(exp_name: str, checkpoint_number: int):
    """Loads a trained NeuralPhi and its param_dict from a checkpoint.

    Args:
        exp_name: Experiment name (subdirectory of ./log/).
        checkpoint_number: Which saved checkpoint to load weights from.

    Returns:
        phi_star_fn: NeuralPhi module on CPU with loaded weights.
        param_dict: Dictionary of problem parameters (x_dim, u_dim, x_lim, …).
    """
    device = torch.device("cpu")

    with open("./log/%s/args.txt" % exp_name, "r") as f:
        args = types.SimpleNamespace(**json.load(f))
    with open("./log/%s/param_dict.pkl" % exp_name, "rb") as f:
        param_dict = pickle.load(f)

    r     = param_dict["r"]
    x_dim = param_dict["x_dim"]
    u_dim = param_dict["u_dim"]

    h_fn        = Rho(param_dict).to(device)
    xdot_fn     = XDot(param_dict, device).to(device)
    uvertices_fn = ULimitSetVertices(param_dict, device).to(device)

    # Optional input transformation (Euclidean vs. spherical coordinates)
    x_e = torch.zeros(1, x_dim).to(device)
    if getattr(args, "phi_nn_inputs", "spherical") == "euc":
        nn_input_modifier = TransformEucNNInput(param_dict["state_index_dict"])
    else:
        nn_input_modifier = None

    phi_star_fn = NeuralPhi(
        h_fn, xdot_fn, r, x_dim, u_dim, device, args,
        x_e=x_e, nn_input_modifier=nn_input_modifier,
    ).to(device)

    phi_load_path = "./checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
    load_model(phi_star_fn, phi_load_path)
    print("Loaded checkpoint: %s" % phi_load_path)
    print("  h = %s, ci = %s" % (phi_star_fn.h, phi_star_fn.ci))

    return phi_star_fn, param_dict
