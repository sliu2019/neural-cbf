import torch
import numpy as np

# TODO: this makes torch and numpy deterministic.
# TODO: This may not be what you want for, (for example) multiple identical runs with different randomness
torch.manual_seed(10)
np.random.seed(2021)

# red_rgb = np.array([168, 26, 26])/(255.0) # Tamarillo
# red_rgb = np.array([195, 26, 26])/(255.0) # Thunderbird

# green_rgb = np.array([64, 150, 43])/(255.0) # Forest green
# green_rgb = np.array([62, 219, 22])/(255.0) # pea green

red_rgb = np.array([199,115,113])/(255.0)
blue_rgb = np.array([118,131,202])/(255.0)
dark_blue_rgb = np.array([92,100,137])/(255.0)