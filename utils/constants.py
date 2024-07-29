import numpy as np
import torch

BOND_N_CA = 0.1459  # N-CA bond length, unit: nm
BOND_CA_C = 0.1525  # CA-C
BOND_C_O = 0.1229   # C=O
BOND_C_N = 0.1336   # C-N in peptide bond
ANGLE_N_CA_C = 1.9373   # N-CA-C angle, unit: radius
ANGLE_CA_C_N = 2.0455   # CA-C-N
ANGLE_CA_C_O = 2.0961   # CA-C=O
ANGLE_O_C_N = 2.1415    # O=C-N
ANGLE_C_N_CA = 2.1241   # C-N-CA


# peptide plane centered on N, order: N, CA, C, O
PEPTIDE_PLANE = np.array([[0, 0, 0], [0.1459, 0, 0], [0.20055, -0.14237, 0], [0.123368, -0.23801, 0]], dtype=float)
PEPTIDE_PLANE_TORCH = torch.from_numpy(PEPTIDE_PLANE)

kB = 0.0083144626  # KJ/mole/Kelvin
T = 300 # Kelvin
k = 1 / (kB * T)   # inverse temperature
