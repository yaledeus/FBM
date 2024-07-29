import torch
from torch.utils.data import DataLoader
from scipy.special import softmax, logsumexp
from openmm.app import PDBFile
import time
import glob
import math
import shutil
import json
import os

from config import inference_config
from data import *
from utils import *
from utils.random_seed import setup_seed, SEED
from simulation import get_openmm_simulation, spring_constraint_energy_minim

### set backend == "pytorch"
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

setup_seed(SEED)
torch.set_default_dtype(torch.float32)


def create_save_dir(args):
    if args.save_dir is None:
        save_dir = '.'.join(args.ckpt.split('.')[:-1]) + '_results'
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def main(args):
    inf_step = args.inf_step
    ode_step = args.ode_step
    bs = args.bs
    # load test set
    test_set = args.test_set
    save_dir = create_save_dir(args)

    # load model
    model = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    # with open(test_set, 'r') as fp:
    #     lines = fp.readlines()
    #
    # items = [json.loads(line) for line in lines]

    pdb = args.name
    out_dir = os.path.join(save_dir, pdb)
    os.makedirs(out_dir, exist_ok=True)

    start = time.time()

    simulation = get_openmm_simulation(PDBFile(test_set).topology, gpu=args.gpu)
    topology = md.load(test_set).topology

    # make test batch
    batch = make_batch(test_set, bs)
    for k in batch:
        if hasattr(batch[k], 'to'):
            batch[k] = batch[k].to(device)

    positions, ideal_positions = [], []

    with torch.no_grad():
        for _ in tqdm(range(inf_step)):
            x = model.ode(batch, ode_step=ode_step, guidance=args.guidance)
            x = x.cpu().numpy() / 10    # Angstrom => nm
            positions.append(x)
            minim_x = spring_constraint_energy_minim(simulation, x)
            ideal_positions.append(minim_x)
            # update batch
            batch["x0"] = torch.from_numpy(10 * minim_x).to(device)  # nm => Angstrom

    positions = np.array(positions, dtype=float)    # (T, N, 3)
    ideal_positions = np.array(ideal_positions, dtype=float)

    md.Trajectory(
        positions,
        topology
    ).save_pdb(os.path.join(out_dir, f'{pdb}_model_ode{ode_step}_inf{inf_step}_guidance{args.guidance}.pdb'))

    md.Trajectory(
        ideal_positions,
        topology
    ).save_pdb(os.path.join(out_dir, f'{pdb}_model_ode{ode_step}_inf{inf_step}_guidance{args.guidance}_ideal.pdb'))

    end = time.time()
    elapsed_time = end - start

    print(f"[*] Inference finished, total elapsed time: {elapsed_time}.")


if __name__ == "__main__":
    args = inference_config()
    main(args)
