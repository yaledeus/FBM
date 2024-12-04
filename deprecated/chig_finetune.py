import glob
import json
import os
import pickle
import sys
import random
from argparse import ArgumentParser

import openmm as mm
import openmm.unit as u
from openmm import app
from openmm.app import CharmmPsfFile

import mdtraj as md
from tqdm import tqdm

sys.path.append('..')
from utils import load_file
from utils.bio_utils import *
from utils.constants import k


def get_simulation_environment_integrator():
    """Obtain integrator from parameters."""

    integrator = mm.LangevinIntegrator(
        350 * u.kelvin,
        0.1 / u.picosecond,
        4.0 * u.femtosecond
    )

    return integrator


def get_simulation_environment_from_model(model, gpu=-1):
    """Obtain simulation environment suitable for energy computation."""
    system = get_system(model)
    integrator = get_simulation_environment_integrator()
    if gpu == -1:
        simulation = mm.app.Simulation(model.topology, system, integrator)
    else:
        platform = mm.Platform.getPlatformByName('CUDA')
        properties = {'DeviceIndex': f'{gpu}'}
        simulation = mm.app.Simulation(model.topology, system, integrator, platform, properties)

    return simulation


def get_simulation_environment_from_pdb(psf, pdb, gpu=-1):
    model = get_openmm_model(psf, pdb)
    return get_simulation_environment_from_model(model, gpu)


def get_system(model):
    """Obtain system to generate e.g. a simulation environment."""
    forcefield = mm.app.ForceField("charmm36.xml", "implicit/obc1.xml")

    system = forcefield.createSystem(
        model.topology,
        nonbondedMethod=mm.app.CutoffNonPeriodic,
        nonbondedCutoff=2.0 * u.nanometer,
        constraints=mm.app.HBonds
    )

    return system


def get_openmm_model(psfpath, state0pdbpath):
    """Create openmm model from pdb file.

    Arguments
    ---------
    psfpath: str
        Topology file
    state0pdbpath : str
        Pathname for all-atom state0.pdb file created by simulate_trajectory.

    Returns
    -------
    model : openmm.app.modeller.Modeller
        Modeller provides tools for editing molecular models, such as adding water or missing hydrogens.
        This object can also be used to create simulation environments.
    """
    # pdb_file = mm.app.pdbfile.PDBFile(state0pdbpath)
    # positions = pdb_file.getPositions()
    # topology = pdb_file.getTopology()
    pdb = mm.app.pdbfile.PDBFile(state0pdbpath)
    psf = CharmmPsfFile(psfpath)
    model = mm.app.modeller.Modeller(psf.topology, pdb.positions[:175])
    return model


def get_potential(simulation, positions):
    simulation.context.setPositions(positions)

    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy().value_in_unit(
            u.kilojoule / u.mole)

    return potential


def get_force(simulation, positions):
    simulation.context.setPositions(positions)

    state = simulation.context.getState(getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(
        u.kilojoules / (u.mole * u.nanometer)).astype(np.float32)

    return forces


def split_chig(raw_data_dir):
    random.seed(42)
    save_dir = os.path.split(raw_data_dir)[0]
    state0_path = os.path.join(raw_data_dir, 'filtered_amber.pdb')
    productions = [os.path.abspath(f) for f in os.scandir(raw_data_dir) if f.is_dir()]
    # randomly pick train/valid/test = 100/20/3
    train_num, valid_num, test_num = 500, 50, 3
    splits = random.sample(productions, train_num + valid_num + test_num)
    train_split, valid_split, test_split = splits[:train_num], splits[train_num:-test_num], splits[-test_num:]
    split_data = {
        "train": train_split,
        "valid": valid_split,
        "test": test_split
    }
    split_files = []
    for split_name, paths in split_data.items():
        with open(split_file := os.path.join(save_dir, f"{split_name}.jsonl"), 'w') as fout:
            for path in paths:
                traj_xtc_path = glob.glob(os.path.join(path, "*.xtc"))[0]
                item = {
                    "pdb": "Chignolin",
                    "state0_path": state0_path,
                    "traj_xtc_path": traj_xtc_path
                }
                item_str = json.dumps(item)
                fout.write(f'{item_str}\n')
        split_files.append(split_file)

    return split_files


def compute_ef(split_file, split='train', gpu=-1):
    items = load_file(split_file)
    # state0_path = items[0]["state0_path"]
    psf_path = "/data/private/yale/Chignolin/filtered/filtered.psf"
    pdb_path = "/data/private/yale/Chignolin/filtered/filtered.pdb"
    sim = get_simulation_environment_from_pdb(psf_path, pdb_path, gpu=gpu)

    post_items = []

    for item in tqdm(items):
        state0_path, traj_xtc_path = item["state0_path"], item["traj_xtc_path"]
        save_dir = os.path.split(traj_xtc_path)[0]
        traj = md.load_xtc(traj_xtc_path, top=state0_path)
        top = traj.topology
        selection = top.select('not name SOD')

        positions = traj.xyz
        energies, forces = [], []
        for position in positions:
            energy = get_potential(sim, position[selection])
            force = get_force(sim, position[selection])
            energies.append(energy)
            forces.append(force)
        energies = np.array(energies, dtype=float)
        forces = np.array(forces, dtype=float)

        traj_ef_path = os.path.join(save_dir, "ef.npz")
        np.savez(traj_ef_path, energies=energies, forces=forces)

        item.update({
            "traj_ef_path": traj_ef_path
        })
        post_items.append(item)

    with open(post_split_file := os.path.join(os.path.split(split_file)[0], f"{split}-post.jsonl"), 'w') as fout:
        for item in post_items:
            item_str = json.dumps(item)
            fout.write(f'{item_str}\n')

    return post_split_file


### preprocess openmm simulation output to curate Peptide train/valid data
def preprocess_chig(split_path, split='train', delta=1):
    """
    :param split_path: split summary file, jsonl format
    :param delta: time interval between training pairs, default 5(x100)ps
    """
    items = load_file(split_path)

    np.random.seed(42)

    save_dir = os.path.split(split_path)[0]
    split_data = []

    for item in tqdm(items):
        traj = md.load_xtc(item["traj_xtc_path"], top=item["state0_path"])
        top = traj.topology
        selection = top.select('not name SOD')
        traj = traj.atom_slice(selection)

        ef = np.load(item["traj_ef_path"])
        xyz = 10 * traj.xyz                     # (T, N, 3), Angstrom
        forces = 0.001 * ef["forces"]           # (T, N, 3), J/mol/nm
        potentials = ef["energies"] / xyz.shape[1]     # (T,)
        T = xyz.shape[0]

        atype = get_atype(traj.topology)  # (N,)
        rtype = np.array([])
        rmask = np.array([])
        bb_index = np.array([])
        # rtype = get_rtype(traj.topology)  # (N,)
        # rmask = get_res_mask(traj.topology)  # (N,)
        # bb_index = get_backbone_index(traj.topology)  # (B, 4)

        # data split
        valid_length = list(range(T - delta))
        idx = np.random.choice(valid_length, 200)

        for i in idx:
            # get protein pairs
            x0, x1 = xyz[i], xyz[i + delta]
            x0c = x0.mean(axis=0)
            # not conservative field!
            x1c = x1.mean(axis=0)
            x0 = x0 - x0c
            x1 = x1 - x0c
            force0, force1 = forces[i], forces[i + delta]
            pot0, pot1 = potentials[i], potentials[i + delta]
            data = {
                "atype": atype,
                "rtype": rtype,
                "rmask": rmask,
                "bb_index": bb_index,
                "x0": x0,
                "x1": x1,
                "force0": force0,
                "force1": force1,
                "pot0": pot0,
                "pot1": pot1
            }
            split_data.append(data)

    with open(os.path.join(save_dir, f"{split}.pkl"), "wb") as f:
        pickle.dump(split_data, f)


def parse():
    arg_parser = ArgumentParser(description='curate Chignolin dataset')
    arg_parser.add_argument('--raw_dir', type=str, required=True, help='raw data dir')
    arg_parser.add_argument('--gpu', type=str, default=-1, help='specify GPU index for acceleration')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    ### curate Peptide data
    split_chig(args.raw_dir)
    split_files = ["/data/private/yale/Chignolin/train.jsonl", "/data/private/yale/Chignolin/valid.jsonl"]
    for split_file, split in zip(split_files, ["train", "valid"]):
        compute_ef(split_file, split, gpu=args.gpu)
    post_split_files = ["/data/private/yale/Chignolin/train-post.jsonl", "/data/private/yale/Chignolin/valid-post.jsonl"]
    for post_split_file, split in zip(post_split_files, ["train", "valid"]):
        preprocess_chig(post_split_file, split)

