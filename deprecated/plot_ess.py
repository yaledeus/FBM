import torch
import numpy as np
import mdtraj as md
import yaml
from types import SimpleNamespace
from argparse import ArgumentParser
import os
import glob
import json
import pickle
from utils.tica_utils import *
from utils.backbone_utils import *
from plots import *


mean_infer_time = {
    "MD": 3101.34,
    "FBM": 472.88,
    "FBM-base": 464.32,
    "Timewarp": 676.00
}


def load_traj(trajfile, top):
    _, ext = os.path.splitext(trajfile)
    if ext in ['.pdb']:
        traj = md.load(trajfile)
    elif ext in ['.xtc']:
        traj = md.load(trajfile, top=top)
    elif ext in ['.npz']:
        positions = np.load(trajfile)["positions"]
        traj = md.Trajectory(
            positions,
            md.load(top).topology
        )
    elif ext in ['.npy']:
        positions = np.load(trajfile)
        if positions.ndim == 4:
            positions = positions[0]
        traj = md.Trajectory(
            positions,
            md.load(top).topology
        )
    else:
        raise NotImplementedError
    return traj


def parse():
    arg_parser = ArgumentParser(description='Perform evaluation on peptides and compare with baselines')
    arg_parser.add_argument('--config', type=str, help='configuration file path')
    return arg_parser.parse_args()


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    else:
        return d


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)
    test_set = config.test_set

    ess_list, method_list = [], []

    with open(test_set, 'r') as fp:
        lines = fp.readlines()
    pdbstats = [json.loads(line) for line in lines]

    ess_pkl = "outputs/ess.pkl"
    if not os.path.exists(ess_pkl):
        for item in pdbstats:
            pdb, traj_ref_path, top = item["pdb"], item["traj_npz_path"], item["state0_path"]
            traj_ref = load_traj(traj_ref_path, top=top)

            # TICA can be loaded if constructed before
            ref_dir = os.path.split(traj_ref_path)[0]
            if os.path.exists(tica_model_path := os.path.join(ref_dir, "tica_model.pic")):
                with open(tica_model_path, "rb") as f:
                    tica_model = pickle.load(f)
            else:
                tica_model = run_tica(traj_ref, lagtime=100, dim=4)
                with open(tica_model_path, "wb") as f:
                    pickle.dump(tica_model, f)

            features = tica_features(traj_ref)
            tics_ref = tica_model.transform(features)

            ess_list.append(ESS(tics_ref, axis=0) / mean_infer_time['MD'])
            method_list.append('MD')

            for model in config.models:
                method, traj_model_base = model["method"], model["traj_model_base"]
                if method == "FBM":
                    traj_model_path = os.path.join(traj_model_base, pdb, f'{pdb}_model_ode25_inf1000_guidance0.06.pdb')
                elif method == 'FBM-base':
                    traj_model_path = os.path.join(traj_model_base, pdb, f'{pdb}_model_ode30_inf1000_guidance0.05.pdb')
                elif method == 'Timewarp':
                    traj_model_dir = os.path.join(traj_model_base, pdb, 'timewarp', 'output-test')
                    traj_model_path = glob.glob(os.path.join(traj_model_dir, '*conditional.npz'))[0]
                else:
                    return
                traj_model = load_traj(traj_model_path, top=top)

                features_model = tica_features(traj_model)
                tics_model = tica_model.transform(features_model)

                ess_list.append(ESS(tics_model, axis=0) / mean_infer_time[method])
                method_list.append(method)

        ess_data = {
            "ess_list": ess_list,
            "method_list": method_list
        }
        with open(ess_pkl, "wb") as f:
            pickle.dump(ess_data, f)
    else:
        with open(ess_pkl, "rb") as f:
            ess_data = pickle.load(f)
            ess_list, method_list = ess_data["ess_list"], ess_data["method_list"]

    plot_ess(ess_list, method_list, save_path=os.path.join(f"outputs/ess_all.pdf"), screen=["MD", "FBM"])


if __name__ == "__main__":
    main(parse())
