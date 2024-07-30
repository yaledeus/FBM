import torch
import numpy as np
import mdtraj as md
from argparse import ArgumentParser
import os
import pickle
from utils.tica_utils import *
from utils.backbone_utils import *
from plots import *


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


def traj_analysis(traj_model_path, traj_ref_path, top=None, plot=False, name="FBM"):
    traj_model = load_traj(traj_model_path, top=top)
    traj_ref = load_traj(traj_ref_path, top=top)

    # TICA can be loaded if constructed before
    ref_dir = os.path.split(traj_ref_path)[0]
    if os.path.exists(tica_model_path := os.path.join(ref_dir, "tica_model.pic")):
        with open(tica_model_path, "rb") as f:
            tica_model = pickle.load(f)
    else:
        tica_model = run_tica(traj_ref, lagtime=100)
        with open(tica_model_path, "wb") as f:
            pickle.dump(tica_model, f)

    # compute tica
    features = tica_features(traj_ref)
    feat_model = tica_features(traj_model)
    tics_ref = tica_model.transform(features)
    tics_model = tica_model.transform(feat_model)

    # compute phi and psi
    phi_ref, psi_ref = compute_phi_psi(traj_ref)
    phi_model, psi_model = compute_phi_psi(traj_model)
    ramachandran_js = compute_joint_js_distance(phi_ref, psi_ref, phi_model, psi_model)

    try:
        # compute JS distance of PwG, Rg and TIC01
        pwd_ref = compute_pairwise_distances(traj_ref)
        pwd_model = compute_pairwise_distances(traj_model)

        rg_ref = compute_radius_of_gyration(traj_ref)
        rg_model = compute_radius_of_gyration(traj_model)

        pwd_js = compute_js_distance(pwd_ref, pwd_model)
        rg_js = compute_js_distance(rg_ref, rg_model)
    except BaseException:
        pwd_js = 0
        rg_js = 0
    tic_js = compute_js_distance(tics_ref[:, :2], tics_model[:, :2])
    tic2d_js = compute_joint_js_distance(tics_ref[:, 0], tics_ref[:, 1], tics_model[:, 0], tics_model[:, 1])

    print(f"JS distance: Ram {ramachandran_js:.4f} PwD {pwd_js:.4f}, Rg {rg_js:.4f}, TIC {tic_js:.4f}, TIC2D {tic2d_js:.4f}")

    # compute Val-CA
    val_ca = compute_validity(traj_model)
    print(f"Validity CA: {val_ca:.4f}")

    # compute RMSE contact
    try:
        contact_ref = compute_contact_matrix(traj_ref)
        contact_model = compute_contact_matrix(traj_model)
        n_residues = contact_ref.shape[0]
        rmse_contact = np.sqrt(2 / (n_residues * (n_residues - 1)) * np.sum((contact_ref - contact_model)**2))
    except BaseException:
        rmse_contact = 0
    print(f"RMSE contact: {rmse_contact:.4f}")


    if plot:
        plot_tic2d(tics_ref[:, 0], tics_ref[:, 1], tics_model[:, 0], tics_model[:, 1],
                   save_path="./tic2d.pdf", xlabel='TIC 0', ylabel='TIC 1', name=name)
        plot_tic2d(phi_ref, psi_ref, phi_model, psi_model,
                   save_path="./ram.pdf", xlabel='Phi', ylabel='Psi', name=name)
        plot_free_energy(tics_ref[:, 0], tics_model[:, 0], save_path="./free_energy_tic0.pdf", xlabel='TIC 0', name=name)
        plot_free_energy(tics_ref[:, 1], tics_model[:, 1], save_path="./free_energy_tic1.pdf", xlabel='TIC 1', name=name)

    return ramachandran_js, pwd_js, rg_js, tic_js, tic2d_js, val_ca, rmse_contact


def parse():
    arg_parser = ArgumentParser(description='simulation')
    arg_parser.add_argument('--top', type=str, default=None, help='topology file path (.pdb)')
    arg_parser.add_argument('--ref', type=str, required=True, help='reference trajectory file path')
    arg_parser.add_argument('--model', type=str, required=True, help='model generated trajectory file path')
    arg_parser.add_argument('--name', type=str, default='FBM', help='model type for plotting figures')
    return arg_parser.parse_args()


def main(args):
    traj_analysis(args.model, args.ref, top=args.top, plot=False, name=args.name)


if __name__ == "__main__":
    main(parse())
