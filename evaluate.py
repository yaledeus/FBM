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

    # top = traj.topology
    # selection = top.select('not name SOD')
    # traj = traj.atom_slice(selection)

    return traj


def traj_analysis(traj_model_path, traj_ref_path, top, plot=False, name="FBM", pdb="default"):
    traj_model = load_traj(traj_model_path, top=top)
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

    # compute tica
    features = tica_features(traj_ref)
    feat_model = tica_features(traj_model)
    tics_ref = tica_model.transform(features)
    tics_model = tica_model.transform(feat_model)

    # compute phi and psi
    phi_ref, psi_ref = compute_phi_psi(traj_ref)
    phi_model, psi_model = compute_phi_psi(traj_model)
    ramachandran_js = compute_joint_js_distance(phi_ref, psi_ref, phi_model, psi_model)

    rg_ref = compute_radius_of_gyration(traj_ref)
    rg_model = compute_radius_of_gyration(traj_model)
    rg_js = compute_js_distance(rg_ref, rg_model)

    is_full_eval = True
    try:
        # compute JS distance of PwG, Rg and TIC01
        pwd_ref = compute_pairwise_distances(traj_ref)
        pwd_model = compute_pairwise_distances(traj_model)
        pwd_js = compute_js_distance(pwd_ref, pwd_model)
    except BaseException as e:
        print(f"[!] Errno: {e}")
        is_full_eval = False
        pwd_js = 0
    tic_js = compute_js_distance(tics_ref[:, :2], tics_model[:, :2])
    tic2d_js = compute_joint_js_distance(tics_ref[:, 0], tics_ref[:, 1], tics_model[:, 0], tics_model[:, 1])

    print(
        f"JS distance: Ram {ramachandran_js:.4f} PwD {pwd_js:.4f}, Rg {rg_js:.4f}, TIC {tic_js:.4f}, TIC2D {tic2d_js:.4f}")

    # compute Val-CA
    val_ca, valid_conformations = compute_validity(traj_model)
    print(f"Validity CA: {val_ca:.4f}")

    # compute RMSE contact
    try:
        contact_ref, res_dist_ref = compute_residue_matrix(traj_ref)
        contact_model, res_dist_model = compute_residue_matrix(traj_model)
        n_residues = contact_ref.shape[0]
        rmse_contact = np.sqrt(2 / (n_residues * (n_residues - 1)) * np.sum((contact_ref - contact_model) ** 2))
    except BaseException as e:
        print(f"[!] Errno: {e}")
        rmse_contact = 0
    print(f"RMSE contact: {rmse_contact:.4f}")

    if plot:
        ### plot tic2d
        tic_range = ((tics_ref[:, 0].min(), tics_ref[:, 0].max()), (tics_ref[:, 1].min(), tics_ref[:, 1].max()))
        plot_tic2d_hist(tics_ref[:, 0], tics_ref[:, 1], save_path=f"outputs/{pdb}/tic2d_hist_md.pdf", tic_range=tic_range, name=name)
        plot_tic2d_hist(tics_model[:, 0], tics_model[:, 1], save_path=f"outputs/{pdb}/tic2d_hist_{name}.pdf", tic_range=tic_range, name=name)
        tic_kde_path = f"outputs/{pdb}/tic_kde.npz"
        ram_kde_path = f"outputs/{pdb}/ram_kde.npz"
        if os.path.exists(tic_kde_path):
            tic_kde = np.load(tic_kde_path)
            tic_x, tic_y, tic_z = tic_kde['x'], tic_kde['y'], tic_kde['z']
            plot_tic2d_contour(tics_ref[:, 0], tics_ref[:, 1], tics_model[:, 0], tics_model[:, 1],
                               save_path=f"outputs/{pdb}/tic2d_contour_{name}.pdf", xlabel='TIC 0', ylabel='TIC 1', name=name,
                               kde=(tic_x, tic_y, tic_z))
        else:
            tic_x, tic_y, tic_z = plot_tic2d_contour(tics_ref[:, 0], tics_ref[:, 1], tics_model[:, 0], tics_model[:, 1],
                                                     save_path=f"outputs/{pdb}/tic2d_contour_{name}.pdf", xlabel='TIC 0', ylabel='TIC 1', name=name)
            np.savez(tic_kde_path, x=tic_x, y=tic_y, z=tic_z)
        if os.path.exists(ram_kde_path):
            ram_kde = np.load(ram_kde_path)
            ram_x, ram_y, ram_z = ram_kde['x'], ram_kde['y'], ram_kde['z']
            plot_tic2d_contour(phi_ref, psi_ref, phi_model, psi_model,
                               save_path=f"outputs/{pdb}/ram_contour_{name}.pdf", xlabel='Phi', ylabel='Psi', name=name,
                               kde=(ram_x, ram_y, ram_z))
        else:
            ram_x, ram_y, ram_z = plot_tic2d_contour(phi_ref, psi_ref, phi_model, psi_model,
                                                     save_path=f"outputs/{pdb}/ram_contour_{name}.pdf", xlabel='Phi', ylabel='Psi', name=name)
            np.savez(ram_kde_path, x=ram_x, y=ram_y, z=ram_z)
        ### plot free energy projected on TIC0 and TIC1
        plot_free_energy(tics_ref[:, 0], tics_model[:, 0], save_path=f"outputs/{pdb}/fe_tic0_{name}.pdf", xlabel='TIC 0', name=name)
        plot_free_energy(tics_ref[:, 1], tics_model[:, 1], save_path=f"outputs/{pdb}/fe_tic1_{name}.pdf", xlabel='TIC 1', name=name)
        # plot residue minimum distance map
        res_dist_map = np.zeros(res_dist_ref.shape, dtype=float)
        res_dist_map += 10 * np.tril(res_dist_model, -1)
        res_dist_map += 10 * np.triu(res_dist_ref, 1)
        plot_residue_dist_map(res_dist_map, save_path=f"outputs/{pdb}/rdist_map_{name}.pdf", name=name, label='Distance (Å)', reverse=True)
        # plot residue contact map
        res_contact_map = np.zeros(contact_ref.shape, dtype=float)
        res_contact_map += np.tril(contact_model.T, -1)
        res_contact_map += np.triu(contact_ref, 1)
        plot_residue_dist_map(res_contact_map, save_path=f"outputs/{pdb}/rcontact_map_{name}.pdf", name=name, label='Contact rate', reverse=False)
        # plot distance distribution
        plot_distance_distribution([10 * rg_ref.flatten(), 10 * rg_model.flatten()], ["MD", name], save_path=f"outputs/{pdb}/rg_distribution_{name}.pdf")
        if is_full_eval:
            plot_distance_distribution([10 * pwd_ref.flatten(), 10 * pwd_model.flatten()], ["MD", name], save_path=f"outputs/{pdb}/pwd_distribution_{name}.pdf")
        # plot RMSD compared to init
        nframe = min(traj_model.xyz.shape[0], int(traj_ref.xyz.shape[0] / 500))
        init_pos = md.load(top).xyz[0]
        rmsd_over_time_ref = compute_rmsd_over_time(traj_ref, init_pos, lagtime=500, nframe=nframe)
        rmsd_over_time_model = compute_rmsd_over_time(traj_model, init_pos, lagtime=1, nframe=nframe)
        plot_rmsd_over_time([[rmsd_over_time_ref], [rmsd_over_time_model]], ["MD", name], save_path=f"outputs/{pdb}/rmsd_over_time_{name}.pdf")
        # plot validity population
        plot_validity(valid_conformations, save_path=f"outputs/{pdb}/validity_population_{name}.pdf")


    return ramachandran_js, pwd_js, rg_js, tic_js, tic2d_js, val_ca, rmse_contact


def parse():
    arg_parser = ArgumentParser(description='Perform evaluation on the test set')
    arg_parser.add_argument('--top', type=str, default=None, help='topology file path (.pdb)')
    arg_parser.add_argument('--ref', type=str, required=True, help='reference trajectory file path')
    arg_parser.add_argument('--model', type=str, required=True, help='model generated trajectory file path')
    arg_parser.add_argument('--name', type=str, default='FBM', help='model type for plotting figures')
    arg_parser.add_argument('--pdb', type=str, default='default', help='PDB name for plotting figures')
    arg_parser.add_argument('--plot', action='store_true', help='if specified, generate plots (much slower!)')
    return arg_parser.parse_args()


def main(args):
    traj_analysis(args.model, args.ref, top=args.top, plot=args.plot, name=args.name, pdb=args.pdb)


if __name__ == "__main__":
    main(parse())
