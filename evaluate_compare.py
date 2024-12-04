import torch
import numpy as np
import mdtraj as md
import yaml
from types import SimpleNamespace
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


def traj_analysis(config):
    pdb, top, traj_ref_path = config.pdb, config.top, config.traj_ref_path
    traj_ref = load_traj(traj_ref_path, top=top)

    # TICA can be loaded if constructed before
    ref_dir = os.path.split(config.traj_ref_path)[0]
    if os.path.exists(tica_model_path := os.path.join(ref_dir, "tica_model.pic")):
        with open(tica_model_path, "rb") as f:
            tica_model = pickle.load(f)
    else:
        tica_model = run_tica(traj_ref, lagtime=100, dim=4)
        with open(tica_model_path, "wb") as f:
            pickle.dump(tica_model, f)

    features = tica_features(traj_ref)
    tics_ref = tica_model.transform(features)
    phi_ref, psi_ref = compute_phi_psi(traj_ref)

    ### plot reference TIC-2D and Ramachandran
    tic_range = ((tics_ref[:, 0].min(), tics_ref[:, 0].max()), (tics_ref[:, 1].min(), tics_ref[:, 1].max()))
    ram_range = ((-np.pi, np.pi), (-np.pi, np.pi))
    # plot TIC-2D
    plot_tic2d_hist(tics_ref[:, 0], tics_ref[:, 1], save_path=f"outputs/{pdb}/tic2d_hist_md.pdf",
                    tic_range=tic_range, xlabel='TIC0', ylabel='TIC1', name='MD')
    ### plot Ramachandran
    plot_tic2d_hist(phi_ref, psi_ref, save_path=f"outputs/{pdb}/ram_hist_md.pdf",
                    tic_range=ram_range, xlabel='Phi', ylabel='Psi', name='MD')

    is_full_eval = True
    try:
        pwd_ref = compute_pairwise_distances(traj_ref)
    except BaseException as e:
        is_full_eval = False
        print(f"[!] Errno: {e}")
    rg_ref = compute_radius_of_gyration(traj_ref)
    contact_ref, res_dist_ref = compute_residue_matrix(traj_ref)
    init_pos = md.load(top).xyz[0]
    rmsd_over_time_ref = compute_rmsd_over_time(traj_ref, init_pos, lagtime=500)

    method_list, tics_list, rmsd_over_time_list, valid_list = [], [], [], []
    method_list.append('MD')
    tics_list.append(tics_ref[:, :2])
    rmsd_over_time_list.append([rmsd_over_time_ref])

    for model in config.models:
        name, traj_model_path_list = model['method'], model['traj_model_path']
        traj_model_path = traj_model_path_list[0]
        traj_model = load_traj(traj_model_path, top=top)

        print(f"Evaluating on method: {name}")

        # compute tica
        feat_model = tica_features(traj_model)
        tics_model = tica_model.transform(feat_model)

        tics_list.append(tics_model[:, :2])
        method_list.append(name)

        # compute phi and psi
        phi_model, psi_model = compute_phi_psi(traj_model)
        rg_model = compute_radius_of_gyration(traj_model)

        if is_full_eval:
            # compute JS distance of PwD
            pwd_model = compute_pairwise_distances(traj_model)

        # compute RMSE contact
        contact_model, res_dist_model = compute_residue_matrix(traj_model)

        ### plot tic2d
        min_num_samples = min(phi_model.shape[0], phi_ref.shape[0])
        plot_tic2d_hist(tics_model[:min_num_samples, 0], tics_model[:min_num_samples, 1], save_path=f"outputs/{pdb}/tic2d_hist_{name}.pdf",
                        tic_range=tic_range, xlabel='TIC0', ylabel='TIC1', name=name)
        ### plot Ramachandran
        plot_tic2d_hist(phi_model[:min_num_samples], psi_model[:min_num_samples], save_path=f"outputs/{pdb}/ram_hist_{name}.pdf",
                        tic_range=ram_range, xlabel='Phi', ylabel='Psi', name=name)
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

        rmsd_over_time_trajs, valid_conf_trajs = [], []
        for other_traj_path in traj_model_path_list:
            other_traj = load_traj(other_traj_path, top=top)
            rmsd_over_time_model = compute_rmsd_over_time(other_traj, init_pos, lagtime=1)
            _, valid_conf = compute_validity(other_traj)
            rmsd_over_time_trajs.append(rmsd_over_time_model)
            valid_conf_trajs.append(valid_conf)

        # save RMSD compared to state0
        rmsd_over_time_list.append(rmsd_over_time_trajs)
        # save valid conformations
        valid_list.append(valid_conf_trajs)

        # calculate metrics
        ramachandran_js = compute_joint_js_distance(phi_ref, psi_ref, phi_model, psi_model)
        n_residues = contact_ref.shape[0]
        if is_full_eval:
            pwd_js = compute_js_distance(pwd_ref, pwd_model)
            rg_js = compute_js_distance(rg_ref, rg_model)
            rmse_contact = np.sqrt(2 / (n_residues * (n_residues - 1)) * np.sum((contact_ref - contact_model) ** 2))
        else:
            pwd_js = rg_js = rmse_contact = 0
        tic_js = compute_js_distance(tics_ref[:, :2], tics_model[:, :2])
        tic2d_js = compute_joint_js_distance(tics_ref[:, 0], tics_ref[:, 1], tics_model[:, 0], tics_model[:, 1])
        val_ca, valid_conformations = compute_validity(traj_model)

        print(f"JS distance: Ram {ramachandran_js:.4f} PwD {pwd_js:.4f}, Rg {rg_js:.4f}, TIC {tic_js:.4f}, TIC2D {tic2d_js:.4f}")
        print(f"Validity CA: {val_ca:.4f}")
        print(f"RMSE contact: {rmse_contact:.4f}")


    ### plot for all baselines
    ### plot free energy projected on TIC0 and TIC1
    plot_free_energy_all([tics[:, 0] for tics in tics_list], method_list, xlabel='TIC 0', save_path=f"outputs/{pdb}/fe_tic0_all.pdf")
    plot_free_energy_all([tics[:, 1] for tics in tics_list], method_list, xlabel='TIC 1', save_path=f"outputs/{pdb}/fe_tic1_all.pdf")
    # plot RMSD compared to init
    plot_rmsd_over_time(rmsd_over_time_list, method_list, save_path=f"outputs/{pdb}/rmsd_over_time_all.pdf")
    # plot validity population
    plot_validity_all(valid_list, method_list[1:], save_path=f"outputs/{pdb}/validity_population_all.pdf")


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
    traj_analysis(config)


if __name__ == "__main__":
    main(parse())
