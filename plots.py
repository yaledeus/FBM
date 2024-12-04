import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.stats import gaussian_kde
import os


def create_save_dir(save_path):
    save_dir = os.path.split(save_path)[0]
    os.makedirs(save_dir, exist_ok=True)


def plot_tic2d_contour(tic0_ref, tic1_ref, tic0_model, tic1_model, save_path, xlabel='TIC 0', ylabel='TIC 1', name="FBM", kde=None):
    create_save_dir(save_path)

    if not kde:
        kde = gaussian_kde(np.vstack([tic0_ref, tic1_ref]))
        tic0_min, tic0_max = tic0_ref.min(), tic0_ref.max()
        tic1_min, tic1_max = tic1_ref.min(), tic1_ref.max()
        X, Y = np.meshgrid(np.linspace(tic0_min, tic0_max, 200), np.linspace(tic1_min, tic1_max, 200))
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kde(positions).T, X.shape)
    else:
        X, Y, Z = kde

    thresh = 0.013
    Z[Z < thresh] = np.nan
    sigma = 1.0  # Gaussian filter sigma
    Z_smooth = gaussian_filter(Z, sigma=sigma)

    plt.figure(figsize=(16, 12))
    ax = plt.gca()
    plt.contour(X, Y, Z_smooth, levels=15, cmap='viridis', linewidths=2.0, alpha=0.5)

    local_max = maximum_filter(Z_smooth, size=20) == Z_smooth
    maxima = np.argwhere(local_max & ~np.isnan(Z_smooth))
    # print(f"local maxima: {maxima}")

    collections = ax.collections
    paths = [path for collection in collections for path in collection.get_paths()]
    vertices = np.concatenate([path.vertices for path in paths])
    contour_x, contour_y = vertices[:, 0], vertices[:, 1]

    tic0_min, tic0_max = contour_x.min(), contour_x.max()
    tic1_min, tic1_max = contour_y.min(), contour_y.max()

    i = 1
    for (y, x) in maxima:
        center_x = X[0, x]
        center_y = Y[y, 0]
        if tic0_min <= center_x <= tic0_max and tic1_min <= center_y <= tic1_max:
            #######################
            # dist_ref = np.sqrt((tic0_ref - center_x) ** 2 + (tic1_ref - center_y) ** 2)
            # dist_model = np.sqrt((tic0_model - center_x) ** 2 + (tic1_model - center_y) ** 2)
            # print(f"cluster {i}, MD index: {np.argmin(dist_ref)}, model index: {np.argmin(dist_model)}")
            #######################
            plt.text(center_x, center_y, str(i), ha='center', va='center', fontsize=40, color='black', fontweight='bold')
            i += 1

    model_points = (tic0_model >= tic0_min) & (tic0_model <= tic0_max) & \
                   (tic1_model >= tic1_min) & (tic1_model <= tic1_max)

    plt.scatter(tic0_model[model_points], tic1_model[model_points], color='#D2691E', alpha=0.4, s=60)
    plt.scatter(tic0_ref[0], tic1_ref[0], color='#FF2626', s=120)

    # plt.text(0.05, 0.05, name, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize=40)

    plt.xlim(tic0_min, tic0_max)
    plt.ylim(tic1_min, tic1_max)

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.title(name, color='black', fontsize=50)
    plt.xlabel(xlabel, color='black', fontsize=50)
    plt.ylabel(ylabel, color='black', fontsize=50)

    plt.savefig(save_path, format='pdf')
    plt.clf()

    return [X, Y, Z]


def plot_tic2d_hist(tic0, tic1, save_path, tic_range=None, bins=100, xlabel='TIC0', ylabel='TIC1', name=None):
    create_save_dir(save_path)

    if tic_range is None:
        tic_range = ((tic0.min(), tic0.max()), (tic1.min(), tic1.max()))

    plt.figure(figsize=(16, 12))
    ax = plt.gca()
    ax.hist2d(tic0, tic1, bins=bins, cmap="OrRd", norm=LogNorm(), range=tic_range)
    ax.tick_params(axis='both', labelsize=30)
    ax.set_xlabel(xlabel, color='black', fontsize=50)
    ax.set_ylabel(ylabel, color='black', fontsize=50)
    plt.title(f"{name}", fontsize=50)

    plt.savefig(save_path, format='pdf')
    plt.clf()


def plot_free_energy(torsion_ref, torsion_model, save_path, xlabel='TIC 0', name='FBM'):
    create_save_dir(save_path)

    plt.figure(figsize=(16, 12))
    feat_bins = np.linspace(torsion_ref.min(), torsion_ref.max(), 100)
    hist_ref, edges_ref = np.histogram(torsion_ref, bins=feat_bins, density=True)
    free_energy_ref = -np.log(hist_ref/hist_ref.max())
    centers_ref = 0.5*(edges_ref[1:] + edges_ref[:-1])
    plt.plot(centers_ref, free_energy_ref, linewidth=4, label='MD', linestyle='-')

    hist_model, edges_model = np.histogram(torsion_model, bins=feat_bins, density=True)
    free_energy_model = -np.log(hist_model/hist_model.max())
    centers_model = 0.5*(edges_model[1:] + edges_model[:-1])
    plt.plot(centers_model, free_energy_model, linewidth=4, label=name, linestyle='--')

    plt.legend(title='', loc='upper right', fontsize=30)

    ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])

    plt.xlabel(xlabel, color='black', fontsize=50)
    plt.ylabel("Free energy/$k_B$T", color='black', fontsize=50)

    plt.savefig(save_path, format='pdf')
    plt.clf()


def plot_free_energy_all(tics_list, method_list, save_path, xlabel='TIC 0'):
    create_save_dir(save_path)

    plt.figure(figsize=(16, 12))
    feat_bins = np.linspace(tics_list[0].min(), tics_list[0].max(), 100)

    for tics, method in zip(tics_list, method_list):
        hist, edges = np.histogram(tics, bins=feat_bins, density=True)
        free_energy = -np.log(hist/hist.max())
        centers = 0.5*(edges[1:] + edges[:-1])
        plt.plot(centers, free_energy, linewidth=4, label=method, linestyle='-' if method == 'MD' else '--')

    plt.legend(title='', loc='upper right', fontsize=30)

    ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])

    plt.xlabel(xlabel, color='black', fontsize=50)
    plt.ylabel("Free energy/$k_B$T", color='black', fontsize=50)

    plt.savefig(save_path, format='pdf')
    plt.clf()


def plot_residue_dist_map(res_dist_matrix, save_path, name=None, label=None, reverse=True):
    create_save_dir(save_path)

    plt.figure(figsize=(16, 12))
    sns.heatmap(
        res_dist_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues_r" if reverse else "Blues",
        cbar_kws={'label': label},
        annot_kws={'size': 25},
        square=True
    )

    # plt.title(name, color='black', fontsize=50)
    plt.xlabel("Residue ID", color='black', fontsize=50)
    plt.ylabel("Residue ID", color='black', fontsize=50)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label(label=label, fontsize=50)

    plt.savefig(save_path, format='pdf')
    plt.clf()


def plot_distance_distribution(adj_dist_list, method_list, save_path, bins=50):
    create_save_dir(save_path)

    sns.set(style="white")
    plt.figure(figsize=(16, 12))

    colors = ['#D62728', '#1F77B4']

    # adj_dist_list[0] default to be reference trajectories
    min_val, max_val = np.min(adj_dist_list[0]), np.max(adj_dist_list[0])
    shared_bins = np.linspace(min_val, max_val, bins)

    for adj_dist, method, color in zip(adj_dist_list, method_list, colors):
        sns.histplot(adj_dist, bins=shared_bins, kde=False, color=color, stat="density", label=method)

    # plt.title("Pairwise distance", fontsize=30)
    plt.xlabel("Distance (Å)", color='black', fontsize=50)
    plt.ylabel("Population", color='black', fontsize=50)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=30)

    plt.savefig(save_path, format='pdf')
    plt.clf()


def plot_rmsd_over_time(rmsd_list, method_list, save_path):
    create_save_dir(save_path)

    plt.figure(figsize=(16, 12))

    nframe = min([len(rmsd_trajs[0]) for rmsd_trajs in rmsd_list])

    for rmsd_trajs, method in zip(rmsd_list, method_list):
        n_traj = len(rmsd_trajs)
        rmsd = []
        for rmsd_traj in rmsd_trajs:
            rmsd += rmsd_traj[:nframe].tolist()
        # 0.5 ns per frame
        sim_time = np.tile(np.linspace(0, 0.5 * (nframe - 1), nframe), n_traj)
        sns.lineplot(x=sim_time, y=rmsd,
                     linewidth=4,
                     label=method,
                     errorbar='sd',
                     linestyle='-' if method == 'MD' else '--')

    plt.legend(title='', loc='lower right', fontsize=30)
    plt.xlabel("Time (ns)", color='black', fontsize=50)
    plt.ylabel("RMSD (Å)", color='black', fontsize=50)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.savefig(save_path, format='pdf')
    plt.clf()


def plot_validity(valid_conformations, save_path):
    create_save_dir(save_path)

    plt.figure(figsize=(16, 12))

    valid_population = np.cumsum(valid_conformations.astype(np.int_))

    # plot y = x for reference
    # max_pop = np.arange(0, max(valid_population))
    # plt.plot(max_pop, max_pop, linewidth=2, color='', linestyle='--')

    # plot cumulated population
    x = np.arange(len(valid_population))
    plt.plot(x, valid_population, color='#2F4F4F', linewidth=2)
    plt.fill_between(x, valid_population, color='#1E3A8A', alpha=0.6)

    plt.xlabel("Inference step", color='black', fontsize=50)
    plt.ylabel("# Valid conformations", color='black', fontsize=50)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(save_path, format='pdf')
    plt.clf()


def plot_validity_all(valid_list, method_list, save_path):
    create_save_dir(save_path)

    plt.figure(figsize=(16, 12))

    for valid_conf_trajs, method in zip(valid_list, method_list):
        n_traj = len(valid_conf_trajs)
        valid_population = []
        nframe = len(valid_conf_trajs[0])
        for valid_conf in valid_conf_trajs:
            valid_population += np.cumsum(valid_conf.astype(np.int_)).tolist()
        # plot cumulated population
        x = np.tile(np.arange(nframe), n_traj)
        sns.lineplot(x=x, y=valid_population,
                     linewidth=4,
                     label=method,
                     errorbar='sd')

    plt.xlabel("Inference step", color='black', fontsize=50)
    plt.ylabel("# Valid conformations", color='black', fontsize=50)

    plt.legend(title='', loc='upper left', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(save_path, format='pdf')
    plt.clf()


def plot_ess(ess_list, method_list, save_path, screen=None):
    create_save_dir(save_path)

    plt.figure(figsize=(16, 12))

    md_ess = []
    for ess, method in zip(ess_list, method_list):
        if method == 'MD':
            md_ess.append(ess)
    md_ess_median = np.median(np.array(md_ess, dtype=float))
    ess_list = np.array(ess_list, dtype=float)
    ess_list /= md_ess_median

    if screen is not None:
        screen_ess_list, screen_method_list = [], []
        for ess, method in zip(ess_list, method_list):
            if method in screen:
                screen_ess_list.append(ess)
                screen_method_list.append(method)
    else:
        screen_ess_list, screen_method_list = ess_list, method_list

    data = {
        "ess": screen_ess_list,
        "method": screen_method_list
    }

    # reference line y = 1
    plt.axhline(y=1, color='#00008B', linestyle='--', linewidth=2)

    sns.boxplot(data=data, x='method', y='ess', showfliers=False,
                boxprops=dict(facecolor='#D2B48C', edgecolor='black'),
                whiskerprops=dict(linewidth=3),
                capprops=dict(linewidth=3),
                medianprops=dict(linewidth=3)
                )

    plt.xlabel("Method", color='black', fontsize=50)
    plt.ylabel("ESS/s", color='black', fontsize=50)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(save_path, format='pdf')
    plt.clf()


if __name__ == "__main__":
    import os
    save_dir = "./tmp"
    os.makedirs(save_dir, exist_ok=True)

    T, A, R = 1000, 200, 15
    ### plot residue distance map
    # xyz = np.random.randn(T, R, 3)
    # res_dist = np.linalg.norm(xyz[:, :, np.newaxis, :] - xyz[:, np.newaxis, :, :], axis=-1)
    # min_dist_matrix = np.min(res_dist, axis=0)
    # plot_residue_dist_map(min_dist_matrix, os.path.join(save_dir, "res_dist_map.pdf"))
    #
    # ### plot distance distribution
    # adj_dist = np.random.randn(T, int(R * (R - 1) / 2))
    # plot_distance_distribution(adj_dist.flatten(), os.path.join(save_dir, "dist_distribution.pdf"))

    ### plot TIC-2D
    # tic01 = np.random.randn(T, 2)
    # plot_tic2d_hist(tic01[:, 0], tic01[:, 1], os.path.join(save_dir, "tic_2d.pdf"))

    ### plot rmsd
    # _time = np.arange(T).repeat(50)
    # rmsd = 2 + 0.3 * np.log(_time.astype(float)) + np.random.normal(scale=0.3, size=len(_time))
    # plot_rmsd_over_time([rmsd], ["MD"], os.path.join(save_dir, "rmsd.pdf"))

    ### plot validity
    validity = np.random.randn(T)
    validity = validity > 0
    plot_validity(validity, os.path.join(save_dir, "valid.pdf"))
