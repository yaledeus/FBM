import numpy as np

from config import inference_config
from openmm.app import PDBFile
import time

from data import *
from evaluate import *
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
    save_dir = create_save_dir(args)

    # load model
    model = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    with open(args.test_set, 'r') as fp:
        lines = fp.readlines()
    pdbstats = [json.loads(line) for line in lines]

    start = time.time()

    pwd_js_list, rg_js_list, tic_js_list, tic2d_js_list, val_ca_list, rmse_contact_list = [], [], [], [], [], []

    for pdbstat in pdbstats:
        pdb = pdbstat["pdb"]
        out_dir = os.path.join(save_dir, pdb)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[+] Inference on peptide {pdb} starts...")

        traj_npz = pdbstat["traj_npz_path"]
        state0 = pdbstat["state0_path"]

        simulation = get_openmm_simulation(PDBFile(state0).topology, gpu=args.gpu)
        topology = md.load(state0).topology

        # make test batch
        batch = make_batch(state0, bs)
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
        ).save_pdb(model_traj := os.path.join(out_dir, f'{pdb}_model_ode{ode_step}_inf{inf_step}_guidance{args.guidance}.pdb'))

        md.Trajectory(
            ideal_positions,
            topology
        ).save_pdb(ideal_model_traj := os.path.join(out_dir, f'{pdb}_model_ode{ode_step}_inf{inf_step}_guidance{args.guidance}_ideal.pdb'))

        # evaluate
        _, pwd_js, rg_js, tic_js, tic2d_js, val_ca, rmse_contact = \
            traj_analysis(ideal_model_traj, traj_npz, top=state0)

        if pwd_js != 0:
            pwd_js_list.append(pwd_js)
        if rg_js != 0:
            rg_js_list.append(rg_js)
        tic_js_list.append(tic_js)
        tic2d_js_list.append(tic2d_js)
        val_ca_list.append(val_ca)
        if rmse_contact != 0:
            rmse_contact_list.append(rmse_contact)

    pwd_js_list = np.array(pwd_js_list)
    rg_js_list = np.array(rg_js_list)
    tic_js_list = np.array(tic_js_list)
    tic2d_js_list = np.array(tic2d_js_list)
    val_ca_list = np.array(val_ca_list)
    rmse_contact_list = np.array(rmse_contact_list)

    print(f"[+] Stats: PwD {np.mean(pwd_js_list):.4f} +- {np.std(pwd_js_list):.4f}\n"
          f"Rg {np.mean(rg_js_list):.4f} +- {np.std(rg_js_list):.4f}\n"
          f"TIC {np.mean(tic_js_list):.4f} +- {np.std(tic_js_list):.4f}\n"
          f"TIC2D {np.mean(tic2d_js_list):.4f} +- {np.std(tic2d_js_list):.4f}\n"
          f"Val CA {np.mean(val_ca_list):.4f} +- {np.std(val_ca_list):.4f}\n"
          f"RMSE Contact {np.mean(rmse_contact_list):.4f} +- {np.std(rmse_contact_list):.4f}")

    stats = {
        "PwD": pwd_js_list.tolist(),
        "Rg": rg_js_list.tolist(),
        "TIC": tic_js_list.tolist(),
        "TIC2D": tic2d_js_list.tolist(),
        "Val-CA": val_ca_list.tolist(),
        "RMSE-Contact": rmse_contact_list.tolist()
    }

    with open(os.path.join(save_dir, f"model_ode{ode_step}_inf{inf_step}_guidance{args.guidance}_stats.json"), "w") as f:
        f.write(json.dumps(stats))

    end = time.time()
    elapsed_time = end - start

    print(f"[*] Inference finished, total elapsed time: {elapsed_time}.")


if __name__ == "__main__":
    args = inference_config()
    main(args)
