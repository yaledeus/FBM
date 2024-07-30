import glob
import json
import os
import pickle
import random
import sys
from argparse import ArgumentParser

import mdtraj as md
import torch
from tqdm import tqdm

sys.path.append('..')
from utils import load_file
from utils.bio_utils import *
from utils.constants import k


### split pep data
def split_pep(sim_dir):
    save_dir = os.path.split(sim_dir)[0]
    splits = ["train", "test"]
    all_items = []

    for _dir in os.listdir(sim_dir):
        _dir = os.path.join(sim_dir, _dir)
        traj_npz_path = glob.glob(_dir + '/*.npz')[0]
        name = os.path.split(traj_npz_path)[-1]
        pdb, chain = name[:4], name[5]
        state0_path = os.path.join(_dir, "state0.pdb")
        all_items.append({
            "pdb": pdb,
            "traj_npz_path": traj_npz_path,
            "state0_path": state0_path
        })

    np.random.seed(42)
    np.random.shuffle(all_items)

    # train:test=4:1
    train_len = int(0.8 * len(all_items))

    split_items = [
        all_items[:train_len],
        all_items[train_len:]
    ]

    split_paths = []

    for split, items in zip(splits, split_items):
        split_path = os.path.join(save_dir, f"{split}.jsonl")
        split_paths.append(split_path)
        with open(split_path, 'w') as fout:
            for item in items:
                item_str = json.dumps(item)
                fout.write(f'{item_str}\n')

    return split_paths


### preprocess openmm simulation output to curate Peptide train/valid data
def preprocess_pep(split_path, delta=500):
    """
    :param split_path: split summary file, jsonl format
    :param delta: time interval between training pairs, default 500ps
    """
    # save_dir = os.path.split(split_path)[0]
    # split = os.path.split(split_path)[-1].split('.')[0]     # train/valid
    items = load_file(split_path)

    np.random.seed(42)

    for item in tqdm(items):
        save_dir = os.path.split(item["state0_path"])[0]
        state0 = md.load(item["state0_path"])
        top = state0.topology
        traj_npz = np.load(item["traj_npz_path"])
        xyz = 10 * traj_npz["positions"]            # (T, N, 3), Angstrom
        forces = 0.002 * k * traj_npz["forces"]     # (T, N, 3), nm
        potentials = k * traj_npz["energies"][:, 0] / xyz.shape[1] / 3     # (T,)
        T = xyz.shape[0]

        atype = get_atype(top)  # (N,)
        rtype = get_rtype(top)  # (N,)
        rmask = get_res_mask(top)   # (N,)
        bb_index = get_backbone_index(top)  # (B, 4)

        # data split
        valid_length = list(range(T - delta))
        train_len = int(0.8 * len(valid_length))
        train_idx = np.random.choice(valid_length[:train_len], 2000)
        valid_idx = np.random.choice(valid_length[train_len:], 400)

        for _type, idx in zip(["train", "valid"], [train_idx, valid_idx]):
            split_data = []
            for i in idx:
                # get protein pairs
                x0, x1 = xyz[i], xyz[i + delta]
                x0c = x0.mean(axis=0)
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

            with open(os.path.join(save_dir, f"{_type}_mul.pkl"), "wb") as f:
                pickle.dump(split_data, f)


def save_xtc(split_path):
    items = load_file(split_path)
    for item in tqdm(items):
        pdb = item["pdb"]
        save_dir = os.path.split(item["state0_path"])[0]
        state0 = md.load(item["state0_path"])
        top = state0.topology
        traj_npz = np.load(item["traj_npz_path"])
        xyz = traj_npz["positions"]  # (T, N, 3), nm
        md.Trajectory(
            xyz,
            top
        ).save_xtc(os.path.join(save_dir, f"{pdb}-sim.xtc"))


class PepDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        super(PepDataset, self).__init__()
        with open(file_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MultiPepDataset(torch.utils.data.Dataset):
    def __init__(self, datalist_path, mode="train"):
        super(MultiPepDataset, self).__init__()
        datalist = load_file(datalist_path)
        data = []
        for data_summary in datalist:
            data_pkl = os.path.join(os.path.split(data_summary["state0_path"])[0], f"{mode}_mul.pkl")
            with open(data_pkl, "rb") as f:
                data.append(pickle.load(f))
        self.data = data

    def __len__(self):
        return sum(len(data) for data in self.data)

    def __getitem__(self, idx):
        file_idx, sample_idx = idx
        return self.data[file_idx][sample_idx]


class MultiPepBatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_indices = list(range(len(data_source.data)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.file_indices)
        for file_idx in self.file_indices:
            file_data = self.data_source.data[file_idx]
            if len(file_data) < self.batch_size:
                continue
            indices = list(range(len(file_data)))
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) < self.batch_size:
                    continue
                yield [(file_idx, idx) for idx in batch_indices]

    def __len__(self):
        return sum(len(data) // self.batch_size for data in self.data_source.data)


class MultiPepDistributedBatchSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.batch_size = batch_size
        self.dataset = dataset
        self.file_indices = list(range(len(dataset.data)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.file_indices)
        total_files = len(self.file_indices)
        files_per_rank = (total_files + self.num_replicas - 1) // self.num_replicas

        rank_start = self.rank * files_per_rank
        rank_end = min(rank_start + files_per_rank, total_files)
        assigned_files = self.file_indices[rank_start:rank_end]

        for file_idx in assigned_files:
            file_data = self.dataset.data[file_idx]
            if len(file_data) < self.batch_size:
                continue
            indices = list(range(len(file_data)))
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) < self.batch_size:
                    continue
                yield [(file_idx, idx) for idx in batch_indices]

    def __len__(self):
        return sum(len(data) // self.batch_size for data in self.dataset.data) // self.num_replicas


def collate_fn(batch):
    keys = ["atype", "rtype", "rmask", "bb_index", "x0", "x1", "force0", "force1"]
    types = [torch.long] * 4 + [torch.float] * 4
    res = {}
    # collate batch elements
    for key, _type in zip(keys, types):
        val = []
        for item in batch:
            val.append(torch.tensor(item[key], dtype=_type))
        res[key] = torch.cat(val, dim=0)
    # special for potential
    pot0, pot1 = [], []
    for item in batch:
        pot0.append(torch.tensor(item["pot0"], dtype=torch.float))
        pot1.append(torch.tensor(item["pot1"], dtype=torch.float))
    res["pot0"], res["pot1"] = torch.tensor(pot0).unsqueeze(1), torch.tensor(pot1).unsqueeze(1)
    abid, rbid = [], []
    i = 0
    for item in batch:
        abid.extend([i] * item["atype"].shape[0])
        rbid.extend([i] * item["bb_index"].shape[0])
        i += 1
    res["abid"] = torch.tensor(abid, dtype=torch.long)  # (N,)
    res["rbid"] = torch.tensor(rbid, dtype=torch.long)  # (B,)
    return res


def make_batch(pdb, bs):
    state0 = md.load(pdb)
    top = state0.topology
    xyz = 10 * state0.xyz[0] # (N, 3), Angstrom

    atype = get_atype(top)  # (N,)
    # rtype = get_rtype(top)  # (N,)
    rmask = get_res_mask(top)  # (N,)
    # bb_index = get_backbone_index(top)  # (B, 4)

    # to tensor
    atype = torch.from_numpy(atype).long()
    # rtype = torch.from_numpy(rtype).long()
    rmask = torch.from_numpy(rmask).long()
    # bb_index = torch.from_numpy(bb_index).long()

    x0 = torch.from_numpy(xyz).float()  # (N, 3)
    x0 = x0 - x0.mean(dim=0)    # CoM

    # batch id
    abid = torch.tensor([[i] * atype.shape[0] for i in range(bs)], dtype=torch.long).flatten()
    # rbid = torch.tensor([[i] * bb_index.shape[0] for i in range(bs)], dtype=torch.long).flatten()

    batch = {
        "atype": atype.repeat(bs),
        # "rtype": rtype.repeat(bs),
        "rmask": rmask.repeat(bs),
        # "bb_index": bb_index.repeat(bs, 1),
        "x0": x0.repeat(bs, 1),
        "abid": abid,
        # "rbid": rbid
    }

    return batch


def parse():
    arg_parser = ArgumentParser(description='curate dataset')
    arg_parser.add_argument('--sim_dir', type=str, required=True, help='openmm simulation directory')
    arg_parser.add_argument('--delta', type=int, default=500, help='time interval between training pairs, unit: ps')
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse()
    ### curate Peptide data
    split_paths = split_pep(args.sim_dir)
    for split_path in split_paths:
        preprocess_pep(split_path, delta=args.delta)
        save_xtc(split_path)
