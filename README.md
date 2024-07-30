# FBM

Force-Guided Bridge Matching for Full-Atom Time-Coarsened Dynamics of Peptides

### Dependencies

Our code works well on Linux with `CUDA==11.7`. It will be required to install `pytorch` and other dependencies listed below with the corresponding CUDA version (if necessary) on your server.

```
biopython
mdtraj
openmm
rdkit
torch_scatter
tqdm
e3nn
requests
```

### Alanine Dipeptide (AD) Dataset

The PDB file and coordinates of MD trajectories (.xtc format) of AD can be downloaded from [mdshare](https://markovmodel.github.io/mdshare/ALA2/#alanine-dipeptide).

### PepMD Dataset

To curate our PepMD dataset, you should first download `pdb_seqres.fasta` file from [PDB](https://files.wwpdb.org/pub/pdb/derived\_data/pdb\_seqres.txt.gz). We have provided the script to download and process raw PDB data in `data/download.py`, please run:

```bash
cd data && python download.py --fasta ./pdb_seqres.fasta --save_dir /your/saving/directory
```

where you should specify a saving directory for PDB and processed peptide files. If all scripts work successfully, there will be `summary.jsonl` and `summary-post.jsonl` in your saving directory now.

Then MD simulations can be performed with `simulation.py`:

```bash
python simulation.py --summary /your/saving/directory/summary-post.jsonl --temp 300 --spacing 1000 --gpu 0
```

where you can specify your own configurations, such as temperature `--temp` (unit: Kelvin) and frame spacing `--spacing` (unit: fs). The script will create a directory `sim` under `/your/saving/directory` where each peptide has its own sub-directory named by PDB id, including a `{PDB-id}_{chain-id}-traj-arrays.npz` file containing coordinates, velocities, forces and energies of MD trajectories and a `state0.pdb` file.

Now we can curate the dataset with train/test splits:

```bash
python dataset.py --sim_dir /your/saving/directory/sim --delta 500
```

where you can specify the coarsened time for prediction `--delta` (unit: ps). Afterwards there will be `train.jsonl` and `test.jsonl` under `/your/saving/directory` for training and evaluation.

### Training

Before running training scripts, first compile TorchMD extensions with:

```bash
python setup.py build_ext --inplace
```

Then you can use the script `train.sh` for training both FBM-base and FBM with multi GPUs. Note that you should first replace `DATA_DIR` in the file with `/your/saving/directory`. You can run the following script to train FBM-base with GPU 0, 1:

```bash
GPU=0,1 bash train.sh
```

For training FBM, please modify the configuration `--model_type SFM` to `--model_type FSFM` and add another line including `--baseline /path/to/FBM-base/checkpoint`, where you should replace with the checkpoint file path (`.ckpt`) of FBM-base.

### Evaluation

We have provided different evaluation scripts for various usage.

If you only want to inference trajectories without evaluation, please run:

```bash
python inference.py --name {any_name_for_identity} --test_set /path/to/state0.pdb --ckpt /path/to/checkpoint --save_dir /path/to/saving/results --inf_step 1000 --sde_step 30 --guidance 0.05 --gpu 0
```

where `--name` is only used to create sub-directory under `--save_dir` for saving generated trajectories. `--test_set` is the path to the initial PDB file you are interested in, `--ckpt` specifies the path to the checkpoint of FBM-base or FBM, `--inf_step` specifies the trajectory length and `--sde_step` indicates discrete-time step $T$. If you use the FBM model for inference, it's required to add `--guidance` to specify the guidance strength.

If you want to evaluate any generated trajectories with MD trajectories, please run:

```bash
python evaluate.py --top /path/to/state0.pdb --ref /path/to/MD/trajectories --model /path/to/generated/trajectories
```

Here `--top` specifies the `.pdb` file that describes the topology, `--ref` and `--model` specify trajectories generated by MD and the model respectively. We support multiple format for `--ref` and `--model`, including: `.pdb`, `.xtc`, `.npz` consisting of the key "positions", `.npy`.

If you want to inference and evaluate the model on the test set of PepMD (or any other test sets), please run:

```bash
python evaluate_all.py --name none --test_set /your/saving/directory/test.jsonl --ckpt /path/to/checkpoint --save_dir /path/to/saving/results --inf_step 1000 --sde_step 30 --guidance 0.05 --gpu 0
```

### License

MIT
