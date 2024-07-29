#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
from argparse import ArgumentParser
from collections import Counter
from Bio import SeqIO
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm
from cluster import seq_cluster

import sys
sys.path.append('..')
from utils import load_file, url_get

parser = PDBParser(QUIET=True)

arg_parser = ArgumentParser(description='download full pdb data')
arg_parser.add_argument('--fasta', type=str, required=True, help='Path to fasta file')
arg_parser.add_argument('--save_dir', type=str, required=True, help='Saving directory for raw PDB files and output json file')
arg_parser.add_argument('--n_cpu', type=int, default=8, help='Number of cpu to use')
args = arg_parser.parse_args()

raw_pdb_dir = os.path.join(args.save_dir, 'PDB')
pep_pdb_dir = os.path.join(args.save_dir, 'pep')


def fetch_from_pdb(identifier, tries=5):
    # example identifier: 1FBI

    identifier = identifier.upper()
    url = 'https://data.rcsb.org/rest/v1/core/entry/' + identifier

    res = url_get(url, tries)
    if res is None:
        return None

    url = f'https://files.rcsb.org/download/{identifier}.pdb'

    text = url_get(url, tries)
    data = res.json()
    data['pdb'] = text.text
    return data


def create_and_fix_pdb(sequence, save_path):
    # create peptide, random set coordinates
    mol = Chem.MolFromSequence(sequence)

    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    with open(save_path, 'w') as file:
        file.write(Chem.MolToPDBBlock(mol))

    # fix
    fixer = PDBFixer(filename=save_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(save_path, 'w'), keepIds=True)


def fasta_process(fasta):
    items = []
    sequences = SeqIO.parse(fasta, "fasta")

    for seq_record in sequences:
        pdb = seq_record.id[:4]     # PDB id
        chain = seq_record.id[5]    # chain id
        if "mol:protein" not in seq_record.description or len(seq_record.seq) > 6:
            continue
        item = {
            "pdb": pdb,
            "chain": chain,
            "seq_len": len(seq_record.seq),
            "seq": str(seq_record.seq),
        }
        items.append(item)

    return items


def download_one_item(item):
    pdb_id = item['pdb']
    try:
        pdb_data = fetch_from_pdb(pdb_id)
    except:
        pdb_data = None
    if pdb_data is None:
        print(f'{pdb_id} invalid')
        item = None
    else:
        pdb_fout = os.path.join(raw_pdb_dir, item['pdb'] + '.pdb')
        with open(pdb_fout, 'w') as pfout:
            pfout.write(pdb_data['pdb'])
        item['pdb_path'] = os.path.abspath(pdb_fout)
    return item


def download(items, save_dir, ncpu=8):
    os.makedirs(raw_pdb_dir, exist_ok=True)
    out_path = os.path.join(save_dir, 'summary.jsonl')

    map_func = download_one_item
    print('downloading raw files')
    valid_entries = thread_map(map_func, items, max_workers=ncpu)
    valid_entries = [item for item in valid_entries if item is not None]
    print(f'number of downloaded entries: {len(valid_entries)}')

    # # select chain
    # print('select target chain from protein')
    # for item in tqdm(valid_entries):
    #     target_fout = os.path.join(raw_pdb_dir, f'{item["pdb"]}_{item["chain"]}.pdb')
    #     structure = parser.get_structure('annoy', item['pdb_path'])
    #     target_chain = [chain for chain in structure.get_chains() if chain.id == item['chain']]
    #
    #     if not len(target_chain):
    #         item['pdb_path'] = None
    #     else:
    #         target_chain = target_chain[0]
    #         io = PDBIO()
    #         io.set_structure(target_chain)
    #         io.save(target_fout)
    #         item['pdb_path'] = os.path.abspath(target_fout)


    fout = open(out_path, 'w')
    for item in valid_entries:
        if item['pdb_path'] is None:
            continue
        item_str = json.dumps(item)
        fout.write(f'{item_str}\n')
    fout.close()

    return out_path


def post_process(summary):
    os.makedirs(pep_pdb_dir, exist_ok=True)
    out_path = os.path.join(os.path.split(summary)[0], 'summary-post.jsonl')
    items = load_file(summary)
    # rule out single residue > 50% and abnormal residues
    new_items = []
    for item in items:
        char_count = Counter(item['seq'])
        if max(list(char_count.values())) <= 0.5 * len(item['seq']) and 'X' not in item['seq']:
            new_items.append(item)
    # cluster based on sequence
    clu_items = seq_cluster(new_items)
    print(f'data length: {len(clu_items)}')
    with open(out_path, 'w') as fout:
        for item in tqdm(clu_items):
            pep_path = os.path.join(pep_pdb_dir, f'{item["pdb"]}_{item["chain"]}.pdb')
            try:
                create_and_fix_pdb(item['seq'], pep_path)
            except:
                continue
            item["pdb_path"] = pep_path
            item_str = json.dumps(item)
            fout.write(f'{item_str}\n')


def main():
    items = fasta_process(args.fasta)
    summary = download(items, args.save_dir, args.n_cpu)
    post_process(summary)


if __name__ == '__main__':
    main()
