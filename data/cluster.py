from collections import defaultdict
import re
import os
import shutil
import numpy as np


def exec_mmseq(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text


def seq_cluster(items):
    # transfer to fasta format
    tmp_dir = './tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    else:
        raise ValueError(f'Working directory {tmp_dir} exists!')
    fasta = os.path.join(tmp_dir, 'seq.fasta')
    with open(fasta, 'w') as fout:
        for item in items:
            pdb = item['pdb']
            seq = item['seq']
            fout.write(f'>{pdb}\n{seq}\n')
    db = os.path.join(tmp_dir, 'DB')
    cmd = f'mmseqs createdb {fasta} {db}'
    exec_mmseq(cmd)
    db_clustered = os.path.join(tmp_dir, 'DB_clu')
    cmd = f'mmseqs cluster {db} {db_clustered} {tmp_dir} --min-seq-id 0.6'  # similarity > 0.6 in the same cluster
    res = exec_mmseq(cmd)
    num_clusters = re.findall(r'Number of clusters: (\d+)', res)
    if len(num_clusters):
        print(f'Number of clusters: {num_clusters[0]}')
    else:
        raise ValueError('cluster failed!')
    tsv = os.path.join(tmp_dir, 'DB_clu.tsv')
    cmd = f'mmseqs createtsv {db} {db} {db_clustered} {tsv}'
    exec_mmseq(cmd)

    # read tsv of class \t pdb
    with open(tsv, 'r') as fin:
        entries = fin.read().strip().split('\n')
    pdb2clu, clu2idx = {}, defaultdict(list)
    for entry in entries:
        cluster, pdb = entry.strip().split('\t')
        pdb2clu[pdb] = cluster
    for i, item in enumerate(items):
        pdb = item['pdb']
        cluster = pdb2clu[pdb]
        clu2idx[cluster].append(i)

    clu_cnt = [len(clu2idx[clu]) for clu in clu2idx]
    print(f'cluster number: {len(clu2idx)}, member number ' +
          f'mean: {np.mean(clu_cnt)}, min: {min(clu_cnt)}, ' +
          f'max: {max(clu_cnt)}')

    shutil.rmtree(tmp_dir)

    clu_items = []
    for clu in clu2idx:
        clu_items.append(items[np.random.choice(clu2idx[clu])])

    return clu_items
