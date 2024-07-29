import numpy as np
from mendeleev import element


ELEMENT_TYPE = ['H', 'C', 'N', 'O', 'S']
ATOM_TYPE = [
    'N', 'CA', 'C', 'O', 'OXT', 'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2', 'CE', 'NZ', 'OG', 'OG1', 'SG', 'ND1',
    'CE1', 'CE2', 'CZ', 'CH2', 'NE', 'NE1', 'NE2', 'ND2', 'OD1', 'OD2', 'OE1', 'OE2', 'OH', 'SD', 'CE3', 'CZ2',
    'CZ3', 'NH1', 'NH2', 'SE',
    'H', 'H2', 'H3', 'HA', 'HA2', 'HA3', 'HB', 'HB1', 'HB2', 'HB3', 'HG', 'HG1', 'HG11', 'HG12', 'HG13', 'HG2', 'HG21',
    'HG22', 'HG23', 'HG3', 'HD', 'HD1', 'HD11', 'HD12', 'HD13', 'HD2', 'HD21', 'HD22', 'HD23', 'HD3', 'HE',
    'HE1', 'HE2', 'HE21', 'HE22', 'HE3', 'HH', 'HH11', 'HH12', 'HH2', 'HH21', 'HH22', 'HZ', 'HZ1', 'HZ2', 'HZ3'
]
ACE_GROUP = {
    "C":    "C",
    "CH3": "CB",
    "H1": "HB1",
    "H2": "HB2",
    "H3": "HB3",
    "O":    "O",
}
NME_GROUP = {
    "N":    "N",
    "H":    "H",
    "C":   "CB",
    "H1": "HB1",
    "H2": "HB2",
    "H3": "HB3",
}
NUM_ATOM_TYPE = len(ATOM_TYPE)

RES_TYPE_3 = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'TYR', 'ASP', 'HIS', 'ASN', 'GLU',
              'LYS', 'GLN', 'MET', 'ARG', 'SER', 'THR', 'CYS', 'PRO']
RES_TYPE_1 = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'H', 'N', 'E', 'K', 'Q', 'M', 'R', 'S',
              'T', 'C', 'P']
NUM_RES_TYPE = len(RES_TYPE_3)


def get_atype(top):
    """
    :param top: mdtraj topology
    :return: atom type (index of ATOM_TYPE): (N,)
    """
    atype = []
    for atom in top.atoms:
        if atom.residue.name == 'ACE':
            atom_index = ATOM_TYPE.index(ACE_GROUP[atom.name])
        elif atom.residue.name == 'NME':
            atom_index = ATOM_TYPE.index(NME_GROUP[atom.name])
        else:
            atom_index = ATOM_TYPE.index(atom.name)
        # atom_index = ELEMENT_TYPE.index(atom.element.symbol)
        atype.append(atom_index)
    atype = np.array(atype, dtype=np.compat.long)
    return atype


def get_rtype(top):
    """
    :param top: mdtraj topology
    :return: residue type of each atom (index of RES_TYPE_3): (N,)
    """
    rtype = []
    for atom in top.atoms:
        residue_index = RES_TYPE_3.index(atom.residue.name)
        rtype.append(residue_index)
    rtype = np.array(rtype, dtype=np.compat.long)
    return rtype


def get_res_mask(top):
    """
    :param top: mdtraj topology
    :return: residue mask: (N,)
    """
    rmask = [atom.residue.index for atom in top.atoms]
    rmask = np.array(rmask, dtype=np.compat.long)
    return rmask


def get_backbone_index(top):
    """
    :param top: mdtraj topology
    :return: backbone index of each residue, order: (N, CA, C, O), shape: (B, 4)
    """
    bb_index = []
    for residue in top.residues:
        backbone = [residue.atom(atom_name) for atom_name in ['N', 'CA', 'C', 'O'] if
                    residue.atom(atom_name) is not None]
        bb_index.append([atom.index for atom in backbone])
    bb_index = np.array(bb_index, dtype=np.compat.long)
    return bb_index
