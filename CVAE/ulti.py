import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals,rms
from math import pi, cos, sin
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.base import AnalysisFromFunction

def nornamlize(value,max=pi,min=-pi):
    s_val = (value-min)/(max-min)
    return s_val

def angle_to_rad(value):
    r_val = value*pi/180
    return r_val
   

def get_dihedrals_angle(selected_atoms):
    dihedrals_angle = []
    dihedrals_angle_value = []

    for residue in selected_atoms.residues:
        phi = residue.phi_selection()
        if phi:
            dihedrals_angle.append(phi)

    for residue in selected_atoms.residues:    
        psi = residue.psi_selection()
        if psi:
            dihedrals_angle.append(psi)

    for i in dihedrals_angle:
        dihedrals_angle_value.append(nornamlize(angle_to_rad(i.dihedral.value())))
    
    return len(dihedrals_angle_value), dihedrals_angle_value


def get_protein_angle(file_name,chainMain,chainSup):
    u = mda.Universe(file_name)
    
    sel_atom = 'backbone or name H or name O'
    
    select_str = chainMain + sel_atom + ' and not ' + ' ' + chainSup 
    
    selected_atoms = u.select_atoms(select_str)
    
    return get_dihedrals_angle(selected_atoms)

    

def get_protein_condition(file_name,chainID,residStar,residEnd):
    u = mda.Universe(file_name)
    
    select_str = chainID + ' and ' + 'resid '+ str(residStar) + ':' + str(residEnd) 
    
    dihedrals_angle = []
    dihedrals_angle_value = []

    selected_atoms = u.select_atoms(select_str)

    return get_dihedrals_angle(selected_atoms)


def get_rmsd(file_name,chainMain,chainSup,residStar,residEnd):
    u = mda.Universe(file_name)
    
    # Atoms from the residue section
    select_str_1 = chainMain + ' and ' + 'resid '+ str(residStar) + ':' + str(residEnd)  
    selected_atoms_1 = u.select_atoms(select_str_1)
    
    # Atoms from the second chain
    sel_atom = 'backbone or name H or name O'
    select_str_2 = chainSup + sel_atom + ' and not ' + ' ' + chainMain    
    selected_atoms_2 = u.select_atoms(select_str_2)
       
    distance = rms.rmsd(selected_atoms_1.positions,  selected_atoms_2.positions, center=True, superposition=True)
    
    return distance
   

   
    
def _expand_universe(universe, length):
    coordinates = AnalysisFromFunction(lambda ag: ag.positions.copy(),
                                       universe.atoms).run().results
    coordinates = np.tile(coordinates, (length, 1, 1))
    universe.load_new(coordinates, format=MemoryReader)


def _set_dihedral(dihedral, atoms, angle):
    current_angle = dihedral.dihedral.value()
    head = atoms[dihedral[2].id:]
    vec = dihedral[2].position - dihedral[1].position
    head.rotateby(angle-current_angle, vec, dihedral[2].position)


def dihedral_backmapping(pdb_path, dihedral_trajectory, rough_n_points=-1):

    step_size = max(1, int(len(dihedral_trajectory) / rough_n_points))
    dihedral_trajectory = dihedral_trajectory[::step_size]

    uni = mda.Universe(pdb_path)

    chainA = 'segid A'
    chainB = 'segid B'
    sel_atom = 'backbone or name H or name O'
    select_str = chainB + sel_atom + ' and not ' + ' ' + chainA
    protein_chainB = uni.select_atoms(select_str)
    
    protein = uni.select_atoms("protein")
    dihedrals = []
    
    for residue in protein_chainB.residues:
        phi = residue.phi_selection()
        if phi:
            dihedrals.append(phi)

    for residue in protein_chainB.residues:
        psi = residue.psi_selection()
        if psi:
            dihedrals.append(psi)

    _expand_universe(uni, len(dihedral_trajectory))

    for dihedral_values, step in zip(dihedral_trajectory, uni.trajectory):
        for dihedral, value in zip(dihedrals, dihedral_values):
            _set_dihedral(dihedral, protein, value / (2 * pi) * 360)

    return uni 
    
