from argparse import ArgumentParser
import openmm as mm
from openmm.app import *
from openmm import *
from tqdm import tqdm
import os
import json
import numpy as np


def load_file(fpath):
    with open(fpath, 'r') as fin:
        lines = fin.read().strip().split('\n')
    items = [json.loads(s) for s in lines]
    return items


forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')


class Spacing:
    """A policy that determines when to record trajectory data."""

    def stepsUntilNextReport(self, currentStep):
        raise NotImplementedError("Derived classes need to implement stepsUntilNextReport method.")


class RegularSpacing(Spacing):
    """Regular spacing, every `reportInterval` steps."""

    def __init__(self, reportInterval):
        """Create a regular spacing.

        Parameters
        ----------
        reportInterval : int
            The interval (in time steps) at which to write frames.
        """
        super(RegularSpacing, self).__init__()
        self._reportInterval = reportInterval

    def stepsUntilNextReport(self, currentStep):
        """Return the number of steps until the next reporting step."""
        steps = self._reportInterval - currentStep % self._reportInterval
        return steps


class NPZReporter(object):
    """NPZReporter outputs positions, velocities, and forces for each frame.

    To use, create a NPZReporter, then add it to the Simulation's list of
    reporters.

    The created NPZ file will contain the following arrays:
      * 'time': (T,) array, simulation time in picoseconds.
      * 'energies': (T,2) array, each row containing [potential, kinetic]
        energies in kJ/mol.
      * 'positions': (T,num_atoms,3) array, positions in nm.
      * 'velocities': (T,num_atoms,3) array, velocities in nm/ps.
      * 'forces': (T,num_atoms,3) array, forces in kJ/(mol nm).
    """

    def __init__(self, filename, spacing, atom_indices=None):
        """Create a NPZReporter.

        Parameters
        ----------
        filename : string
            The filename to write to, should end with '.npz'.
        spacing : Spacing
            The report spacing at which to write frames.
        atom_indices : Range or List or None
            The list of atoms to record in that order in the NPZ file.
            If None, all atom coordinates are saved.
        """
        self._filename = filename
        self._spacing = spacing
        self._atom_indices = atom_indices
        self._nextModel = 0
        self._positions = []
        self._velocities = []
        self._forces = []
        self._energies = []
        self._time = []
        self._step = []

    def describeNextReport(self, simulation):
        steps = self._spacing.stepsUntilNextReport(simulation.currentStep)
        return steps, True, True, True, True, None  # PVFE

    def filter_atoms(self, data):
        if self._atom_indices:
            data = data[self._atom_indices, :]
        return data

    def report(self, simulation, state):
        self._time.append(state.getTime().value_in_unit(unit.picoseconds))
        self._step.append(simulation.currentStep)
        self._energies.append(
            [
                state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
                state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole),
            ]
        )

        # Positions
        positions = state.getPositions(asNumpy=True)
        positions = positions.value_in_unit(unit.nanometer)
        positions = positions.astype(np.float32)
        positions = self.filter_atoms(positions)
        self._positions.append(positions)

        # Velocities
        velocities = state.getVelocities(asNumpy=True)
        velocities = velocities.value_in_unit(unit.nanometer / unit.picosecond)
        velocities = velocities.astype(np.float32)
        velocities = self.filter_atoms(velocities)
        self._velocities.append(velocities)

        # Forces
        forces = state.getForces(asNumpy=True)
        forces = forces.value_in_unit(unit.kilojoules / (unit.mole * unit.nanometer))
        forces = forces.astype(np.float32)
        forces = self.filter_atoms(forces)
        self._forces.append(forces)

    def __del__(self):
        # Save all trajectory data to the NPZ file
        step = np.array(self._step)
        time = np.array(self._time)
        energies = np.array(self._energies)

        positions = np.stack(self._positions)
        velocities = np.stack(self._velocities)
        forces = np.stack(self._forces)

        np.savez_compressed(
            self._filename,
            step=step,
            time=time,
            energies=energies,
            positions=positions,
            velocities=velocities,
            forces=forces,
        )


def openmm_simulate(pdb_path, save_path, T=300, spacing=1000, gpu=-1):
    summary_path = os.path.join(save_path, 'stats.txt')
    pdb_name = os.path.split(pdb_path)[1].split('.')[0]
    pdb = PDBFile(pdb_path)

    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens()
    modeller.deleteWater()

    protein_atoms = len(modeller.positions)
    print("Pre-processed protein has %d atoms." % protein_atoms)
    # Write state0 file
    PDBFile.writeFile(modeller.topology, modeller.positions, open(os.path.join(save_path, 'state0.pdb'), "w"))

    system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    # T=300K, integrator time step=1fs
    integrator = LangevinMiddleIntegrator(T * openmm.unit.kelvin,
                                          1.0 / openmm.unit.picosecond,
                                          1.0 * openmm.unit.femtosecond)
    if gpu == -1:
        simulation = Simulation(modeller.topology, system, integrator)
    else:
        platform = Platform.getPlatformByName('CUDA')
        properties = {'DeviceIndex': f'{gpu}'}
        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)

    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(T)

    # frame spacing=1ps
    # simfile = os.path.join(save_path, f'{pdb_name}-sim.pdb')
    # simulation.reporters.append(PDBReporter(simfile, spacing))
    # save NPZ file (energies, positions, velocities, forces)
    trajnpzfile = os.path.join(save_path, f'{pdb_name}-traj-arrays.npz')
    simulation.reporters.append(
        NPZReporter(trajnpzfile, RegularSpacing(spacing), atom_indices=range(protein_atoms))
    )
    simulation.reporters.append(StateDataReporter(summary_path, spacing, step=True, elapsedTime=True,
                                                  potentialEnergy=True))

    # simulation time=100ns
    simulation.step(1e8)

    del simulation


def get_openmm_simulation(topology, T=300, gpu=-1):
    system = forcefield.createSystem(topology, nonbondedMethod=NoCutoff, constraints=None)
    # integrator time step=1fs
    integrator = LangevinMiddleIntegrator(T * openmm.unit.kelvin,
                                          1.0 / openmm.unit.picosecond,
                                          1.0 * openmm.unit.femtosecond)
    if gpu == -1:
        simulation = Simulation(topology, system, integrator)
    else:
        platform = Platform.getPlatformByName('CUDA')
        properties = {'DeviceIndex': f'{gpu}'}
        simulation = Simulation(topology, system, integrator, platform, properties)

    return simulation


def get_potential(topology, positions, T=300, gpu=-1, simulation=None):
    if not simulation:
        simulation = get_openmm_simulation(topology, T, gpu)

    simulation.context.setPositions(positions)

    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy().value_in_unit(
            openmm.unit.kilojoule / openmm.unit.mole)

    return potential


def get_force(topology, positions, T=300, gpu=-1, simulation=None):
    if not simulation:
        simulation = get_openmm_simulation(topology, T, gpu)

    simulation.context.setPositions(positions)

    state = simulation.context.getState(getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules / (unit.mole * unit.nanometer)).astype(np.float32)

    return forces


def spring_constraint_energy_minim(simulation, positions):
    spring_constant = 10.0 * unit.kilocalories_per_mole / unit.angstroms ** 2
    restraint = mm.CustomExternalForce('0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)')
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')
    restraint.addGlobalParameter('k', spring_constant)

    for atom in simulation.topology.atoms():
        if atom.element.symbol != 'H':
            index = atom.index
            position = positions[index]
            restraint.addParticle(index, [position[0], position[1], position[2]])

    simulation.system.addForce(restraint)
    simulation.context.setPositions(positions)
    tolerance = (2.39 * unit.kilocalories_per_mole / unit.angstroms ** 2)\
        .value_in_unit(unit.kilojoules_per_mole / unit.nanometers ** 2)
    simulation.minimizeEnergy(tolerance=tolerance)

    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True)\
        .value_in_unit(unit.nanometer)\
        .astype(np.float32)
    return positions


def parse():
    arg_parser = ArgumentParser(description='simulation')
    arg_parser.add_argument('--summary', type=str, required=True, help='Path to summary file')
    arg_parser.add_argument('--temp', type=float, default=300, help='simulation temperature, default: 300K')
    arg_parser.add_argument('--spacing', type=int, default=1000, help='frame spacing, unit: fs')
    arg_parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return arg_parser.parse_args()


def main(args):
    save_dir = os.path.join(os.path.split(args.summary)[0], 'sim')
    os.makedirs(save_dir, exist_ok=True)
    items = load_file(args.summary)
    for item in tqdm(items):
        pdb = item['pdb']
        item_dir = os.path.join(save_dir, pdb)
        os.makedirs(item_dir, exist_ok=True)
        openmm_simulate(item['pdb_path'], item_dir, T=args.temp, gpu=args.gpu)


if __name__ == "__main__":
    main(parse())
