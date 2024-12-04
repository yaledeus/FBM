### Reference: https://github.com/microsoft/timewarp/blob/main/simulation/md.py

from argparse import ArgumentParser
import openmm
import openmm as mm
from openmm import app, unit
from tqdm import tqdm
import os
import json
import yaml
import numpy as np


def load_file(fpath):
    with open(fpath, 'r') as fin:
        lines = fin.read().strip().split('\n')
    items = [json.loads(s) for s in lines]
    return items


def get_default_parameters():
    default_parameters = {
        "force-field": "amber14-only",
        "integrator": "LangevinMiddleIntegrator",
        "waterbox-pad": 1.0,
        "temperature": 300,
        "timestep": 1.0,
        "friction": 1.0,
        "sampling": 100_000_000,
        "spacing": 1_000,
        "min-tol": 2.0,
        "gpu": -1
    }
    return default_parameters


def get_simulation_environment_integrator(parameters):
    """Obtain integrator from parameters.

    Arguments
    ---------
    parameters : dict or str
        Parameter dictionary or preset name.

    Returns
    -------
    integrator : openmm.Integrator
    """
    temperature = parameters["temperature"]
    friction = parameters["friction"]
    timestep = parameters["timestep"]
    if parameters["integrator"] == "LangevinIntegrator":
        integrator = mm.LangevinIntegrator(
            temperature * unit.kelvin,
            friction / unit.picosecond,
            timestep * unit.femtosecond
        )
    elif parameters["integrator"] == "LangevinMiddleIntegrator":
        # assert version.parse(mm.__version__) >= version.parse("7.5")
        integrator = mm.LangevinMiddleIntegrator(
            temperature * unit.kelvin,
            friction / unit.picosecond,
            timestep * unit.femtosecond
        )
    else:
        raise NotImplementedError(f'Integrator type {parameters["integrator"]} not implemented.')

    return integrator


def get_simulation_environment_from_model(model, parameters=None):
    """Obtain simulation environment suitable for energy computation.

    Arguments
    ---------
    model : openmm.app.modeller.Modeller
        Fully instantiated OpenMM model.
    parameters : dict or str
        Parameter dictionary or preset name.

    Returns
    -------
    simulation : openmm.Simulation
        Simulation (topology, forcefield and computation parameters).  This
        object can be passed to the compute_forces_and_energy method.
    """
    if not parameters:
        parameters = get_default_parameters()
    system = get_system(model, parameters)
    integrator = get_simulation_environment_integrator(parameters)
    if parameters["gpu"] == -1:
        simulation = mm.app.Simulation(model.topology, system, integrator)
    else:
        platform = mm.Platform.getPlatformByName('CUDA')
        properties = {'DeviceIndex': f'{parameters["gpu"]}'}
        simulation = mm.app.Simulation(model.topology, system, integrator, platform, properties)

    return simulation


def get_simulation_environment_from_pdb(pdb, parameters):
    model = get_openmm_model(pdb)
    return get_simulation_environment_from_model(model, parameters)


def get_system(model, parameters):
    """Obtain system to generate e.g. a simulation environment.

    Arguments
    ---------
    model : openmm.app.modeller.Modeller
        Fully instantiated OpenMM model.
    parameters : dict or str
        Parameter dictionary or preset name.

    Returns
    -------
    system : openmm.system
        System (topology, forcefield).  This
        is required for a simulation object.
    """
    # TODO: use openmmforcefields package to support GAFF2
    # TODO: support CHARMM36 with implicit water

    # amber99-implicit and amber14-implicit
    if parameters["force-field"].endswith("-implicit"):
        if parameters["force-field"] == "amber99-implicit":
            forcefield = mm.app.ForceField("amber99sbildn.xml", "amber99_obc.xml")
        elif parameters["force-field"] == "amber14-implicit":
            # (Onufriev, Bashford, Case, "Exploring Protein Native States and
            # Large-Scale Conformational Changes with a modified Generalized
            # Born Model", PROTEINS 2004) using the GB-OBC I parameters
            # (corresponds to `igb=2` in AMBER)
            # assert version.parse(mm.__version__) >= version.parse("7.7")
            forcefield = mm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")
        else:
            raise ValueError("Invalid forcefield parameter '%s'" % parameters["force-field"])

        model.addExtraParticles(forcefield)

        # Peter Eastman recommends a large cutoff value for implicit solvent
        # models, around 20 Angstrom (= 2nm), see
        # https://github.com/openmm/openmm/issues/3104
        system = forcefield.createSystem(
            model.topology,
            nonbondedMethod=mm.app.CutoffNonPeriodic,
            nonbondedCutoff=2.0 * unit.nanometer,  # == 20 Angstrom
            constraints=mm.app.HBonds,
        )
    elif parameters["force-field"] == "amber14-explicit":
        forcefield = mm.app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        model.addExtraParticles(forcefield)
        model.addSolvent(forcefield, padding=parameters["waterbox-pad"])

        system = forcefield.createSystem(
            model.topology,
            nonbondedMethod=mm.app.PME,  # .NoCutoff, .PME for particle mesh Ewald
            constraints=mm.app.HBonds,  # .HBonds   # constrain H-bonds (fastest vibrations)
        )
    elif parameters["force-field"] == "amber14-only":
        forcefield = mm.app.ForceField("amber14-all.xml")   # without solvation, just for engineering purpose
        model.addExtraParticles(forcefield)

        system = forcefield.createSystem(
            model.topology,
            nonbondedMethod=mm.app.NoCutoff,
            constraints=mm.app.HBonds
        )
    else:
        raise ValueError("Invalid forcefield parameter '%s'" % parameters["force-field"])

    return system


def get_openmm_model(state0pdbpath):
    """Create openmm model from pdf file.

    Arguments
    ---------
    state0pdbpath : str
        Pathname for all-atom state0.pdb file created by simulate_trajectory.

    Returns
    -------
    model : openmm.app.modeller.Modeller
        Modeller provides tools for editing molecular models, such as adding water or missing hydrogens.
        This object can also be used to create simulation environments.
    """
    pdb_file = mm.app.pdbfile.PDBFile(state0pdbpath)
    positions = pdb_file.getPositions()
    topology = pdb_file.getTopology()
    model = mm.app.modeller.Modeller(topology, positions)
    return model


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


def get_potential(simulation, positions):
    simulation.context.setPositions(positions)

    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule / unit.mole)

    return potential


def get_force(simulation, positions):
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


def simulate_trajectory(pdb_path, save_path, parameters):
    print(f"Simulation parameters: {parameters}")

    summary_path = os.path.join(save_path, 'stats.txt')
    pdb_name = os.path.split(pdb_path)[1].split('.')[0]

    model = get_openmm_model(pdb_path)
    model.addHydrogens()
    model.deleteWater()

    protein_atoms = len(model.positions)
    print("Pre-processed protein has %d atoms." % protein_atoms)
    # Write state0 file
    mm.app.pdbfile.PDBFile.writeFile(model.topology, model.positions, open(os.path.join(save_path, 'state0.pdb'), "w"))

    simulation = get_simulation_environment_from_model(model, parameters)

    simulation.context.setPositions(model.positions)

    tolerance = float(parameters["min-tol"])
    print("Performing ENERGY MINIMIZATION to tolerance %2.2f kJ/mol" % tolerance)
    simulation.minimizeEnergy(tolerance=tolerance)
    print("Completed ENERGY MINIMIZATION")

    temperature = parameters["temperature"]
    print("Initializing VELOCITIES to %s" % temperature)
    simulation.context.setVelocitiesToTemperature(temperature)

    # frame spacing=1ps
    # simfile = os.path.join(save_path, f'{pdb_name}-sim.pdb')
    # simulation.reporters.append(PDBReporter(simfile, spacing))
    spacing = parameters["spacing"]
    # save NPZ file (energies, positions, velocities, forces)
    trajnpzfile = os.path.join(save_path, f'{pdb_name}-traj-arrays.npz')
    simulation.reporters.append(
        NPZReporter(trajnpzfile, RegularSpacing(spacing), atom_indices=range(protein_atoms))
    )
    simulation.reporters.append(mm.app.StateDataReporter(summary_path, spacing, step=True, elapsedTime=True,
                                                         potentialEnergy=True))
    with open(os.path.join(save_path, "simulation_env.yaml"), 'w') as yaml_file:
        yaml.dump(parameters, yaml_file, default_flow_style=False)

    sampling = parameters["sampling"]
    print(f"Begin SAMPLING for {sampling} steps.")
    simulation.step(sampling)
    print("Completed SAMPLING")

    del simulation


def parse():
    parser = ArgumentParser(description='simulation')
    parser.add_argument('--summary', type=str, required=True, help='Path to summary file')
    parser.add_argument('--force-field', type=str, default="amber14-only",
                        choices=["amber99-implicit", "amber14-implicit", "amber14-explicit", "amber14-only"],
                        help='(preset) Force field, "amber99-implicit", "amber14-implicit", "amber14-explicit" '
                             'or "amber14-only". [default: amber14-only]')
    parser.add_argument('--integrator', type=str, default="LangevinMiddleIntegrator",
                        choices=["LangevinMiddleIntegrator", "LangevinIntegrator"])
    parser.add_argument('--waterbox-pad', type=float, default=1.0, help='Waterbox padding width in nm [default: 1.0]')
    parser.add_argument('--temperature', type=int, default=300, help='simulation temperature [default: 300K]')
    parser.add_argument('--timestep', type=float, default=1.0,
                        help='Integration time step in femtoseconds [default: 1.0]')
    parser.add_argument('--friction', type=float, default=1.0, help='Langevin friction in 1.0/ps [default: 1.0]')
    parser.add_argument('--sampling', type=int, default=100_000_000,
                        help='Number of total integration steps [default: 100_000_000].')
    parser.add_argument('--spacing', type=int, default=1_000, help='frame spacing in femtoseconds [default: 1000]')
    parser.add_argument('--min-tol', type=float, default=10.0,
                        help='Energy minimization tolerance in kJ/mol [default: 10.0].')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='whether to use CUDA to accelerate simulation, -1 for cpu and {>0} for GPU index')
    return parser.parse_args()


def main(args):
    param_keys = ["force-field", "integrator", "waterbox-pad", "temperature", "timestep", "friction",
                  "sampling", "spacing", "min-tol", "gpu"]
    parameters = {key: getattr(args, key.replace('-', '_')) for key in param_keys}
    save_dir = os.path.join(os.path.split(args.summary)[0], 'sim')
    os.makedirs(save_dir, exist_ok=True)
    items = load_file(args.summary)
    for item in tqdm(items):
        pdb = item['pdb']
        print(f"[+] Start MD simulations on pdb: {pdb}.")
        item_dir = os.path.join(save_dir, pdb)
        if os.path.exists(item_dir):
            continue
        os.makedirs(item_dir, exist_ok=True)
        simulate_trajectory(item['pdb_path'], item_dir, parameters=parameters)


if __name__ == "__main__":
    main(parse())
