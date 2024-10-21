'''
Simple diatomic Lennard-Jones calculator for ASE.
Created by Tomasz Dymkowski
Date: 21.10.2024
'''

import numpy as np

from ase.calculators.calculator import Calculator, all_changes


class LJCalculator(Calculator):

    implemented_properties = ['energy', 'forces']
    default_parameters = {
            'epsilon': 0.01,
            'sigma': 3.4
            }

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def calculate(self, atoms=None, properties=None,
                  system_changes=all_changes):

        if properties is None:
            properties = self.implemented_properties

        super().calculate(atoms, properties, system_changes)

        sigma = self.parameters.sigma
        epsilon = self.parameters.epsilon

        positions = atoms.get_positions()
        n_atoms = len(positions)
        energy = 0.0
        forces = np.zeros((n_atoms, 3))

        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                rij = positions[i] - positions[j]
                r = np.linalg.norm(rij)
                s = sigma / r
                s6 = s**6
                s12 = s6 ** 2
                e = 4 * epsilon * (s12 - s6)
                energy += e
                f_mag = 24 * epsilon * (2 * s12 - s6) / r

                forces[i] += f_mag * (rij / r)
                forces[j] -= f_mag * (rij / r)

        self.results['energy'] = energy
        self.results['forces'] = forces
