import numpy as np
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList
from ase.calculators.calculator import Calculator, all_changes

class MultiMorse(Calculator):   
    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, params, cutoff, combo='mean', **kwargs):
        Calculator.__init__(self, **kwargs)
        self.params = params
        self.combo = combo
        self.cutoff = cutoff

    def initialize(self, image):
        params_dict = self.params
        chemical_symbols = np.array(image.get_chemical_symbols())
        params = []
        for element in chemical_symbols:
            re = params_dict[element]["re"]
            D = params_dict[element]["D"]
            #sig = params_dict[element]["sig"]
            sig = re - np.log(2)/params_dict[element]["a"]
            params.append(np.array([[re, D, sig]]))
        params = np.vstack(np.array(params)) 
        return params

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        image = atoms
        
        n = NeighborList(cutoffs=[cutoff / 2.0] * len(image),
                         self_interaction=False, primitive=NewPrimitiveNeighborList)
        n.update(image)
        image_neighbors = [n.get_neighbors(index) for index in range(len(image))]
         
        params = self.initialize(image)
        
        natoms = len(image)

        positions = image.positions
        cell = image.cell

        energy = 0.0
        forces = np.zeros((natoms, 3))

        for a1 in range(natoms):
            re_1 = params[a1][0]
            D_1 = np.abs(params[a1][1])
            sig_1 = params[a1][2]
            neighbors, offsets = image_neighbors[a1]
            cells = np.dot(offsets, cell)
            d = positions[neighbors] + cells - positions[a1]
            re_n = params[neighbors][:, 0]
            D_n = params[neighbors][:, 1]
            sig_n = params[neighbors][:, 2]
            if self.combo == 'mean':
                D = np.sqrt(D_1*D_n)
                sig = (sig_1 + sig_n) / 2
                re = (re_1 + re_n) / 2
            elif self.combo == 'yang':
                D = (2 * D_1 * D_n) / (D_1 + D_n)
                sig = (sig_1 * sig_n) * (sig_1 + sig_n) / (sig_1 ** 2 + sig_n ** 2)
                re = (re_1 * re_n) * (re_1 + re_n) / (re_1 ** 2 + re_n ** 2)
            r = np.sqrt((d ** 2).sum(1))
            r_star = r / sig
            re_star = re / sig
            C = np.log(2) / (re_star - 1)
            atom_energy = D * (
                np.exp(-2 * C * (r_star - re_star))
                - 2 * np.exp(-C * (r_star - re_star))
            )
            energy += atom_energy.sum()
            f = (
                (2 * D * C / sig)
                * (1 / r)
                * (
                    np.exp(-2 * C * (r_star - re_star))
                    - np.exp(-C * (r_star - re_star))
                    ))[:, np.newaxis] * d
            forces[a1] -= f.sum(axis=0)
            for a2, f2 in zip(neighbors, f):
                forces[a2] += f2

        self.results['energy'] = energy
        self.results['forces'] = forces