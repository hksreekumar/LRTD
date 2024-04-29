# Copyright (c) 2024. Authors listed in AUTHORS.md
#
# This file is part of elPaSo-AcMoRe.
#
# elPaSo-AcMoRe is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# elPaSo-AcMoRe is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with elPaSo-AcMoRe (COPYING.txt). If not, see
# <https://www.gnu.org/licenses/>. 

### Details: cSamplingSparseGrids
### Date: 12.01.2024
### Author: Harikrishnan Sreekumar

# python imports
import numpy as np
import sys, os
import scipy.io
from joblib import Parallel, delayed

# project modules
from scipy.interpolate import RBFInterpolator
from pysgpp import *
from pysgpp.extensions.datadriven.uq.operations import (hierarchize,
                                                        evalSGFunction, evalSGFunctionMulti)

class cSamplingSparseGrids:
    # @brief init
    # @note unit-tested
    def __init__(self, _dim=2, _level=3):
        print(f'> SparseGrid initializing dim {_dim} | level {_level}')

        ## create a two-dimensional piecewise bi-linear grid
        self.dim = int(_dim)
        #self.grid = Grid.createModLinearGrid(self.dim)
        self.grid = Grid.createModLinearGrid(self.dim)
        HashGridStorage = self.grid.getStorage()
        print(f">> dimensionality:                   {self.dim}")

        # create regular grid, level _level
        level = int(_level)
        gridGen = self.grid.getGenerator()
        gridGen.regular(level)
        print(f">> number of initial grid points:    {HashGridStorage.getSize()}")
        
        # refinement level
        self.num_refine = 0

    # @brief returns the coordinates in current grid
    # @note unit-tested
    def get_grid_coordinates(self):
        HashGridStorage = self.grid.getStorage()
        coords = np.zeros((HashGridStorage.getSize(), HashGridStorage.getDimension()))
        for i in range(HashGridStorage.getSize()):
            gp = HashGridStorage.getPoint(i)
            for idim in range(HashGridStorage.getDimension()):
                coords[i,idim] = gp.getStandardCoordinate(idim)

        return coords

    # @brief returns _num_training number of training points
    # @note unit-tested
    def establish_random_training_grid(self, _num_training):
        file = './AcMoRe_sgpp' + str(self.dim) + '_' + str(_num_training) + '.mat'
        if os.path.exists(file) and "pytest" not in sys.modules:
            samples = scipy.io.loadmat(file)['samples']
            print('>> training grid loaded from file')
        else:
            samples = np.random.rand(_num_training, self.dim)
            # x1 = np.linspace(0,1,_num_training)
            # x2 = np.linspace(0,1,_num_training)
            # xx,yy = np.meshgrid(x1,x2)
            # samples = np.vstack([xx.ravel(), yy.ravel()]).T
            #scipy.io.savemat(file, dict(samples=samples))
            
        return samples
    
    # @brief exports the passed grid to grid file and data file
    # @note unit-tested
    def export_grid_data(self):
        _filename = 'AcMoRe_sgpp_grid'
        self.grid_filename = _filename + '.grid'
        self.data_filename = _filename + '.mat'

        # get current coordinates
        coords = self.get_grid_coordinates()

        ## grid file
        f = open(self.grid_filename, "w")
        f.write(self.grid.serialize())
        f.close()
        ## data file
        scipy.io.savemat(self.data_filename, dict(num_refine=self.num_refine, coords=coords))

        print(f'> Grid exported in file {self.grid_filename}')

    # @brief refines the grid
    # @param functional_value is the evaluated function at grid coordinates
    # @note unit-tested
    def perform_grid_refinement_dimension_adaptive(self, functional_value, _refine=1):
        curr_refine = self.num_refine
        
        HashGridStorage = self.grid.getStorage()
        old_size = HashGridStorage.getSize()
        gridGen = self.grid.getGenerator()
        # refine
        alpha = DataVector(HashGridStorage.getSize())
        #print("length of alpha vector:           {}".format(alpha.getSize()))

        for refnum in range(_refine):
            print(f' > pysgpp diad refine step {curr_refine+refnum+1}...')

            # evaluate function values on grid
            f_on_interpolated_grid = np.array(functional_value).flatten()
            for i_grid in range(alpha.getSize()):
                alpha[i_grid] = float(np.abs(f_on_interpolated_grid[i_grid]))

            # perform hierarchization to obtain surplus values
            createOperationHierarchisation(self.grid).doHierarchisation(alpha)

            #refinement  stuff
            refinement = HashRefinement()
            decorator = SubspaceRefinement(refinement)
            functor = SurplusRefinementFunctor(alpha,1)
            decorator.free_refine(HashGridStorage,functor)

            #gridGen.refine(SurplusVolumeRefinementFunctor(alpha, 1))
            print(f"     >> pysgpp Refinement step {curr_refine+refnum+1}, new grid size: {HashGridStorage.getSize()}. Added {HashGridStorage.getSize()-old_size} points")
            self.num_refine = self.num_refine + 1

        return self.get_grid_coordinates()
    
    # @brief refines the grid in file and exports the new refined grid
    def perform_grid_refinement_spatially_adaptive(self, functional_value, error_values_training, training_grid, _refine=1):
        curr_refine = self.num_refine
        
        HashGridStorage = self.grid.getStorage()
        old_size = HashGridStorage.getSize()
        gridGen = self.grid.getGenerator()
                
        # refine
        alpha = DataVector(HashGridStorage.getSize())

        errorVector = DataVector(error_values_training.flatten())
        trainSet = DataMatrix(training_grid)

        for refnum in range(_refine):
            print(f' > pysgpp spad refine step {curr_refine+refnum+1}...')

            # evaluate function values on grid
            f_on_interpolated_grid = np.array(functional_value).flatten()
            for i_grid in range(f_on_interpolated_grid.shape[0]):
                alpha[i_grid] = float(f_on_interpolated_grid[i_grid])
                
            createOperationHierarchisation(self.grid).doHierarchisation(alpha)
            #print(f'hi 2alpha {alpha.toString()}')

            # perform refinement
            refinement = HashRefinement()
            decorator = PredictiveRefinement(refinement)
            indicator = PredictiveRefinementIndicator(self.grid,trainSet,errorVector,1)
            decorator.free_refine(HashGridStorage,indicator)

            print(f' > WARNING! SpAD refinement is performing an additional dimensional refinement...')
            gridGen.refine(SurplusRefinementFunctor(alpha, 1))
            print(f"     >> pysgpp Refinement step {curr_refine+refnum+1}, new grid size: {HashGridStorage.getSize()}. Added {HashGridStorage.getSize()-old_size} points")
            self.num_refine = self.num_refine + 1

        return self.get_grid_coordinates()

    # @brief function perform evaluation inside grid
    # @param functional_value is the evaluated function at grid coordinates
    # @note unit-tested
    def perform_grid_evaluation(self, functional_value, to_evaluate_grid, run_serial=True, run_sgpp_routine=True):
        HashGridStorage = self.grid.getStorage()
        grid = self.grid

        # load data
        evaluate_functional_value = functional_value
        n_dof = evaluate_functional_value.shape[0]
        n_eval = to_evaluate_grid.shape[0]
        n_d = to_evaluate_grid.shape[1]
        # assert number of grids and functions
        if HashGridStorage.getSize() != evaluate_functional_value.shape[1]:
            print(f'!! sgpp: wrong data for interpolation - size of grid {self.get_grid_coordinates().shape} and size of f {evaluate_functional_value.shape}')
            exit(-1)
        #print(evaluate_functional_value.shape)
        #print(f'knots: size{get_grid_coordinates(HashGridStorage).shape}: {get_grid_coordinates(HashGridStorage)}')
        #print(f'evaluate_functional_value: size{evaluate_functional_value.shape}: {evaluate_functional_value}')
        #print(f'to_evaluate_grid: size{to_evaluate_grid.shape}: {to_evaluate_grid}')

        # perform interpolation
        interpolated_f = np.zeros((n_dof, n_eval), dtype=complex)
        
        self.export_grid_data()

        coords = self.get_grid_coordinates()
        if run_sgpp_routine: # rbf require dim+1 number of nodes
            if run_serial:
                for i_dof in range(n_dof):
                    alpha_real = np.real(evaluate_functional_value[i_dof, :])
                    alpha_real = hierarchize(grid, alpha_real)
                    real_part = evalSGFunctionMulti(grid, alpha_real, np.atleast_2d( to_evaluate_grid))

                    alpha_imag = np.imag(evaluate_functional_value[i_dof, :])
                    if (alpha_imag == 0).all():
                        imag_part = np.zeros_like(real_part)
                    else:
                        alpha_imag = hierarchize(grid, alpha_imag)
                        imag_part = evalSGFunctionMulti(grid, alpha_imag, np.atleast_2d( to_evaluate_grid))

                    interpolated_f[i_dof, :] = real_part + 1j*imag_part
            else:
                parallel_manager = cParallelManager(evaluate_functional_value, self.grid_filename, coords, to_evaluate_grid)
                interpolated_f = parallel_manager.perform_parallel(interpolated_f, n_dof)
        else:
            #print(to_evaluate_grid.shape)
            if run_serial:
                for i_dof in range(n_dof):
                    interpolated_f[i_dof, :] = RBFInterpolator(coords, evaluate_functional_value[i_dof, :])(to_evaluate_grid)
            else:
                parallel_manager = cParallelManager(evaluate_functional_value, self.grid_filename, coords, to_evaluate_grid)
                interpolated_f = parallel_manager.perform_parallel_rbf(interpolated_f, n_dof)

        return np.array(interpolated_f)


class cParallelManager:
    def __init__(self, report_evaluate_functional_value, grid_filename, coords, report_to_evaluate_grid) -> None:
        self.report_evaluate_functional_value = report_evaluate_functional_value
        self.coords = coords
        self.grid_filename = grid_filename
        self.report_to_evaluate_grid = report_to_evaluate_grid

    def perform_parallel(self, interpolated_f, n_dof):
        num_cores = int(12)#int(multiprocessing.cpu_count())
        interpolated_f = Parallel(n_jobs=num_cores)(delayed(self.parallelised_function_grideval_dofwise)(i) for i in range(n_dof))
        #pool = multiprocessing.Pool(4)
        #interpolated_f = zip(*pool.map(self.parallelised_function_grideval_dofwise, range(n_dof)))
        return interpolated_f
    
    def perform_parallel_rbf(self, interpolated_f, n_dof):
        num_cores = int(48)#int(multiprocessing.cpu_count())
        interpolated_f = Parallel(n_jobs=num_cores)(delayed(self.parallelised_function_rbfeval_dofwise)(i) for i in range(n_dof))
        return interpolated_f

    def parallelised_function_grideval_dofwise(self, i_dof):
        grid = Grid.unserializeFromFile(self.grid_filename)
        alpha_real = np.real(self.report_evaluate_functional_value[i_dof, :])
        alpha_real = hierarchize(grid, alpha_real)
        real_part = evalSGFunctionMulti(grid, alpha_real, np.atleast_2d( self.report_to_evaluate_grid))

        alpha_imag = np.imag(self.report_evaluate_functional_value[i_dof, :])
        if (alpha_imag == 0).all():
            imag_part = np.zeros_like(real_part)
        else:
            alpha_imag = hierarchize(grid, alpha_imag)
            imag_part = evalSGFunctionMulti(grid, alpha_imag, np.atleast_2d( self.report_to_evaluate_grid))
        return real_part + 1j*imag_part

    def parallelised_function_rbfeval_dofwise(self, i_dof):
        return RBFInterpolator(self.coords, self.report_evaluate_functional_value[i_dof, :])(self.report_to_evaluate_grid)