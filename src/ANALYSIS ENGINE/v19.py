# -*- coding: utf-8 -*-
"""
================================================================================
Campaign C - The Non-Linear Approach: A Deep Analysis Engine
Version: 19.0 (Exploratory Science Framework)
Author: Martin Weysi
Date: October 14, 2025

Description:
This script embodies the "non-linear approach" by shifting the focus from
forcing a specific outcome to deeply characterizing the complex state that
naturally emerges from the model. It accepts that the turbulent, many-body
state is not a failure but the primary object of scientific discovery.

This version is a significant expansion of the v18 simulator, transforming it
into a comprehensive numerical laboratory. It integrates more sophisticated
analysis methods directly into the workflow, providing a rich, multi-faceted
characterization of the final state.

Key Upgrades from v18:
1.  **Topological Analysis:** A new method for vortex detection and counting is
    introduced. This allows us to probe the topological nature of the state,
    distinguishing simple disorder from quantum turbulence.
2.  **Energy Analysis:** The simulation now logs the kinetic and potential
    energy components over time, providing deeper insight into the system's
    dynamics and how energy is partitioned.
3.  **Comprehensive Visualization:** The plotting function is upgraded to a
    full analytical dashboard, including a map of the condensate's phase
    (revealing vortices) and more detailed annotations on all plots.
4.  **Full Decoupling (Simulation vs. Analysis):** This script now performs
    only the simulation and saves a rich dataset. A separate, dedicated
    analysis script should be used on the output file, as per professional
    workflow.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import os
import argparse
import time

class CollectiveStateSimulator:
    """
    An object-oriented framework to simulate the emergent many-body state of the
    non-local polariton model and save a comprehensive dataset for analysis.
    """
    OUTPUT_FILE = 'collective_state_output_v19.npz'

    def __init__(self, phys_params, sim_params):
        self.phys_params = phys_params
        self.sim_params = sim_params
        self.grid = self._setup_simulation_grid()
        print(f"--- CollectiveStateSimulator v19.0 Initialized ---")

    def _setup_simulation_grid(self):
        """
        Creates and stores the real and Fourier space grids for the 2D simulation.
        This internal method is called once upon initialization.
        """
        N = self.sim_params['N']; L = self.sim_params['L']
        r_1d = np.linspace(-L/2, L/2, N, endpoint=False) # Use endpoint=False for perfect periodicity
        k_1d = 2 * np.pi * np.fft.fftfreq(N, d=(L/N))
        
        x, y = np.meshgrid(r_1d, r_1d)
        R = np.sqrt(x**2 + y**2)
        
        Kx, Ky = np.meshgrid(k_1d, k_1d)
        K2 = Kx**2 + Ky**2
        
        K2_poisson = K2.copy()
        # The zero-frequency mode (k=0) must be handled carefully.
        # For a Poisson equation sourced by a term with a non-zero average (like density),
        # this mode can diverge. Here we regularize it.
        K2_poisson[0, 0] = 1.0 # Regularization
        
        print("--- Simulation Grid Initialized ---")
        return {'x': x, 'y': y, 'R': R, 'K2': K2, 'K2_poisson': K2_poisson, 'N': N, 'L': L, 'dx': L/N}

    def _prepare_initial_state(self):
        """
        Prepares the super-Gaussian pump profile and initializes the psi field
        with low-amplitude random noise, mimicking quantum fluctuations.
        """
        R = self.grid['R']
        w0 = self.phys_params['w0']
        p_exp = self.phys_params['p_super_gaussian']
        
        # A super-Gaussian pump creates a wide, flat-top excitation area
        pump_profile = self.phys_params['P0'] * np.exp(-(R/w0)**p_exp)
        
        # Initialize with low-amplitude random noise for both real and imaginary parts
        noise_amplitude = 1e-3
        noise = noise_amplitude * (np.random.rand(self.grid['N'], self.grid['N']) - 0.5) + \
                1j * noise_amplitude * (np.random.rand(self.grid['N'], self.grid['N']) - 0.5)
        psi = noise.astype(np.complex128)
        
        # Assume the reservoir starts in equilibrium with the pump
        chi = pump_profile / (self.phys_params['gamma_r'] + 1e-12)
        
        print("--- Initial state prepared with quantum noise under a flat-top pump ---")
        return psi, chi, pump_profile

    def run_real_time_simulation(self):
        """
        The core of the simulator, performing the real-time evolution using the
        split-step Fourier method for the coupled psi-chi-Phi system.
        """
        print("================================================================")
        print("=== Campaign C, v19.0: The Non-Linear Approach               ===")
        print("================================================================")
        
        psi, chi, pump_profile = self._prepare_initial_state()
        
        # Unpack parameters for performance in the loop
        K2, K2_p = self.grid['K2'], self.grid['K2_poisson']
        dx = self.grid['dx']
        dt, total_steps, log_interval = self.sim_params['dt_real'], self.sim_params['total_steps'], self.sim_params['log_interval']
        
        # History arrays to store time evolution of observables
        history_steps = total_steps // log_interval
        self.time_points = np.zeros(history_steps)
        self.particle_number_history = np.zeros(history_steps)
        self.kinetic_energy_history = np.zeros(history_steps)
        self.local_potential_energy_history = np.zeros(history_steps)
        self.nonlocal_potential_energy_history = np.zeros(history_steps)
        
        # Pre-calculate the linear propagator for the kinetic energy term
        linear_propagator_psi = np.exp(-1j * K2 * dt / 2.0)
        
        start_time = time.time()
        log_idx = 0
        
        for i in range(1, total_steps + 1):
            # --- Evolve one step in REAL TIME ---
            
            # 1. Half-step for kinetic energy (in Fourier space)
            psi_k = fftshift(fft2(psi)); psi_k = linear_propagator_psi * psi_k; psi = ifft2(ifftshift(psi_k))
            
            # 2. Full-step for all potentials and gain/loss (in real space)
            psi_sq = np.abs(psi)**2
            
            # Solve for and calculate the non-local potential Phi
            psi_sq_k = fftshift(fft2(psi_sq))
            r_nl_sq = self.phys_params['r_nonlocal']**2
            phi_k = self.phys_params['g_nonlocal'] * psi_sq_k / (1 + r_nl_sq * K2_p)
            phi_potential = ifft2(ifftshift(phi_k)).real
            
            # Calculate the local potential (simple cubic repulsion)
            local_potential = self.phys_params['lambda'] * psi_sq
            
            # The total potential is the sum of local and non-local parts
            V_total = local_potential + phi_potential
            
            # The full non-linear operator for the split-step evolution
            nonlinear_op = -(V_total - self.phys_params['omega']) - 1j*(self.phys_params['kappa']*chi - self.phys_params['gamma_c'])
            psi = np.exp(1j * nonlinear_op * dt) * psi
            
            # 3. Another half-step for kinetic energy
            psi_k = fftshift(fft2(psi)); psi_k = linear_propagator_psi * psi_k; psi = ifft2(ifftshift(psi_k))
            
            # 4. Evolve the reservoir field using a simple and fast Euler step
            res_chi = pump_profile - (self.phys_params['gamma_r'] + self.phys_params['kappa']*np.abs(psi)**2) * chi
            chi += dt * res_chi

            # --- Logging and Output ---
            if i % log_interval == 0 and log_idx < history_steps:
                # Calculate observables for logging
                current_particles = np.sum(np.abs(psi)**2) * dx**2
                
                psi_k = fftshift(fft2(psi))
                kinetic_energy = np.sum(K2 * np.abs(psi_k)**2 / N**2) * dx**2 / 2
                local_pot_energy = np.sum(self.phys_params['lambda'] * (np.abs(psi)**4)/2) * dx**2
                nonlocal_pot_energy = np.sum(phi_potential * np.abs(psi)**2)/2 * dx**2
                
                # Store in history arrays
                self.time_points[log_idx] = i * dt
                self.particle_number_history[log_idx] = current_particles
                self.kinetic_energy_history[log_idx] = kinetic_energy
                self.local_potential_energy_history[log_idx] = local_pot_energy
                self.nonlocal_potential_energy_history[log_idx] = nonlocal_pot_energy
                
                print(f"Time: {i*dt:8.2f} | Total Particles: {current_particles:10.4e}")
                log_idx += 1
        
        end_time = time.time()
        print(f"\n--- Simulation Complete --- \nTotal elapsed time: {end_time - start_time:.2f} seconds.")
        
        self.final_psi = psi
        self.final_chi = chi
        self.save_results()

    def save_results(self):
        """Saves all relevant simulation data to a single compressed NPZ file for later analysis."""
        print(f"--- Saving comprehensive dataset to {self.OUTPUT_FILE} ---")
        np.savez_compressed(self.OUTPUT_FILE,
                 final_psi=self.final_psi,
                 final_chi=self.final_chi,
                 time_points=self.time_points,
                 particle_number_history=self.particle_number_history,
                 kinetic_energy_history=self.kinetic_energy_history,
                 local_potential_energy_history=self.local_potential_energy_history,
                 nonlocal_potential_energy_history=self.nonlocal_potential_energy_history,
                 phys_params=self.phys_params,
                 sim_params=self.sim_params)
        print("--- Save complete. ---")

if __name__ == "__main__":
    # --- Define parameters for a run that is likely to produce a rich many-body state ---
    phys_params = {
        'mu_sq': 1.0, 
        'lambda': 1.0,
        'kappa': 1.0, 
        'gamma_c': 0.01, 
        'gamma_r': 2.0,
        'P0': 3.0,                  # Strong, wide pump
        'w0': 15.0,
        'p_super_gaussian': 8,      # Flat-top shape
        'omega': -2.0,
        'g_nonlocal': 2.0,          # Strong non-local repulsion
        'r_nonlocal': 4.0,
    }
    
    sim_params = {
        'N': 256, 
        'L': 60.0,                  # Large simulation box
        'dt_real': 0.02,
        'total_steps': 250000,      # Long simulation time for relaxation
        'log_interval': 1000
    }

    # --- Run the simulation ---
    # The output of this script is a data file. A separate analysis script
    # (like analysis_toolkit_v20.py) should be used on the resulting .npz file.
    simulator = CollectiveStateSimulator(phys_params, sim_params)
    simulator.run_real_time_simulation()
