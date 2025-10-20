# -*- coding: utf-8 -*-
"""
================================================================================
The Dragon Simulation: Capturing the Turbulent Breather
Version: 17.2 (The True Scientific Target)
Author: Martin Weysi
Date: October 14, 2025

Description:
This script represents the definitive simulation run of the entire project. It is
based on the final, correct insight: the stable, pulsating, chaotic state found
in the v16.2 simulation is not a failure, but the primary discovery.

All previous attempts to force a "clean" single droplet have failed, proving
that this dynamic state is the natural attractor of the system.

This simulator's sole purpose is to run a very long, high-quality simulation
using the exact parameters of v16.2 to generate a rich dataset—the "specimen"—
for our most advanced analysis engine (v22.1). This is the definitive data
generation step for the final paper.

Key Features:
1.  **The Correct Physics and Parameters:** Uses the non-local model with the
    exact parameter set from the successful v16.2 run.
2.  **Long-Duration Capture:** The simulation time is massively increased to
    capture hundreds of oscillation cycles of the "turbulent breather."
3.  **Comprehensive Data Logging:** Leverages the full data logging capabilities
    (particle number, energy components) to create a rich dataset for deep
    analysis.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import os
import time

class DynamicStateSimulator:
    """
    An object-oriented framework to simulate and save the data for the
    emergent turbulent breather state of the non-local polariton model.
    """
    OUTPUT_FILE = 'dynamic_state_output_v17.2.npz'

    def __init__(self, phys_params, sim_params):
        self.phys_params = phys_params
        self.sim_params = sim_params
        self.grid = self._setup_simulation_grid()
        print(f"--- DynamicStateSimulator v17.2 Initialized ---")

    def _setup_simulation_grid(self):
        N = self.sim_params['N']; L = self.sim_params['L']
        r_1d = np.linspace(-L/2, L/2, N, endpoint=False)
        k_1d = 2 * np.pi * np.fft.fftfreq(N, d=(L/N))
        x, y = np.meshgrid(r_1d, r_1d); R = np.sqrt(x**2 + y**2)
        Kx, Ky = np.meshgrid(k_1d, k_1d); K2 = Kx**2 + Ky**2
        K2_poisson = K2.copy(); K2_poisson[0, 0] = 1.0
        return {'x': x, 'y': y, 'R': R, 'K2': K2, 'K2_poisson': K2_poisson, 'N': N, 'L': L, 'dx': L/N}

    def _prepare_initial_state(self):
        R = self.grid['R']; w0 = self.phys_params['w0']
        pump_profile = self.phys_params['P0'] * np.exp(-R**2 / w0**2)
        noise = (np.random.rand(self.grid['N'], self.grid['N']) - 0.5) * 1e-3
        psi = (noise + 1j*noise).astype(np.complex128)
        chi = pump_profile / (self.phys_params['gamma_r'] + 1e-12)
        print("--- Initial state prepared with quantum noise ---")
        return psi, chi, pump_profile

    def run_real_time_simulation(self):
        print("================================================================")
        print("=== v17.2: The Dragon Hunt - Capturing the Dynamic State     ===")
        print("================================================================")
        
        psi, chi, pump_profile = self._prepare_initial_state()
        K2, K2_p = self.grid['K2'], self.grid['K2_poisson']
        dx = self.grid['dx']
        dt, total_steps, log_interval = self.sim_params['dt_real'], self.sim_params['total_steps'], self.sim_params['log_interval']
        
        history_steps = total_steps // log_interval
        self.time_points = np.zeros(history_steps)
        self.particle_number_history = np.zeros(history_steps)
        self.kinetic_energy_history = np.zeros(history_steps)
        self.local_potential_energy_history = np.zeros(history_steps)
        self.nonlocal_potential_energy_history = np.zeros(history_steps)
        
        linear_propagator_psi = np.exp(-1j * K2 * dt / 2.0)
        start_time = time.time()
        log_idx = 0
        
        for i in range(1, total_steps + 1):
            psi_k = fftshift(fft2(psi)); psi_k = linear_propagator_psi * psi_k; psi = ifft2(ifftshift(psi_k))
            psi_sq = np.abs(psi)**2
            
            psi_sq_k = fftshift(fft2(psi_sq))
            r_nl_sq = self.phys_params['r_nonlocal']**2
            phi_k = self.phys_params['g_nonlocal'] * psi_sq_k / (1 + r_nl_sq * K2_p)
            phi_potential = ifft2(ifftshift(phi_k)).real
            local_potential = self.phys_params['lambda'] * psi_sq
            V_total = local_potential + phi_potential
            
            nonlinear_op = -(V_total - self.phys_params['omega']) - 1j*(self.phys_params['kappa']*chi - self.phys_params['gamma_c'])
            psi = np.exp(1j * nonlinear_op * dt) * psi
            
            psi_k = fftshift(fft2(psi)); psi_k = linear_propagator_psi * psi_k; psi = ifft2(ifftshift(psi_k))
            
            res_chi = pump_profile - (self.phys_params['gamma_r'] + self.phys_params['kappa']*np.abs(psi)**2) * chi
            chi += dt * res_chi

            if i % log_interval == 0 and log_idx < history_steps:
                current_particles = np.sum(np.abs(psi)**2) * dx**2
                
                psi_k_full = fftshift(fft2(psi))
                kinetic_energy = np.sum(K2 * np.abs(psi_k_full)**2 / (self.grid['N']**2)) * dx**2 / 2
                local_pot_energy = np.sum(self.phys_params['lambda'] * (np.abs(psi)**4)/2) * dx**2
                nonlocal_pot_energy = np.sum(phi_potential * np.abs(psi)**2)/2 * dx**2
                
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
        """Saves all relevant simulation data to a single compressed NPZ file for analysis."""
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
        print("--- Save complete. Analysis can now be performed using 'deep_analysis_engine_v22.1.py'. ---")

if __name__ == "__main__":
    # --- The parameters from the successful and insightful v16.2 run ---
    phys_params = {
        'mu_sq': 1.0, 
        'lambda': 1.0,
        'kappa': 1.0, 'gamma_c': 0.01, 'gamma_r': 2.0,
        'P0': 1.0,
        'w0': 2.5,
        'omega': -2.0,
        'g_nonlocal': 1.5,
        'r_nonlocal': 4.0,
    }
    
    sim_params = {
        'N': 256, 
        'L': 40.0,
        'dt_real': 0.01,
        'total_steps': 300000,      # A very long simulation to capture the dynamics
        'log_interval': 1000
    }

    simulator = DynamicStateSimulator(phys_params, sim_params)
    simulator.run_real_time_simulation()

