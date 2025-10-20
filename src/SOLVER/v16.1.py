# -*- coding: utf-8 -*-
"""
================================================================================
The Non-Local Model: Single Droplet Creation
Version: 16.1 (Controlled Creation Experiment)
Author: Martin Weysi
Date: October 13, 2025

Description:
This script builds on the critical success of v16.0, which demonstrated that the
non-local model successfully produces a stable, non-trivial steady state,
thereby taming the instabilities of the local models.

The result of v16.0 was a complex, turbulent/multi-peaked state, not a single
droplet. This script tests the hypothesis that a single, clean droplet can be
formed by using a more focused, less powerful pump, thereby favoring the
nucleation of a single structure over a collective state.

This is the final experiment of Campaign A, aiming to produce the clean,
publication-ready plot for Figure 2 using the correct, non-local physics.

Key Upgrades:
1.  **Targeted Pumping:** The pump width (w0) is significantly reduced, and the
    power (P0) is adjusted to create conditions favorable for a single droplet.
2.  **Refined Goal:** The goal is explicitly to generate a single, stable,
    localized condensate, providing the definitive success case for the model.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import time

class NonLocalSimulator:
    """
    Simulates the dynamics of a polariton condensate with non-local interactions.
    """

    def __init__(self, phys_params, sim_params):
        self.phys_params = phys_params
        self.sim_params = sim_params
        self.grid = self._setup_simulation_grid()
        print("--- NonLocalSimulator v16.1 (Single Droplet Target) Initialized ---")

    def _setup_simulation_grid(self):
        N = self.sim_params['N']; L = self.sim_params['L']
        r_1d = np.linspace(-L/2, L/2, N)
        k_1d = 2 * np.pi * np.fft.fftfreq(N, d=(r_1d[1] - r_1d[0]))
        x, y = np.meshgrid(r_1d, r_1d); R = np.sqrt(x**2 + y**2)
        Kx, Ky = np.meshgrid(k_1d, k_1d); K2 = Kx**2 + Ky**2
        K2_poisson = K2.copy(); K2_poisson[0, 0] = 1.0 
        return {'x': x, 'y': y, 'R': R, 'K2': K2, 'K2_poisson': K2_poisson, 'N': N, 'L': L, 'r_1d': r_1d, 'dx': r_1d[1]-r_1d[0]}

    def _prepare_initial_state(self):
        R = self.grid['R']; w0 = self.phys_params['w0']
        pump_profile = self.phys_params['P0'] * np.exp(-R**2 / w0**2)
        noise = (np.random.rand(self.grid['N'], self.grid['N']) - 0.5) * 1e-4
        psi = (noise + 1j*noise).astype(np.complex128)
        chi = pump_profile / (self.phys_params['gamma_r'] + 1e-12)
        print("--- Initial state prepared with quantum noise ---")
        return psi, chi, pump_profile

    def run_real_time_simulation(self):
        print("================================================================")
        print("=== v16.1: The Controlled Single Droplet Creation Test       ===")
        print("================================================================")
        
        psi, chi, pump_profile = self._prepare_initial_state()
        K2 = self.grid['K2']; K2_p = self.grid['K2_poisson']; R = self.grid['R']
        dt = self.sim_params['dt_real']; total_steps = self.sim_params['total_steps']
        log_interval = self.sim_params['log_interval']
        
        history_steps = total_steps // log_interval
        self.time_points = np.zeros(history_steps)
        self.particle_number_history = np.zeros(history_steps)
        
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
                current_particles = np.sum(np.abs(psi)**2) * self.grid['dx']**2
                self.time_points[log_idx] = i * dt
                self.particle_number_history[log_idx] = current_particles
                print(f"Time: {i*dt:8.2f} | Particles: {current_particles:10.4e}")
                log_idx += 1
        
        end_time = time.time()
        print(f"\n--- Simulation Complete --- \nTotal elapsed time: {end_time - start_time:.2f} seconds.")
        
        self.final_psi, self.final_chi = psi, chi
        
    def analyze_and_plot(self):
        print("\n--- Analysis and Plotting ---")
        psi_density = np.abs(self.final_psi)**2
        N, L = self.grid['N'], self.grid['L']
        r_1d = self.grid['r_1d']
        
        final_change = np.std(self.particle_number_history[-20:]) / np.mean(self.particle_number_history[-20:])
        if final_change < 1e-4:
            print(f"System has converged to a steady state (final fluctuation: {final_change:.2e}).")
        else:
            print(f"Warning: System may not have fully converged (final fluctuation: {final_change:.2e}).")

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        ax1.plot(self.time_points, self.particle_number_history, color='navy', lw=2)
        ax1.set_xlabel('Time'); ax1.set_ylabel('Total Particle Number (N)')
        ax1.set_title('Time Evolution', weight='bold'); ax1.grid(True, linestyle='--'); ax1.set_yscale('log')

        im = ax2.imshow(psi_density, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='inferno')
        ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_title('Final State: 2D Condensate Density')
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).set_label(r'$|\psi|^2$')

        radial_r_axis = r_1d[N//2:]
        radial_psi = psi_density[N//2, N//2:]
        radial_chi = self.final_chi[N//2, N//2:]
        color = 'tab:blue'
        ax3.set_xlabel('Radius (r)'); ax3.set_ylabel(r'Condensate Density $|\psi|^2$', color=color)
        ax3.plot(radial_r_axis, radial_psi, color=color, lw=3)
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.grid(True, linestyle='--'); ax3.set_ylim(bottom=0); ax3.set_title('Final State: Radial Profile')
        ax4 = ax3.twinx()
        color = 'tab:red'
        ax4.set_ylabel(r'Reservoir Density $\chi$', color=color)
        ax4.plot(radial_r_axis, radial_chi, color=color, lw=3, linestyle='--')
        ax4.tick_params(axis='y', labelcolor=color); ax4.set_ylim(bottom=0)

        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Parameters are adjusted for single droplet formation
    phys_params = {
        'mu_sq': 1.0, 
        'lambda': 1.0,
        'kappa': 1.0, 'gamma_c': 0.01, 'gamma_r': 2.0,
        'P0': 1.5,         # <<< REDUCED PUMP POWER
        'w0': 4.0,         # <<< NARROWER PUMP
        'omega': -2.0,
        'g_nonlocal': 1.5,
        'r_nonlocal': 4.0,
    }
    
    sim_params = {
        'N': 256, 'L': 40.0, # Smaller box is fine for a single droplet
        'dt_real': 0.01,
        'total_steps': 100000, # Shorter time should be enough
        'log_interval': 500
    }

    simulator = NonLocalSimulator(phys_params, sim_params)
    simulator.run_real_time_simulation()
    simulator.analyze_and_plot()
