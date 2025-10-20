# -*- coding: utf-8 -*-
"""
================================================================================
Campaign A - The Saturation Test
Version: 15.5 (Physics Upgrade: Strong Saturation)
Author: Martin Weysi
Date: October 13, 2025

Description:
This script is the next logical step in our scientific investigation. The
previous test (reducing P0) proved that the pump strength alone is not the
culprit for the runaway instability. This was a crucial null result.

This version formulates and tests a new hypothesis: the saturation mechanism
(the eta*|psi|^6 term) is too weak to counteract the strong repulsive
interactions (lambda) at high densities.

This is a controlled experiment. We keep the reduced pump power and the trap,
and modify only one parameter: eta.

Key Upgrades:
1.  **New Physical Hypothesis:** The sextic saturation coefficient `eta` is
    significantly increased (5x) to act as a much stronger "brake" at high
    condensate densities.
2.  **Targeted Experiment:** This simulation is no longer a blind search but a
    direct test of the role of higher-order saturation in stabilizing the
    system in the realistic, strong-interaction regime.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import time

class SaturationTestSimulator:
    """
    An object-oriented framework for testing the effect of strong saturation.
    """

    def __init__(self, phys_params, sim_params):
        self.phys_params = phys_params
        self.sim_params = sim_params
        self.grid = self._setup_simulation_grid()
        print("--- SaturationTestSimulator v15.5 (Strong Saturation) Initialized ---")

    def _setup_simulation_grid(self):
        N = self.sim_params['N']; L = self.sim_params['L']
        r_1d = np.linspace(-L/2, L/2, N)
        k_1d = 2 * np.pi * np.fft.fftfreq(N, d=(r_1d[1] - r_1d[0]))
        x, y = np.meshgrid(r_1d, r_1d); R = np.sqrt(x**2 + y**2)
        Kx, Ky = np.meshgrid(k_1d, k_1d); K2 = Kx**2 + Ky**2
        return {'x': x, 'y': y, 'R': R, 'K2': K2, 'N': N, 'L': L, 'r_1d': r_1d, 'dx': r_1d[1]-r_1d[0]}

    def _prepare_initial_state(self):
        R = self.grid['R']; w0 = self.phys_params['w0']
        pump_profile = self.phys_params['P0'] * np.exp(-R**2 / w0**2)
        noise_real = (np.random.rand(self.grid['N'], self.grid['N']) - 0.5) * 1e-4
        noise_imag = (np.random.rand(self.grid['N'], self.grid['N']) - 0.5) * 1e-4
        psi = (noise_real + 1j * noise_imag).astype(np.complex128)
        chi = pump_profile / (self.phys_params['gamma_r'] + 1e-12)
        print("--- Initial state prepared with quantum noise ---")
        return psi, chi, pump_profile

    def run_real_time_simulation(self):
        print("================================================================")
        print("=== Campaign A, v15.5: Testing the Brakes (Strong Saturation) ===")
        print("================================================================")
        
        psi, chi, pump_profile = self._prepare_initial_state()
        K2 = self.grid['K2']; R = self.grid['R']; dt = self.sim_params['dt_real']
        total_steps = self.sim_params['total_steps']; log_interval = self.sim_params['log_interval']
        
        history_steps = total_steps // log_interval
        self.time_points = np.zeros(history_steps)
        self.particle_number_history = np.zeros(history_steps)
        
        trap_potential = 0.5 * self.phys_params['V_trap'] * R**2
        linear_propagator_psi = np.exp(-1j * K2 * dt / 2.0)
        
        start_time = time.time()
        log_idx = 0
        
        for i in range(1, total_steps + 1):
            psi_k = fftshift(fft2(psi)); psi_k = linear_propagator_psi * psi_k; psi = ifft2(ifftshift(psi_k))
            psi_sq = np.abs(psi)**2
            V_deriv = -self.phys_params['mu_sq'] + 2*self.phys_params['lambda']*psi_sq + 3*self.phys_params['eta']*psi_sq**2
            nonlinear_op = -(V_deriv + trap_potential - self.phys_params['omega']) - 1j*(self.phys_params['kappa']*chi - self.phys_params['gamma_c'])
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
        print("\n--- Analysis and Plotting of the Saturated State ---")
        psi_density = np.abs(self.final_psi)**2
        N, L = self.grid['N'], self.grid['L']
        r_1d = self.grid['r_1d']
        
        # Check for convergence
        final_change = np.std(self.particle_number_history[-10:]) / np.mean(self.particle_number_history[-10:])
        if final_change < 1e-3:
            print(f"System has converged to a steady state (final fluctuation: {final_change:.2e}).")
        else:
            print(f"Warning: System may not have fully converged (final fluctuation: {final_change:.2e}).")
            
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.time_points, self.particle_number_history, color='navy', lw=2)
        ax1.set_xlabel('Time'); ax1.set_ylabel('Total Particle Number (N)')
        ax1.set_title('Time Evolution', weight='bold'); ax1.grid(True, linestyle='--'); ax1.set_yscale('log')

        ax2 = fig.add_subplot(gs[1, 0])
        im = ax2.imshow(psi_density, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='inferno')
        ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_title('Final State: 2D Condensate Density')
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).set_label(r'$|\psi|^2$')

        ax3 = fig.add_subplot(gs[1, 1])
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
    phys_params = {
        'mu_sq': 1.0, 'lambda': 0.6,
        'eta': 0.5,        # <<< THE HYPOTHESIS: STRONGER BRAKES
        'kappa': 1.0, 'gamma_c': 0.01, 'gamma_r': 2.0,
        'P0': 1.2,         # Using the reduced pump power from the last test
        'w0': 5.0, 'D': 0.05, 'omega': -2.0,
        'V_trap': 0.01
    }
    
    sim_params = {
        'N': 256, 'L': 40.0,
        'dt_real': 0.005,
        'total_steps': 150000,
        'log_interval': 500
    }

    simulator = SaturationTestSimulator(phys_params, sim_params)
    simulator.run_real_time_simulation()
    simulator.analyze_and_plot()
