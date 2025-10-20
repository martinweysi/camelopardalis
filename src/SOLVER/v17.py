# -*- coding: utf-8 -*-
"""
================================================================================
The Final Experiment: Controlled Creation in the Non-Local Model
Version: 17.0 (The Synthesis)
Author: Martin Weysi
Date: October 14, 2025

Description:
This script represents the synthesis of all lessons learned throughout Campaign A.
We have established two critical facts:
1.  Local models (v15.x series) are fundamentally incapable of producing stable
    droplets in the realistic parameter regime.
2.  The non-local model (v16.x series), while physically correct, is prone to
    chaotic, multi-peaked states when subjected to a sudden, strong pump from
    a noisy initial state.

This version combines the correct physics (non-local model from v16) with the
correct methodology (controlled creation from v16.4). It tests the final,
most refined hypothesis:

"A single, stable droplet IS a stable attractor of the non-local model,
provided the system is gently guided into that state via seeding and
adiabatic pumping, rather than being shocked into chaos."

This is the definitive test to produce the clean, single-droplet result for
Figure 2 of the manuscript.

Key Features:
1.  **The Correct Physics:** Implements the non-local interaction model.
2.  **The Correct Methodology:** Uses a clean Gaussian seed and an adiabatic
    pump ramp to avoid shock and chaos.
3.  **The Final Question:** This simulation provides a definitive yes/no answer
    to the existence of a stable single-droplet state in this model.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import time

class FinalSimulator:
    """
    A simulator that combines the non-local physics with a controlled
    seeding and adiabatic pumping methodology.
    """

    def __init__(self, phys_params, sim_params):
        self.phys_params = phys_params
        self.sim_params = sim_params
        self.grid = self._setup_simulation_grid()
        print("--- FinalSimulator v17.0 (The Synthesis) Initialized ---")

    def _setup_simulation_grid(self):
        N = self.sim_params['N']; L = self.sim_params['L']
        r_1d = np.linspace(-L/2, L/2, N)
        k_1d = 2 * np.pi * np.fft.fftfreq(N, d=(r_1d[1] - r_1d[0]))
        x, y = np.meshgrid(r_1d, r_1d); R = np.sqrt(x**2 + y**2)
        Kx, Ky = np.meshgrid(k_1d, k_1d); K2 = Kx**2 + Ky**2
        K2_poisson = K2.copy(); K2_poisson[0, 0] = 1.0 
        return {'x': x, 'y': y, 'R': R, 'K2': K2, 'K2_poisson': K2_poisson, 'N': N, 'L': L, 'r_1d': r_1d, 'dx': r_1d[1]-r_1d[0]}

    def _prepare_initial_state(self):
        """ Initializes the system with a clean Gaussian seed and zero pump. """
        R = self.grid['R']
        seed_amplitude = self.sim_params['seed_amplitude']
        seed_width = self.sim_params['seed_width']
        psi = seed_amplitude * np.exp(-R**2 / seed_width**2)
        psi = psi.astype(np.complex128)
        
        pump_profile = np.zeros_like(R)
        chi = np.zeros_like(R)
        
        print(f"--- Initial state prepared with a Gaussian seed (amp={seed_amplitude}, width={seed_width}) ---")
        return psi, chi

    def run_real_time_simulation(self):
        print("================================================================")
        print("=== v17.0: The Final Experiment - Controlled Creation        ===")
        print("================================================================")
        
        psi, chi = self._prepare_initial_state()
        K2 = self.grid['K2']; K2_p = self.grid['K2_poisson']
        dt = self.sim_params['dt_real']; total_steps = self.sim_params['total_steps']
        log_interval = self.sim_params['log_interval']
        
        history_steps = total_steps // log_interval
        self.time_points = np.zeros(history_steps)
        self.particle_number_history = np.zeros(history_steps)
        
        linear_propagator_psi = np.exp(-1j * K2 * dt / 2.0)
        start_time = time.time()
        log_idx = 0
        
        P_final = self.phys_params['P0']
        ramp_time = self.sim_params['ramp_duration']
        
        for i in range(1, total_steps + 1):
            current_time = i * dt
            
            if current_time < ramp_time:
                p_val = P_final * (current_time / ramp_time)
            else:
                p_val = P_final
                
            pump_profile = p_val * np.exp(-self.grid['R']**2 / self.phys_params['w0']**2)

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
                self.time_points[log_idx] = current_time
                self.particle_number_history[log_idx] = current_particles
                print(f"Time: {current_time:8.2f} | Pump: {p_val:.2f} | Particles: {current_particles:10.4e}")
                log_idx += 1
        
        end_time = time.time()
        print(f"\n--- Simulation Complete --- \nTotal elapsed time: {end_time - start_time:.2f} seconds.")
        
        self.final_psi, self.final_chi = psi, chi
        
    def analyze_and_plot(self):
        print("\n--- Analysis and Plotting ---")
        psi_density = np.abs(self.final_psi) ** 2
        N, L, r_1d = self.grid['N'], self.grid['L'], self.grid['r_1d']
        
        final_change = np.std(self.particle_number_history[-50:]) / np.mean(self.particle_number_history[-50:])
        if final_change < 1e-4:
            print(f"System has converged to a stable steady state (final fluctuation: {final_change:.2e}).")
        else:
            print(f"Warning: System has not fully stabilized (final fluctuation: {final_change:.2e}).")

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        ax1.plot(self.time_points, self.particle_number_history, color='navy', lw=2, label='Particle Number (N)')
        ax1.set_xlabel('Time'); ax1.set_ylabel('Total Particle Number (N)')
        ax1.set_title('Time Evolution with Adiabatic Pump', weight='bold'); ax1.grid(True, linestyle='--'); ax1.set_yscale('log')
        ax1_pump = ax1.twinx()
        pump_history = self.phys_params['P0'] * np.minimum(self.time_points / self.sim_params['ramp_duration'], 1.0)
        ax1_pump.plot(self.time_points, pump_history, color='cyan', linestyle='--', label='Pump Power (P0)')
        ax1_pump.set_ylabel('Pump Power', color='cyan'); ax1_pump.tick_params(axis='y', labelcolor='cyan')
        fig.legend(loc="center right", bbox_to_anchor=(0.85, 0.75))

        im = ax2.imshow(psi_density, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='inferno')
        ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_title('Final State: 2D Condensate Density')
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).set_label(r'$|\psi|^2$')

        radial_r_axis = r_1d[N//2:]; radial_psi = psi_density[N//2, N//2:]; radial_chi = self.final_chi[N//2, N//2:]
        color = 'tab:blue'; ax3.set_xlabel('Radius (r)'); ax3.set_ylabel(r'$|\psi|^2$', color=color)
        ax3.plot(radial_r_axis, radial_psi, color=color, lw=3)
        ax3.tick_params(axis='y', labelcolor=color); ax3.grid(True, linestyle='--'); ax3.set_ylim(bottom=0); ax3.set_title('Final State: Radial Profile')
        ax4 = ax3.twinx(); color = 'tab:red'; ax4.set_ylabel(r'$\chi$', color=color)
        ax4.plot(radial_r_axis, radial_chi, color=color, lw=3, linestyle='--')
        ax4.tick_params(axis='y', labelcolor=color); ax4.set_ylim(bottom=0)
        
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    phys_params = {
        'mu_sq': 1.0, 'lambda': 1.0,
        'kappa': 1.0, 'gamma_c': 0.01, 'gamma_r': 2.0,
        'P0': 1.5, 'w0': 4.0, 'omega': -2.0,
        'g_nonlocal': 4.0, 'r_nonlocal': 5.0,
    }
    
    sim_params = {
        'N': 256, 'L': 40.0,
        'dt_real': 0.02,
        'total_steps': 100000, # total time = 2000
        'log_interval': 250,
        'ramp_duration': 800.0, # Gently ramp up the pump over a long period
        'seed_amplitude': 0.5,
        'seed_width': 3.0,
    }

    simulator = FinalSimulator(phys_params, sim_params)
    simulator.run_real_time_simulation()
    simulator.analyze_and_plot()
