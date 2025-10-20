# -*- coding: utf-8 -*-
"""
================================================================================
Campaign A - The Revolutionary Test: Thermal Self-Trapping
Version: 15.3 (Physics Upgrade: Three-Field Model)
Author: Martin Weysi
Date: October 13, 2025

Description:
This script marks the transition from "good science" to "revolutionary science."
Instead of just simulating a droplet with calibrated parameters, we now attempt
to explain the *specific physical mechanism* of self-trapping observed in the
target experimental paper (Ballarini et al., PRL 2019).

The key insight from that paper is that self-trapping is a THERMAL effect.
This script fundamentally upgrades the model by incorporating this physics.

Key Upgrades:
1.  **New Physical Field (T):** A new scalar field, T(r,t), is introduced to
    represent the local lattice temperature.
2.  **Heat Equation:** A third coupled equation is added to model the dynamics of T,
    including a heating term from reservoir decay and a cooling term from
    thermal diffusion.
3.  **Thermal Potential:** The condensate equation is modified to include a new
    attractive potential term, V_thermal = -alpha * T, which directly couples
    the condensate to the lattice temperature.
4.  **Three-Field Model:** The simulation now evolves three coupled fields
    (psi, chi, T), making it a much more complex and physically rich model.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import time

class ThermalSelfTrapSimulator:
    """
    An object-oriented framework for the upgraded three-field (psi, chi, T) model.
    """

    def __init__(self, phys_params, sim_params):
        self.phys_params = phys_params
        self.sim_params = sim_params
        self.grid = self._setup_simulation_grid()
        print("--- ThermalSelfTrapSimulator v15.3 (Three-Field Model) Initialized ---")

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
        T = np.zeros_like(psi, dtype=float) # Lattice starts at base temperature (T=0)
        print("--- Initial state prepared (psi, chi, T) ---")
        return psi, chi, T, pump_profile

    def run_real_time_simulation(self):
        print("================================================================")
        print("=== Campaign A, v15.3: The Revolutionary Test                ===")
        print("================================================================")
        
        psi, chi, T, pump_profile = self._prepare_initial_state()
        K2 = self.grid['K2']; dt = self.sim_params['dt_real']
        total_steps = self.sim_params['total_steps']; log_interval = self.sim_params['log_interval']
        
        history_steps = total_steps // log_interval
        self.time_points = np.zeros(history_steps)
        self.particle_number_history = np.zeros(history_steps)
        
        linear_propagator_psi = np.exp(-1j * K2 * dt / 2.0)
        # Propagator for the thermal diffusion equation
        linear_propagator_T = np.exp(-self.phys_params['D_T'] * K2 * dt)

        start_time = time.time()
        log_idx = 0
        
        for i in range(1, total_steps + 1):
            # --- Evolve one step in REAL TIME for the THREE fields ---
            
            # --- PSI evolution (split-step) ---
            psi_k = fftshift(fft2(psi)); psi_k = linear_propagator_psi * psi_k; psi = ifft2(ifftshift(psi_k))
            psi_sq = np.abs(psi)**2
            # NEW: Add the thermal potential to the condensate's energy
            thermal_potential = -self.phys_params['alpha_T'] * T
            V_deriv = -self.phys_params['mu_sq'] + 2*self.phys_params['lambda']*psi_sq + 3*self.phys_params['eta']*psi_sq**2
            nonlinear_op = -(V_deriv + thermal_potential - self.phys_params['omega']) - 1j*(self.phys_params['kappa']*chi - self.phys_params['gamma_c'])
            psi = np.exp(1j * nonlinear_op * dt) * psi
            psi_k = fftshift(fft2(psi)); psi_k = linear_propagator_psi * psi_k; psi = ifft2(ifftshift(psi_k))

            # --- CHI evolution (Euler step) ---
            res_chi = pump_profile - (self.phys_params['gamma_r'] + self.phys_params['kappa']*np.abs(psi)**2) * chi
            chi += dt * res_chi

            # --- T evolution (split-step for diffusion) ---
            # Source term (heating from non-radiative reservoir decay)
            heating_term = self.phys_params['gamma_nonrad_r'] * chi
            T += heating_term * dt
            # Diffusion step (cooling)
            T_k = fftshift(fft2(T)); T_k = linear_propagator_T * T_k; T = ifft2(ifftshift(T_k))
            # Ensure temperature is non-negative
            T = T.real.clip(min=0)

            if i % log_interval == 0 and log_idx < history_steps:
                current_particles = np.sum(np.abs(psi)**2) * self.grid['dx']**2
                self.time_points[log_idx] = i * dt
                self.particle_number_history[log_idx] = current_particles
                print(f"Time: {i*dt:8.2f} | Particles: {current_particles:10.4e} | Max Temp: {np.max(T):.2f}")
                log_idx += 1
        
        end_time = time.time()
        print(f"\n--- Simulation Complete --- \nTotal elapsed time: {end_time - start_time:.2f} seconds.")
        
        self.final_psi, self.final_chi, self.final_T = psi, chi, T
        
    def analyze_and_plot(self):
        print("\n--- Analysis and Plotting of the Three-Field State ---")
        psi_density = np.abs(self.final_psi)**2
        N, L = self.grid['N'], self.grid['L']
        
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1.2])

        # --- Plot 1: Time Evolution ---
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.time_points, self.particle_number_history, color='navy', lw=2)
        ax1.set_xlabel('Time'); ax1.set_ylabel('Total Particle Number (N)')
        ax1.set_title('Time Evolution', weight='bold'); ax1.grid(True, linestyle='--'); ax1.set_yscale('log')

        # --- Plot 2: 2D Condensate Density ---
        ax2 = fig.add_subplot(gs[1, 0])
        im2 = ax2.imshow(psi_density, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='inferno')
        ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_title('Final State: Condensate Density $|\psi|^2$')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # --- Plot 3: 2D Temperature Profile ---
        ax3 = fig.add_subplot(gs[1, 1])
        im3 = ax3.imshow(self.final_T, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='hot')
        ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_title('Final State: Lattice Temperature T')
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # --- Plot 4: Radial Profiles ---
        ax4 = fig.add_subplot(gs[2, :])
        r_axis = self.grid['r_1d'][N//2:]
        radial_psi = psi_density[N//2, N//2:]
        radial_T = self.final_T[N//2, N//2:]
        
        color = 'tab:blue'
        ax4.set_xlabel('Radius (r)'); ax4.set_ylabel(r'$|\psi|^2$', color=color)
        ax4.plot(r_axis, radial_psi, color=color, lw=3, label='Condensate')
        ax4.tick_params(axis='y', labelcolor=color); ax4.set_ylim(bottom=0)
        
        ax5 = ax4.twinx()
        color = 'tab:orange'
        ax5.set_ylabel('Temperature T', color=color)
        ax5.plot(r_axis, radial_T, color=color, lw=3, linestyle='-.', label='Temperature')
        ax5.tick_params(axis='y', labelcolor=color); ax5.set_ylim(bottom=0)

        ax4.set_title('Final State: Radial Profiles'); ax4.grid(True, linestyle=':')
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.3))
        
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Parameters for the three-field model, including new thermal parameters
    phys_params = {
        'mu_sq': 1.0, 'lambda': 0.6, 'eta': 0.1,
        'kappa': 1.0, 'gamma_c': 0.01, 'gamma_r': 2.0,
        'P0': 2.0, # Start with a moderate pump
        'w0': 8.0, # A wider pump to see the trapping effect clearly
        'D': 0.05, 'omega': -2.0,
        # --- NEW THERMAL PARAMETERS ---
        'alpha_T': 2.5,          # Strength of the thermal potential (how much temperature traps polaritons)
        'gamma_nonrad_r': 0.5,   # Rate of non-radiative decay from reservoir that causes heating
        'D_T': 0.1               # Thermal diffusion coefficient (how fast heat spreads)
    }
    
    sim_params = {
        'N': 256, 'L': 50.0,
        'dt_real': 0.01,
        'total_steps': 100000,
        'log_interval': 500
    }

    simulator = ThermalSelfTrapSimulator(phys_params, sim_params)
    simulator.run_real_time_simulation()
    simulator.analyze_and_plot()
