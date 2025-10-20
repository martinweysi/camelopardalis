# -*- coding: utf-8 -*-
"""
================================================================================
The Non-Local Model: A New Beginning
Version: 16.0 (Physics Revolution: Non-Local Interactions)
Author: Martin Weysi
Date: October 14, 2025

Description:
This script marks a major turning point in the project, born from the "glorious
failure" of the v15 series. The inability of any local potential to stabilize
the system in the realistic regime was not a failure of the model, but a
profound scientific discovery: local interactions are not enough.

This version introduces the critical missing piece of physics: non-local
interactions. The strong, short-range repulsion is now counter-balanced by a
long-range repulsion mediated by the photonic component of the polaritons.

This is implemented by coupling the GPE to a Poisson-like equation, a standard
and powerful method in the field. This is our most physically accurate and
promising model to date.

Key Upgrades:
1.  **New Physics Engine:** The complex local potential is replaced by a simple
    cubic repulsion coupled to a new field, Phi, representing the non-local
    interaction potential.
2.  **Poisson Solver:** A fast Fourier-space solver for the Poisson-like
    equation for Phi is integrated into the main loop.
3.  **Return to a Simpler, More Powerful Model:** By identifying the correct
    physical mechanism, we can discard the complex, fine-tuned potentials
    of the previous versions.
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
        print("--- NonLocalSimulator v16.0 Initialized ---")

    def _setup_simulation_grid(self):
        N = self.sim_params['N']; L = self.sim_params['L']
        r_1d = np.linspace(-L/2, L/2, N)
        k_1d = 2 * np.pi * np.fft.fftfreq(N, d=(r_1d[1] - r_1d[0]))
        x, y = np.meshgrid(r_1d, r_1d); R = np.sqrt(x**2 + y**2)
        Kx, Ky = np.meshgrid(k_1d, k_1d); K2 = Kx**2 + Ky**2
        # For the Poisson solver: k^2 operator in Fourier space. Avoid division by zero at k=0.
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
        print("=== v16.0: The Non-Local Interaction Test                    ===")
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
            
            # --- NEW PHYSICS: Non-local potential ---
            # 1. Source for the non-local potential is the condensate density
            psi_sq_k = fftshift(fft2(psi_sq))
            # 2. Solve the Poisson-like equation in Fourier space: (r_nl^2 * k^2 + 1) * Phi_k = g_nl * psi_sq_k
            r_nl_sq = self.phys_params['r_nonlocal']**2
            phi_k = self.phys_params['g_nonlocal'] * psi_sq_k / (1 + r_nl_sq * K2_p)
            # 3. Transform back to real space to get the potential
            phi_potential = ifft2(ifftshift(phi_k)).real
            
            # Local potential is just the simple cubic term now
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
        if final_change < 1e-3:
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
    phys_params = {
        'mu_sq': 1.0, 
        'lambda': 1.0,           # Strong LOCAL repulsion
        'kappa': 1.0, 'gamma_c': 0.01, 'gamma_r': 2.0,
        'P0': 2.0, 'w0': 8.0, 
        'omega': -2.0,
        # --- NEW NON-LOCAL PARAMETERS ---
        'g_nonlocal': 1.5,      # Strength of the long-range repulsion
        'r_nonlocal': 4.0,      # Range of the non-local interaction
    }
    
    sim_params = {
        'N': 256, 'L': 60.0,
        'dt_real': 0.01,
        'total_steps': 150000,
        'log_interval': 500
    }

    simulator = NonLocalSimulator(phys_params, sim_params)
    simulator.run_real_time_simulation()
    simulator.analyze_and_plot()

