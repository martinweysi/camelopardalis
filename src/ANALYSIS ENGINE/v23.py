# -*- coding: utf-8 -*-
"""
================================================================================
Campaign C - The Ultimate Analysis Engine
Version: 23.0 (Final Diagnostic Toolkit)
Author: Martin Weysi
Date: October 14, 2025

Description:
This script is the ultimate evolution of our analysis toolkit, upgraded while
awaiting the results of the long-duration v17.2 simulation. It incorporates
two new, critical diagnostic tools requested by the user to perform the deepest
possible analysis of the turbulent breather state.

This is the final instrument for our scientific discovery.

Key Analytical Upgrades from v22:
1.  **The "Seismograph" (Temporal Power Spectrum):** A new analysis method,
    `_analyze_temporal_dynamics`, calculates the power spectrum of the total
    particle number fluctuations. This will reveal if the "breather" state has
    characteristic oscillation frequencies (a limit cycle) or is broadband
    chaotic.
2.  **The "Microscope" (Vortex Zoom):** The plotting dashboard is upgraded with
    an inset plot that automatically zooms in on the phase structure around a
    detected vortex. This provides a clear, qualitative check of its
    topological nature (the 2-pi phase winding).
3.  **Refined Dashboard Layout:** The final plot layout is further improved to
    a comprehensive 4x2 grid to cleanly present all available diagnostic data.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift, fftfreq
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
import os
import argparse

class UltimateAnalysisEngine:
    """
    The final, comprehensive object-oriented framework for deep analysis of
    the collective polariton state dataset.
    """
    DEFAULT_INPUT_FILE = 'dynamic_state_output_v17.2.npz'

    def __init__(self, input_file=None):
        """Initializes the analysis engine."""
        self.input_file = input_file if input_file else self.DEFAULT_INPUT_FILE
        self.data = None
        self.grid = None
        self.analysis_results = {}
        print(f"--- UltimateAnalysisEngine v23 Initialized for file: {self.input_file} ---")

    def load_data(self):
        """Loads and validates the simulation data from the NPZ file."""
        print(f"\n--- Loading dataset from {self.input_file} ---")
        if not os.path.exists(self.input_file):
            print(f"**FATAL ERROR**: Input file '{self.input_file}' not found.")
            return False
        try:
            self.data = np.load(self.input_file, allow_pickle=True)
            required_keys = ['final_psi', 'time_points', 'particle_number_history']
            for key in required_keys:
                if key not in self.data:
                    print(f"**ERROR**: Required data key '{key}' not found in the input file.")
                    return False
            sim_params = self.data['sim_params'].item()
            N, L = sim_params['N'], sim_params['L']; dx = L / N
            k_1d = 2 * np.pi * fftfreq(N, d=dx)
            Kx, Ky = np.meshgrid(k_1d, k_1d); K = np.sqrt(Kx**2 + Ky**2)
            self.grid = {'N': N, 'L': L, 'dx': dx, 'K': K}
            self.phys_params = self.data['phys_params'].item()
            self.sim_params = sim_params
            print("--- Data loaded successfully. ---")
            return True
        except Exception as e:
            print(f"**ERROR**: Failed to load file. Error: {e}")
            return False

    def perform_full_analysis(self):
        """Runs the complete suite of diagnostic analyses."""
        if self.data is None: return
        self._analyze_structure()
        self._analyze_topology()
        self._analyze_temporal_dynamics() # The "Seismograph"
        print("\n--- Full diagnostic analysis complete. ---")

    def _analyze_structure(self):
        """Analyzes static spatial structure (droplet positions, g(r), Psi_6)."""
        print("\n--- Analyzing Spatial Structure ---")
        psi_density = np.abs(self.data['final_psi'])**2
        N, L, dx = self.grid['N'], self.grid['L'], self.grid['dx']
        threshold = 0.4 * np.max(psi_density)
        min_dist_pixels = int(2.5 / dx)
        peaks, _ = find_peaks(psi_density.flatten(), height=threshold, distance=min_dist_pixels)
        if len(peaks) < 10:
            print(f"  Warning: Only {len(peaks)} droplets found. Statistical analysis may be noisy.")
            return
        peak_coords = np.array(np.unravel_index(peaks, (N, N))).T
        droplet_positions = (peak_coords - N/2) * dx
        print(f"  Found {len(droplet_positions)} droplets.")
        self.analysis_results['droplet_positions'] = droplet_positions
        self._calculate_g_r(droplet_positions, L)
        self._calculate_psi_6(droplet_positions)

    def _calculate_g_r(self, positions, L):
        """Calculates the pair correlation function."""
        num_particles = len(positions)
        area_density = num_particles / (L**2)
        tree = cKDTree(positions)
        distances = tree.query_pairs(r=L/2.1, output_type='ndarray')
        if distances.shape[0] == 0: return
        r_vals = np.linalg.norm(positions[distances[:,0]] - positions[distances[:,1]], axis=1)
        dr_hist = 0.4
        r_bins = np.arange(0, L/2.1, dr_hist)
        hist, bin_edges = np.histogram(r_vals, bins=r_bins)
        r_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        shell_areas = 2 * np.pi * r_mids * dr_hist
        ideal_gas_counts = shell_areas * area_density
        g_r_vals = hist / (ideal_gas_counts * num_particles / 2)
        self.analysis_results['g_r'] = (r_mids, g_r_vals)
        print("  Pair correlation function g(r) calculated.")

    def _calculate_psi_6(self, positions):
        """Calculates the global bond-orientational order parameter."""
        tree = cKDTree(positions)
        dist, idx = tree.query(positions, k=7)
        psi_6_local_values = []
        for i in range(len(positions)):
            neighbors_idx = idx[i, 1:]
            bonds = positions[neighbors_idx] - positions[i]
            angles = np.angle(bonds[:, 0] + 1j * bonds[:, 1])
            psi_6_i = np.mean(np.exp(6j * angles))
            psi_6_local_values.append(psi_6_i)
        psi_6_global = np.abs(np.mean(psi_6_local_values))
        self.analysis_results['Psi_6'] = psi_6_global
        print(f"  Calculated Global Hexagonal Order (Psi_6): {psi_6_global:.4f}")

    def _analyze_topology(self):
        """Finds and counts vortices and anti-vortices."""
        print("\n--- Analyzing Topology (Vortex Detection) ---")
        phase = np.angle(self.data['final_psi'])
        vortices, antivortices = [], []
        for i in range(self.grid['N'] - 1):
            for j in range(self.grid['N'] - 1):
                p1, p2, p3, p4 = phase[i, j], phase[i+1, j], phase[i+1, j+1], phase[i, j+1]
                d12 = np.angle(np.exp(1j * (p2 - p1))); d23 = np.angle(np.exp(1j * (p3 - p2)))
                d34 = np.angle(np.exp(1j * (p4 - p3))); d41 = np.angle(np.exp(1j * (p1 - p4)))
                winding_number = (d12 + d23 + d34 + d41) / (2 * np.pi)
                if np.abs(winding_number - 1) < 0.2: vortices.append((i, j))
                elif np.abs(winding_number + 1) < 0.2: antivortices.append((i, j))
        self.analysis_results['vortices'] = np.array(vortices)
        self.analysis_results['antivortices'] = np.array(antivortices)
        print(f"  Found {len(vortices)} vortices and {len(antivortices)} anti-vortices.")

    def _analyze_temporal_dynamics(self):
        """
        NEW in v23: The "Seismograph". Analyzes the power spectrum of particle
        number fluctuations to find characteristic oscillation frequencies.
        """
        print("\n--- Analyzing Temporal Dynamics (Power Spectrum) ---")
        # Use the latter half of the simulation for steady-state analysis
        steady_state_start_index = len(self.data['time_points']) // 2
        
        N_history = self.data['particle_number_history'][steady_state_start_index:]
        t_history = self.data['time_points'][steady_state_start_index:]
        
        if len(N_history) < 2:
            print("  Not enough data for temporal analysis.")
            return
            
        # Subtract the mean to focus on fluctuations
        N_fluctuations = N_history - np.mean(N_history)
        
        # Calculate power spectrum using FFT
        dt = t_history[1] - t_history[0]
        n_points = len(N_fluctuations)
        power_spectrum = np.abs(fft(N_fluctuations) / n_points)**2
        frequencies = fftfreq(n_points, d=dt)
        
        # We only care about positive frequencies
        positive_freq_mask = frequencies > 0
        self.analysis_results['power_spectrum'] = (frequencies[positive_freq_mask], power_spectrum[positive_freq_mask])
        print("  Power spectrum of particle number fluctuations calculated.")

    def plot_diagnostic_dashboard(self):
        """Plots the final, comprehensive diagnostic dashboard."""
        if self.data is None: return

        print("\n--- Generating Ultimate Diagnostic Dashboard (v23) ---")
        final_psi = self.data['final_psi']
        psi_density = np.abs(final_psi)**2
        phase = np.angle(final_psi)
        N, L, dx = self.grid['N'], self.grid['L'], self.grid['dx']
        
        fig = plt.figure(figsize=(22, 28))
        gs = fig.add_gridspec(4, 2)
        fig.suptitle('Ultimate Analysis of the Collective State', fontsize=24, weight='bold')

        # Row 1: Time Evolutions
        ax1a = fig.add_subplot(gs[0, 0]); 
        ax1a.plot(self.data['time_points'], self.data['particle_number_history'], 'navy')
        ax1a.set_title('Particle Number Evolution'); ax1a.grid(True); ax1a.set_xlabel('Time'); ax1a.set_ylabel('N'); ax1a.set_yscale('log')
        
        ax1b = fig.add_subplot(gs[0, 1])
        if 'power_spectrum' in self.analysis_results:
            freq, power = self.analysis_results['power_spectrum']
            ax1b.plot(freq, power, color='darkred')
            # Find and label the main peak
            if len(power) > 0:
                peak_idx = np.argmax(power)
                peak_freq = freq[peak_idx]
                ax1b.axvline(peak_freq, color='orange', linestyle='--', label=f'Peak at f={peak_freq:.3f}')
                ax1b.legend()
        ax1b.set_title('Power Spectrum of N(t) Fluctuations ("Seismograph")'); ax1b.grid(True); ax1b.set_xlabel('Frequency'); ax1b.set_ylabel('Power'); ax1b.set_yscale('log'); ax1b.set_xlim(left=0)

        # Row 2: Real Space Views
        ax2a = fig.add_subplot(gs[1, 0]); im2a = ax2a.imshow(psi_density, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='inferno')
        ax2a.set_title('Final State: Density $|\psi|^2$'); ax2a.set_xlabel('x'); ax2a.set_ylabel('y')
        fig.colorbar(im2a, ax=ax2a, fraction=0.046, pad=0.04)

        ax2b = fig.add_subplot(gs[1, 1]); im2b = ax2b.imshow(phase, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='hsv')
        if 'vortices' in self.analysis_results and len(self.analysis_results['vortices']) > 0:
            v_coords = (self.analysis_results['vortices'] - N/2) * dx
            ax2b.plot(v_coords[:, 1], v_coords[:, 0], 'o', c='white', ms=5, mfc='none', mew=1.5, label='Vortices')
        ax2b.set_title('Final State: Phase & Vortices'); ax2b.set_xlabel('x'); ax2b.set_ylabel('y');
        if 'vortices' in self.analysis_results and len(self.analysis_results['vortices']) > 0: ax2b.legend()
        fig.colorbar(im2b, ax=ax2b, fraction=0.046, pad=0.04)
        
        # --- NEW in v23: The "Microscope" ---
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_inset = inset_axes(ax2b, width="40%", height="40%", loc='upper right')
        if 'vortices' in self.analysis_results and len(self.analysis_results['vortices']) > 0:
            vortex_center = self.analysis_results['vortices'][0].astype(int)
            zoom_size = 10 # 20x20 pixel window
            y_slice = slice(max(0, vortex_center[0]-zoom_size), min(N, vortex_center[0]+zoom_size))
            x_slice = slice(max(0, vortex_center[1]-zoom_size), min(N, vortex_center[1]+zoom_size))
            phase_zoom = phase[y_slice, x_slice]
            ax_inset.imshow(phase_zoom, origin='lower', cmap='hsv', extent=[-zoom_size*dx, zoom_size*dx, -zoom_size*dx, zoom_size*dx])
            ax_inset.plot(0, 0, '+', color='white', markersize=10, mew=2)
            ax_inset.set_title("Vortex Core Zoom")
            ax_inset.set_xticks([]); ax_inset.set_yticks([])

        # Row 3: Structural Analysis
        ax3a = fig.add_subplot(gs[2, 0])
        spectrum = fftshift(np.abs(fft2(psi_density))**2)
        k_lim = np.pi / dx
        ax3a.imshow(np.log10(spectrum), extent=[-k_lim,k_lim,-k_lim,k_lim], origin='lower', cmap='magma')
        ax3a.set_title('Spatial Fourier Spectrum'); ax3a.set_xlabel('$k_x$'); ax3a.set_ylabel('$k_y$')
        
        ax3b = fig.add_subplot(gs[2, 1])
        if 'g_r' in self.analysis_results and self.analysis_results.get('g_r') is not None:
            r, g = self.analysis_results['g_r']
            ax3b.plot(r, g, 'o-', color='purple'); ax3b.axhline(1, color='k', ls='--')
        ax3b.set_title('Pair Correlation Function g(r)'); ax3b.set_xlabel('Radius (r)'); ax3b.set_ylabel('g(r)')
        ax3b.grid(True, ls=':'); ax3b.set_ylim(bottom=0)

        # Row 4: Summary Text
        ax4a = fig.add_subplot(gs[3, :]); ax4a.axis('off')
        summary_text = "Quantitative Summary:\n\n"
        if 'droplet_positions' in self.analysis_results and self.analysis_results['droplet_positions'] is not None:
            summary_text += f"  - Droplets Found: {len(self.analysis_results['droplet_positions'])}\n"
        if 'Psi_6' in self.analysis_results:
            summary_text += f"  - Hexagonal Order (Psi_6): {self.analysis_results['Psi_6']:.4f}\n"
        if 'vortices' in self.analysis_results:
            summary_text += f"  - Vortices / Anti-vortices: {len(self.analysis_results['vortices'])} / {len(self.analysis_results['antivortices'])}\n"
        if 'power_spectrum' in self.analysis_results:
             freq, power = self.analysis_results['power_spectrum']
             if len(power)>0:
                 peak_freq = freq[np.argmax(power)]
                 summary_text += f"  - Main Oscillation Freq: {peak_freq:.4f}"
        ax4a.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=18, family='monospace', bbox=dict(boxstyle="round,pad=1", fc="lightgray", alpha=0.5))

        fig.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Analysis Engine for collective polariton states.")
    parser.add_argument('input_file', nargs='?', default='dynamic_state_output_v17.2.npz', 
                        help="Path to the simulation output NPZ file.")
    
    import sys
    if 'ipykernel' in sys.modules or 'google.colab' in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    engine = AdvancedAnalysisEngine(input_file=args.input_file)
    if engine.load_data():
        engine.perform_full_analysis()
        engine.plot_diagnostic_dashboard()
