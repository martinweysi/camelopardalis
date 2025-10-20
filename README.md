Camelopardalis: A Numerical Laboratory for Non-Local Polariton Dynamics

Author: Martin Weysi (martinweysi@gmail.com)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete history of Python simulation and analysis codes for the research leading to the paper: "Discovery of a Turbulent Correlated Liquid in a Non-Local Driven-Dissipative Quantum Fluid."

Abstract of the Discovery

We report the discovery of a novel non-equilibrium state of matter in a driven-dissipative quantum fluid, described by a non-local Gross-Pitaevskii equation relevant to exciton-polariton condensates. Our numerical investigation reveals that the interplay between strong local and long-range repulsion prevents the system from forming a simple ordered state. Instead, it spontaneously self-organizes into a statistically steady state best described as a turbulent correlated liquid. This "quantum chimera" phase is characterized by a unique combination of contradictory features: it is highly dynamic and filled with topological defects, yet possesses strong short-range positional and orientational order, while lacking any long-range order. This state represents a new paradigm for understanding many-body phenomena in strongly interacting, non-equilibrium systems.

The Scientific Journey: From Engineered Droplets to a Discovered "Dragon"

This project did not begin with the goal of finding a turbulent liquid. It began with a far simpler, more "engineered" objective that ultimately led to a profound scientific failure, and from that failure, to a genuine discovery.

Phase 1: The Failure of Local Models (The v1 to v15.x Graveyard)

Our initial hypothesis was that a stable, single, localized polariton condensate (a "droplet") could be engineered in the experimentally relevant regime using a sophisticated local potential with competing nonlinearities (V(\psi) \propto \lambda|\psi|^4 - |\eta||\psi|^6 + \xi|\psi|^8).

Result: Catastrophic failure. When calibrated with realistic parameters (strong interactions, low loss), this entire class of models proved to be fundamentally unstable. The simulations consistently collapsed into either:

Modulational Instability: A violent fragmentation of the condensate into a turbulent "sandstorm."

Runaway Growth: An uncontrolled, explosive growth of the particle number, leading to numerical blow-up.

This crucial null result demonstrated that no amount of fine-tuning of a local potential can stabilize a condensate in this physically important regime.

Phase 2: The Paradigm Shift to Non-Local Physics (v16.x and v17.x series)

The failure of local models forced a paradigm shift. We hypothesized that the missing ingredient was non-local interactions, a physical mechanism representing the long-range repulsion mediated by the photonic component of polaritons. We implemented this by coupling the GPE to a screened Poisson equation.

Our initial goal was still to engineer a clean, single droplet. However, the first successful simulation that did not collapse or explode (v16.2) revealed something unexpected: a stable, but chaotic and pulsating, many-body state. We initially mistook this for another failure.

After numerous failed attempts to "tame" the model into producing a single droplet (v16.1, v17.0), we correctly identified that this chaotic, dynamic state was not a bug, but the true physical solution. We named this state the "Turbulent Breather" or the "Quantum Chimera."

Phase 3: The Dragon Hunt & Autopsy (v17.2 + v22)

The final phase of the project was to accept this new reality and characterize it. The simulation dynamic_state_capture_v17.2.py was designed to capture a long, high-quality "specimen" of this state. The analysis toolkit deep_analysis_engine_v22.1.py was then used to perform a full "autopsy," leading to its final identification as a Turbulent Correlated Liquid.

Repository Structure

This repository is organized as a historical record of our scientific journey.

.
├── simulators/             # All time-evolution simulation codes
│   ├── local_models/       # The v15.x series: The graveyard of failed local models
│   └── nonlocal_models/    # The v16.x and v17.x series: The path to discovery
│       ├── ...
│       └── dynamic_state_capture_v17.2.py  (The Definitive Simulation)
│
├── analysis_tools/         # Standalone scripts for data analysis and plotting
│   ├── ...
│   └── deep_analysis_engine_v22.1.py       (The Definitive Analysis Toolkit)
│
├── manuscript/             # LaTeX source for the final paper
│   ├── manuscript_v10.tex
│   └── references.bib
│
└── README.md


How to Reproduce the Main Scientific Result

To reproduce the discovery of the "Turbulent Correlated Liquid," you need to perform the final two-step workflow.

Step 1: The Dragon Hunt (Simulation)

Run the definitive simulation script. Warning: This is a computationally intensive simulation and may take several hours to complete.

python simulators/nonlocal_models/dynamic_state_capture_v17.2.py


This will generate a file named dynamic_state_output_v17.2.npz, which contains the complete raw data of the simulated state.

Step 2: The Autopsy (Analysis)

Run the final analysis engine on the output file from Step 1. This script will load the data, perform all structural, topological, and dynamic analyses, and generate the final diagnostic dashboard.

python analysis_tools/deep_analysis_engine_v22.1.py


The script will automatically find dynamic_state_output_v17.2.npz and produce the multi-panel figure that constitutes the central evidence of our discovery.

Citation

Our manuscript is currently in preparation for submission to a peer-reviewed journal. In the meantime, if you use this code or our results, please link back to this repository.

Stay tuned for updates!
