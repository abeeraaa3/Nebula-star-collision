# 🌌 Nebula Star Collision Simulation

[![Taichi](https://img.shields.io/badge/Physics-Taichi-blueviolet.svg?style=flat-square)](https://github.com/taichi-dev/taichi)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg?style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

A cinematic, real-time astrophysics simulation of a binary neutron star merger (**Kilonova event**) built with Python and the high-performance **Taichi** programming language. This project features GPU-accelerated N-body physics, gravitational wave emission, relativistic jets, and dynamic r-process nucleosynthesis visualization.

---

## ✨ Features

### 🌟 Simulation Phases
The simulation dynamically transitions through four distinct astrophysically motivated phases:
1. **Phase 0 — Inspiral**: Two neutron stars orbit their common center of mass, gradually losing orbital energy and angular momentum through the emission of gravitational waves (approximated via the Peters 1964 formulation). The orbits tighten, and the orbital frequency chirps upward.
2. **Phase 1 — Collision**: The stars merge at relativistic speeds ($\sim 0.3c$). The collision triggers a massive release of energy, producing a brilliant thermal flash, launching relativistic jets, and ejecting high-velocity matter.
3. **Phase 2 — Kilonova**: An expanding fireball where rapid neutron-capture ($r$-process) nucleosynthesis occurs, synthesizing heavy elements like gold, platinum, and uranium. The ejecta cools dynamically, transitioning from blue-white to red-gold.
4. **Phase 3 — Aftermath**: The central remnant (either a hypermassive neutron star or a stellar-mass black hole, depending on the combined mass and the Tolman-Oppenheimer-Volkoff limit) settles, surrounded by a fallback accretion disk.

### 🔬 Physics & Mathematics Model
- **Peters Formula Gravitational Decay**: Realistic simulation of orbital decay due to gravitational radiation.
- **Softened N-Body Gravity**: Inter-particle and star-particle interactions computed using a Plummer-softened gravitational potential ($\epsilon$) to prevent singularities at close proximity.
- **Semi-Implicit Euler Integration**: Ensures orbital stability during the high-velocity inspiral.
- **Relativistic Jets**: Bipolar outflow matching approximations of the Blandford-Znajek mechanism.
- **Lanthanide-Rich/Poor Ejecta**: Model splits ejecta into high-opacity (red) and low-opacity (blue) components, simulating actual kilonova light curves.

### 🎨 Rendering & Visual Effects
- **Additive Gaussian Glow Splatting**: Particles render with soft, overlapping light profiles to create gas and plasma aesthetics.
- **Image Persistence Trails**: Pixel buffer decay creates elegant motion blur, emphasizing high-velocity orbits and outflows.
- **Gravitational Wave Visualizer**: Concentric propagation waves ripple outward from the binary orbit, scaling in frequency and amplitude as merger approaches.
- **Dynamic Color Modes**: Real-time color-mapping modes including Temperature, Velocity, Particle Types, and Synthesized Elements.

---

## 🎮 Controls

Use the on-screen GUI panel to adjust physics parameters in real time, or use the keyboard and mouse shortcuts below:

| Key / Action | Description |
|:---|:---|
| `SPACE` | Pause / Resume simulation |
| `R` | Reset simulation to initial state |
| `Q` | Quit simulation |
| `C` | Cycle particle color mode (Temperature, Velocity, Element, etc.) |
| `Left-Click` | Spawn new particles in the interstellar medium |
| `Drag (Hold)` | Apply custom gravitational forces to influence the stars / ejecta |

---

## 🛠️ Installation & Usage

### Prerequisites
Make sure you have Python 3.7+ installed. 

### Step 1: Install Dependencies
This project relies on **Taichi** for GPU/CPU parallelized computation and rendering:
```bash
pip install taichi
```

### Step 2: Run the Simulation
Run the main script using python:
```bash
python neutronstarcollision.py
```
*Note: The simulation automatically attempts to run on the GPU. If no compatible GPU/drivers are found, it will gracefully fall back to CPU execution.*

---

## 📚 Astrophysical Context

### Gravitational Waves & Peters Formula
The rate of change of the semi-major axis $a$ of a binary system due to gravitational wave radiation is described by the Peters formula (1964):
$$\frac{da}{dt} = -\frac{64}{5}\frac{G^3 m_1 m_2 (m_1 + m_2)}{c^5 a^3 (1 - e^2)^{7/2}} \left(1 + \frac{73}{24}e^2 + \frac{37}{96}e^4\right)$$
For circular orbits ($e=0$), this drives the rapid, accelerating inspiral demonstrated in the simulation.

### The r-Process & Heavy Elements
In the intense neutron flux of the merger, iron-peak seed nuclei rapidly capture free neutrons faster than they can beta-decay. This is the **$r$-process** (rapid neutron-capture process), responsible for creating about half of the elements heavier than iron, including gold ($\text{Au}$), platinum ($\text{Pt}$), and uranium ($\text{U}$).

---

## ⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.
