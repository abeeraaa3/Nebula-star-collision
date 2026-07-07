"""
================================================================================
 ★  NEUTRON STAR COLLISION SIMULATION  ★
================================================================================
A cinematic real-time physics simulation of a binary neutron star merger
(kilonova event), built with Python + Taichi.

THREE PHASES:
  Phase 0 - INSPIRAL:   Two neutron stars orbit each other, losing energy via
                         gravitational wave emission (Peters 1964 formula).
                         Orbits tighten, frequency chirps upward.
  Phase 1 - COLLISION:   Stars merge at ~0.3c. Energy release produces a
                         brilliant flash, relativistic jets, and kilonova ejecta.
  Phase 2 - KILONOVA:    Expanding fireball where r-process nucleosynthesis
                         creates heavy elements (gold, platinum, uranium).
                         Ejecta cools from blue-white to red-gold.
  Phase 3 - AFTERMATH:   Central remnant (hypermassive NS or black hole) with
                         fallback accretion disk. System slowly relaxes.

PHYSICS MODEL:
  - Keplerian two-body orbits with Peters formula gravitational wave decay
  - Softened N-body gravity (particles + stars), Plummer softening
  - Semi-implicit Euler integration for orbital stability
  - Tidal heating near Roche lobe boundaries
  - Velocity-stratified kilonova ejecta (power-law distribution)
  - Bipolar relativistic jets (Blandford-Znajek mechanism approximation)
  - Two-component kilonova shockwave (blue + red components)

RENDERING:
  - Additive Gaussian glow splatting per particle
  - Image persistence trails (pixel buffer decay)
  - Gravitational wave ripple visualization (concentric rings)
  - Expanding shockwave ring with color-stratified components
  - Temperature/velocity/age/field color mapping

CONTROLS:
  On-screen sliders (left panel) for real-time parameter adjustment
  SPACE: Pause/Resume | R: Reset | Q: Quit | C: Cycle color mode
  LEFT-CLICK: Spawn particles | DRAG (hold): Apply gravitational force

Author: Antigravity AI  |  Framework: Taichi (ti.GUI)
================================================================================
"""

# pyrefly: ignore [missing-import]
import taichi as ti
import math
import time

# =============================================================================
# TAICHI INITIALIZATION (GPU preferred, CPU fallback for integrated graphics)
# =============================================================================
try:
    ti.init(arch=ti.gpu, offline_cache=False)
    print("[OK] Running on GPU")
except Exception:
    ti.init(arch=ti.cpu, offline_cache=False)
    print("[!!] Running on CPU (integrated graphics). Expected on this hardware.")

# =============================================================================
# CONSTANTS — Every value is named, commented, and physically motivated.
# =============================================================================

PI = 3.14159265358979

# --- Display ---
WIDTH  = 1280                   # Window width  [pixels]
HEIGHT = 800                    # Window height [pixels]
ASPECT = float(WIDTH) / float(HEIGHT)  # Aspect ratio (1.6)

# --- Gravitational Physics ---
G = 3.0e-4                     # Gravitational constant [sim units]
                                # Real: 6.674e-11 N*m^2/kg^2.
                                # Scaled so visual orbital period at separation=0.22
                                # is ~2-3 seconds (aesthetically pleasing).
SOFTENING = 4.0e-3              # Plummer gravitational softening epsilon [sim length]
                                # Prevents 1/r^2 divergence at r->0.
                                # Chosen ~1/3 of visual star radius for smooth dynamics.

# --- Neutron Star Properties ---
NS_MASS_1 = 1.0                # Mass of star 1 [sim mass units]
                                # Represents ~1.4 M_sun (typical NS, near Chandrasekhar limit)
NS_MASS_2 = 1.0                # Mass of star 2 [sim mass units]
                                # Equal-mass merger (most common observed; GW170817 was ~1.36+1.17)
NS_VIS_RADIUS = 0.010          # Visual display radius [normalized screen coords]
                                # Real NS radius: ~10 km. Exaggerated ~1000x for visibility.
                                # (Real NS is a pinpoint at any reasonable orbital scale.)

# --- Orbital Mechanics ---
INIT_SEPARATION = 0.22         # Initial orbital separation [normalized screen coords]
                                # Tuned for ~5-8 second inspiral at 1x time speed.
GW_DECAY_COEFF = 1.8e-5        # Increased for faster visual evolution
                                # SIMPLIFIED Peters (1964) formula:  da/dt = -C / a^3
MERGE_DISTANCE = 0.024         # Merger trigger distance [normalized coords]
DT_BASE = 2.0e-3               # Base simulation timestep [sim time units]

# --- Remnant Constants ---
TOV_LIMIT = 2.15               # Tolman-Oppenheimer-Volkoff mass limit (sim units)
                                # Below this -> Magnetar. Above this -> Black Hole.
EH_RADIUS = 0.015              # Event horizon physical radius for Black Hole.

# --- Explosion / Kilonova ---
EXPLODE_SPEED = 0.10           # Maximum ejecta radial velocity [sim velocity]
                                # Real: ~0.1-0.3c. Scaled so ejecta crosses screen in ~6 sec.
PEAK_TEMP = 1.0                # Normalized peak temperature at collision [dimensionless]
                                # Maps to ~10^10 K in reality. Normalized to [0,1] for color.
COOL_RATE = 0.0006             # Radiative cooling rate per timestep [dimensionless]
                                # VIBECODED: Tuned for ~15s blue-to-red color transition.
                                # Real cooling: complex interplay of neutrino emission,
                                # r-process radioactive decay heating, and photon diffusion.
JET_FRACTION = 0.08            # Fraction of particles that form relativistic jets
                                # Real jets: powered by Blandford-Znajek mechanism
                                # (magnetic field + spinning BH). ~few% of ejecta mass.
JET_SPEED_MULT = 2.5           # Jet speed multiplier vs spherical ejecta
                                # Real jets: Lorentz factor ~10-100. We approximate
                                # with 2.5x spherical speed for visual effect.
JET_HALF_ANGLE = 0.35          # Jet opening half-angle [radians, ~20 degrees]
                                # Real: ~5-15 deg. Widened slightly for visibility.

# --- Shockwave ---
SHOCK_SPEED = 0.0025           # Primary shockwave expansion rate [norm coords/frame]
SHOCK_THICK = 0.005            # Visual thickness of shockwave ring [norm coords]
SHOCK2_RATIO = 0.55            # Secondary (red) shockwave radius ratio to primary
                                # Represents slower lanthanide-rich "red kilonova" ejecta.

# --- Gravitational Wave Visualization ---
GW_BASE_WAVELEN = 60.0         # Base visual wavelength of GW ripples [sim units]
GW_ANIM_SPEED = 5.0            # Animation speed of GW ripple propagation
GW_MAX_ALPHA = 0.055           # Maximum brightness of GW ripple overlay

# --- Particle System ---
MAX_P     = 15000               # Maximum particles (preallocated field size)
DEFAULT_N = 10000               # Default active particle count
MIN_N     = 5000                # Minimum (slider lower bound)
MAX_N_SL  = 15000               # Maximum (slider upper bound)

# --- Damping ---
DAMP_DEFAULT = 0.9985          # Velocity damping per frame [dimensionless]
                                # 0.9985 = 0.15% energy loss/step.
                                # Approximates viscous/radiative dissipation.

# --- Rendering ---
TRAIL_DECAY_DEF = 0.935        # Image persistence per frame [dimensionless]
                                # 0.935 → ~15 frame visible trail.
GLOW_R_BASE = 2                # Base glow splat radius per particle [pixels]
STAR_GLOW_R = 28               # Glow splat radius per star [pixels]
BG = ti.Vector([0.003, 0.003, 0.014])  # Background color (near-black, slight blue tint)
P_BRIGHT = 0.30                # Particle glow base brightness multiplier
S_BRIGHT = 2.2                 # Star glow base brightness multiplier

# =============================================================================
# TAICHI FIELDS — All per-particle state, accessed from @ti.kernel functions.
# =============================================================================

pos     = ti.Vector.field(2, dtype=ti.f32, shape=MAX_P)   # Position [norm 0-1]
vel     = ti.Vector.field(2, dtype=ti.f32, shape=MAX_P)   # Velocity [sim units]
temp    = ti.field(dtype=ti.f32, shape=MAX_P)              # Temperature [0-1]
age     = ti.field(dtype=ti.f32, shape=MAX_P)              # Age [sim time]
alive   = ti.field(dtype=ti.i32, shape=MAX_P)              # 1=active, 0=dead
ptype   = ti.field(dtype=ti.i32, shape=MAX_P)              # 0=disk1, 1=disk2, 2=ejecta,
                                                            # 3=ambient, 4=jet
element = ti.field(dtype=ti.i32, shape=MAX_P)              # 0=Fe/disk, 1=Pt, 2=Au, 3=U

# Pixel buffer: additive rendering with image persistence for trails
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# =============================================================================
# COLOR MAPPING — @ti.func (called from kernels only)
# =============================================================================

@ti.func
def color_by_temperature(t: ti.f32) -> ti.Vector:
    """
    Temperature → color for neutron star merger.
    Physically motivated by observed kilonova color evolution (AT2017gfo):
      Cold blue → warm cyan → hot white → cooling gold → red (r-process glow)
    """
    tc = ti.max(ti.min(t, 1.0), 0.0)
    r, g, b = 0.0, 0.0, 0.0

    if tc < 0.10:
        # Very cold: deep indigo (ambient ISM, ~10^4 K)
        f = tc / 0.10
        r = 0.02 + f * 0.06
        g = 0.02 + f * 0.10
        b = 0.12 + f * 0.48
    elif tc < 0.25:
        # Cool: blue-cyan (NS accretion disk, ~10^6-10^7 K)
        f = (tc - 0.10) / 0.15
        r = 0.08 + f * 0.17
        g = 0.12 + f * 0.48
        b = 0.60 + f * 0.30
    elif tc < 0.45:
        # Warm: cyan-white (compressed matter, ~10^8 K)
        f = (tc - 0.25) / 0.20
        r = 0.25 + f * 0.60
        g = 0.60 + f * 0.35
        b = 0.90 + f * 0.10
    elif tc < 0.65:
        # Hot: white → golden (GOLD/PLATINUM SYNTHESIS ZONE)
        # This is where r-process nucleosynthesis forges heavy elements
        f = (tc - 0.45) / 0.20
        r = 0.85 + f * 0.15
        g = 0.95 - f * 0.10
        b = 1.00 - f * 0.50
    elif tc < 0.82:
        # Very hot: golden-white ("blue kilonova" lanthanide-free ejecta)
        f = (tc - 0.65) / 0.17
        r = 1.00
        g = 0.85 + f * 0.10
        b = 0.50 + f * 0.40
    else:
        # Extreme: brilliant blue-white (NS surface, ~10^10 K)
        f = (tc - 0.82) / 0.18
        r = 1.00
        g = 0.95 + f * 0.05
        b = 0.90 + f * 0.10
    return ti.Vector([r, g, b])


@ti.func
def color_by_velocity(vx: ti.f32, vy: ti.f32) -> ti.Vector:
    """Velocity magnitude: blue (slow) → cyan → yellow → red (fast)."""
    spd = ti.sqrt(vx * vx + vy * vy + 1e-10)
    s = ti.min(spd / 0.08, 1.0)
    r, g, b = 0.0, 0.0, 0.0
    if s < 0.25:
        f = s / 0.25
        r = 0.04;  g = 0.04 + f * 0.20;  b = 0.20 + f * 0.60
    elif s < 0.50:
        f = (s - 0.25) / 0.25
        r = 0.04 + f * 0.26;  g = 0.24 + f * 0.56;  b = 0.80 - f * 0.20
    elif s < 0.75:
        f = (s - 0.50) / 0.25
        r = 0.30 + f * 0.60;  g = 0.80 + f * 0.20;  b = 0.60 - f * 0.55
    else:
        f = (s - 0.75) / 0.25
        r = 0.90 + f * 0.10;  g = 1.00 - f * 0.45;  b = 0.05
    return ti.Vector([r, g, b])


@ti.func
def color_by_age(a: ti.f32) -> ti.Vector:
    """Age: young = bright blue-white, old = fading amber-red."""
    t = ti.min(a / 600.0, 1.0)
    r = 0.30 + t * 0.70
    g = 0.55 + t * 0.25 - t * t * 0.65
    b = 0.90 - t * 0.85
    return ti.Vector([r, g, b])


@ti.func
def color_by_field(px: ti.f32, py: ti.f32) -> ti.Vector:
    """Distance from center → field strength proxy. Close=bright, far=dim."""
    dx = px - 0.5
    dy = py - 0.5
    d = ti.min(ti.sqrt(dx * dx + dy * dy + 1e-8) / 0.40, 1.0)
    r = 1.00 - d * 0.70
    g = 0.90 - d * 0.70
    b = 0.50 + d * 0.30
    return ti.Vector([r, g, b])


@ti.func
def color_by_element(i: ti.i32) -> ti.Vector:
    """Color particles by synthesized r-process elements: Platinum (Pt), Gold (Au), Uranium (U)."""
    col = ti.Vector([0.5, 0.5, 0.5])
    el = element[i]
    if el == 0:
        # Fe/light elements (accretion disk, ISM): glowing blue-cyan
        col = ti.Vector([0.1, 0.45, 0.75])
    elif el == 1:
        # Platinum (Pt): Silver/White-blue
        col = ti.Vector([0.85, 0.90, 0.98])
    elif el == 2:
        # Gold (Au): Rich warm gold
        col = ti.Vector([1.0, 0.78, 0.05])
    elif el == 3:
        # Uranium (U): Radioactive bright lime-green
        col = ti.Vector([0.22, 0.95, 0.40])
    return col


@ti.func
def get_particle_color(i: ti.i32, mode: ti.i32) -> ti.Vector:
    """Dispatch to the active color mapping mode."""
    col = ti.Vector([1.0, 1.0, 1.0])
    if mode == 0:
        col = color_by_temperature(temp[i])
    elif mode == 1:
        col = color_by_velocity(vel[i][0], vel[i][1])
    elif mode == 2:
        col = color_by_age(age[i])
    elif mode == 3:
        col = color_by_field(pos[i][0], pos[i][1])
    else:
        col = color_by_element(i)
    return col


# =============================================================================
# PHYSICS KERNELS
# =============================================================================

@ti.kernel
def init_particles(n: ti.i32,
                   s1x: ti.f32, s1y: ti.f32, s2x: ti.f32, s2y: ti.f32,
                   s1vx: ti.f32, s1vy: ti.f32, s2vx: ti.f32, s2vy: ti.f32,
                   m1: ti.f32, m2: ti.f32):
    """
    Distribute particles into accretion disks around each star,
    tidal streams at the L1 Lagrange point, and ambient halo.
    Each disk particle gets Keplerian orbital velocity + parent star velocity.
    """
    for i in range(n):
        alive[i] = 1
        age[i] = ti.random() * 60.0   # Stagger ages for visual variety
        element[i] = 0

        r = ti.random()
        ang = ti.random() * 2.0 * PI
        px, py = 0.0, 0.0
        vx, vy = 0.0, 0.0

        if r < 0.37:
            # --- Accretion disk around star 1 ---
            dr = 0.006 + ti.random() * 0.038
            px = s1x + dr * ti.cos(ang)
            py = s1y + dr * ti.sin(ang)
            v_orb = ti.sqrt(G * m1 / (dr + SOFTENING))
            vx = -v_orb * ti.sin(ang) + s1vx + (ti.random() - 0.5) * v_orb * 0.12
            vy =  v_orb * ti.cos(ang) + s1vy + (ti.random() - 0.5) * v_orb * 0.12
            temp[i] = 0.12 + ti.random() * 0.22
            ptype[i] = 0

        elif r < 0.74:
            # --- Accretion disk around star 2 ---
            dr = 0.006 + ti.random() * 0.038
            px = s2x + dr * ti.cos(ang)
            py = s2y + dr * ti.sin(ang)
            v_orb = ti.sqrt(G * m2 / (dr + SOFTENING))
            vx = -v_orb * ti.sin(ang) + s2vx + (ti.random() - 0.5) * v_orb * 0.12
            vy =  v_orb * ti.cos(ang) + s2vy + (ti.random() - 0.5) * v_orb * 0.12
            temp[i] = 0.12 + ti.random() * 0.22
            ptype[i] = 1

        elif r < 0.87:
            # --- Tidal stream (L1 region between stars) ---
            t_frac = ti.random()
            px = s1x + (s2x - s1x) * t_frac + (ti.random() - 0.5) * 0.018
            py = s1y + (s2y - s1y) * t_frac + (ti.random() - 0.5) * 0.018
            vx = s1vx + (s2vx - s1vx) * t_frac + (ti.random() - 0.5) * 0.004
            vy = s1vy + (s2vy - s1vy) * t_frac + (ti.random() - 0.5) * 0.004
            temp[i] = 0.22 + ti.random() * 0.15   # Tidally heated
            ptype[i] = 3
        else:
            # --- Ambient halo ---
            px = 0.5 + (ti.random() - 0.5) * 0.50
            py = 0.5 + (ti.random() - 0.5) * 0.35
            vx = (ti.random() - 0.5) * 0.0015
            vy = (ti.random() - 0.5) * 0.0015
            temp[i] = 0.03 + ti.random() * 0.05
            ptype[i] = 3

        pos[i] = ti.Vector([px, py])
        vel[i] = ti.Vector([vx, vy])

    # Deactivate unused capacity
    for i in range(n, MAX_P):
        alive[i] = 0


@ti.kernel
def step_binary(n: ti.i32,
                s1x: ti.f32, s1y: ti.f32, s2x: ti.f32, s2y: ti.f32,
                m1: ti.f32, m2: ti.f32, dt: ti.f32, damp: ti.f32,
                g_mult: ti.f32, mx: ti.f32, my: ti.f32, m_act: ti.i32):
    """
    Physics update during inspiral: each particle feels gravity from TWO stars.
    Semi-implicit Euler integration (velocity first, then position) for stability.
    Includes tidal heating for particles near either star.
    """
    for i in range(n):
        if alive[i] == 0:
            continue
        p = pos[i]
        v = vel[i]

        # -- Gravity from star 1 --
        d1x = s1x - p[0]
        d1y = s1y - p[1]
        r1s = d1x * d1x + d1y * d1y + SOFTENING * SOFTENING
        r1  = ti.sqrt(r1s)
        r1c = r1s * r1                              # r^3 for inverse-square
        a1x = G * m1 * g_mult * d1x / (r1c + 1e-12)
        a1y = G * m1 * g_mult * d1y / (r1c + 1e-12)

        # -- Gravity from star 2 --
        d2x = s2x - p[0]
        d2y = s2y - p[1]
        r2s = d2x * d2x + d2y * d2y + SOFTENING * SOFTENING
        r2  = ti.sqrt(r2s)
        r2c = r2s * r2
        a2x = G * m2 * g_mult * d2x / (r2c + 1e-12)
        a2y = G * m2 * g_mult * d2y / (r2c + 1e-12)

        ax = a1x + a2x
        ay = a1y + a2y

        # -- Mouse force (drag interaction) --
        if m_act == 1:
            dmx = mx - p[0]
            dmy = my - p[1]
            rms = dmx * dmx + dmy * dmy + 1e-5
            rm  = ti.sqrt(rms)
            if rm < 0.09:
                fsc = 0.0025 / (rms + 1e-5)
                ax += dmx * fsc
                ay += dmy * fsc

        # -- Tidal heating (particles near stars gain temperature) --
        rmin = ti.min(r1, r2)
        if rmin < 0.045:
            heat = 0.0025 * (0.045 - rmin) / 0.045
            temp[i] = ti.min(temp[i] + heat * dt * 120.0, 0.55)

        # -- Semi-implicit Euler --
        v[0] += ax * dt
        v[1] += ay * dt
        v   *= damp
        p[0] += v[0] * dt
        p[1] += v[1] * dt

        vel[i] = v
        pos[i] = p
        age[i] += dt
        temp[i] = ti.max(temp[i] - COOL_RATE * 0.25 * dt * 120.0, 0.02)

        # -- Respawn out-of-bounds particles near a random star --
        if p[0] < -0.12 or p[0] > 1.12 or p[1] < -0.12 or p[1] > 1.12:
            ang2 = ti.random() * 2.0 * PI
            rr = 0.008 + ti.random() * 0.032
            if ti.random() < 0.5:
                pos[i] = ti.Vector([s1x + rr * ti.cos(ang2),
                                    s1y + rr * ti.sin(ang2)])
            else:
                pos[i] = ti.Vector([s2x + rr * ti.cos(ang2),
                                    s2y + rr * ti.sin(ang2)])
            vel[i] = ti.Vector([(ti.random()-0.5)*0.004, (ti.random()-0.5)*0.004])
            temp[i] = 0.10 + ti.random() * 0.12
            age[i]  = 0.0


@ti.kernel
def step_merged(n: ti.i32, cx: ti.f32, cy: ti.f32, mt: ti.f32,
                dt: ti.f32, damp: ti.f32, g_mult: ti.f32,
                mx: ti.f32, my: ti.f32, m_act: ti.i32,
                is_black_hole: ti.i32):
    """
    Physics update post-merger: single central remnant gravity.
    Remnant may be a rapidly spinning Magnetar or a collapsed Black Hole.
    Black Hole absorbs particles crossing the Event Horizon and recycles them.
    """
    for i in range(n):
        if alive[i] == 0:
            continue
        p = pos[i]
        v = vel[i]

        dx = cx - p[0]
        dy = cy - p[1]
        rs = dx * dx + dy * dy + SOFTENING * SOFTENING
        r  = ti.sqrt(rs)
        rc = rs * r
        ax = G * mt * g_mult * dx / (rc + 1e-12)
        ay = G * mt * g_mult * dy / (rc + 1e-12)

        if m_act == 1:
            dmx = mx - p[0]
            dmy = my - p[1]
            rms = dmx * dmx + dmy * dmy + 1e-5
            rm  = ti.sqrt(rms)
            if rm < 0.09:
                fsc = 0.0025 / (rms + 1e-5)
                ax += dmx * fsc
                ay += dmy * fsc

        v[0] += ax * dt
        v[1] += ay * dt
        v   *= damp
        p[0] += v[0] * dt
        p[1] += v[1] * dt

        vel[i] = v
        pos[i] = p
        age[i] += dt
        temp[i] = ti.max(temp[i] - COOL_RATE * dt * 120.0, 0.02)

        # Swallowed or out of bounds check
        dist_rem = ti.sqrt(dx * dx + dy * dy + 1e-9)
        off_screen = p[0] < -0.12 or p[0] > 1.12 or p[1] < -0.12 or p[1] > 1.12
        swallowed = is_black_hole == 1 and dist_rem < EH_RADIUS

        if off_screen or swallowed:
            ang2 = ti.random() * 2.0 * PI
            # If swallowed by BH, recycle to the hot accretion disk. If off-screen, spawn near central remnant.
            rr = 0.045 + ti.random() * 0.12 if swallowed else 0.015 + ti.random() * 0.055
            v_o = ti.sqrt(G * mt * g_mult / (rr + SOFTENING)) * (0.85 if swallowed else 0.65)
            pos[i] = ti.Vector([cx + rr * ti.cos(ang2), cy + rr * ti.sin(ang2)])
            vel[i] = ti.Vector([-v_o * ti.sin(ang2), v_o * ti.cos(ang2)])
            temp[i] = 0.28 + ti.random() * 0.32 if swallowed else 0.20 + ti.random() * 0.25
            age[i] = 0.0
            ptype[i] = 2
            if swallowed:
                element[i] = 0  # Re-melted into light disk material


@ti.kernel
def trigger_explosion(n: ti.i32, cx: ti.f32, cy: ti.f32,
                      exp_spd: ti.f32, jet_frac: ti.f32):
    """
    Kilonova explosion with relativistic jets.
    90% of particles: spherical ejecta with v^(-1.5) power-law distribution.
    ~8% of particles: bipolar jets (narrow, fast, hot).
    Physically motivated by:
      - Spherical ejecta: tidal + shock-driven mass ejection
      - Jets: Blandford-Znajek mechanism (spinning BH + magnetic field)
    """
    for i in range(n):
        if alive[i] == 0:
            continue

        ang = ti.random() * 2.0 * PI

        if ti.random() < jet_frac:
            # --- RELATIVISTIC JET ---
            # Jets are perpendicular to orbital plane (vertical on screen)
            jet_ang = PI / 2.0 + (ti.random() - 0.5) * JET_HALF_ANGLE
            if ti.random() < 0.5:
                jet_ang = -jet_ang   # Counter-jet
            jspd = exp_spd * JET_SPEED_MULT * (0.5 + ti.random() * 0.5)
            vel[i] = ti.Vector([jspd * ti.cos(jet_ang), jspd * ti.sin(jet_ang)])
            temp[i] = PEAK_TEMP          # Jets are ultrarelativistic-hot
            ptype[i] = 4                 # Jet particle
            element[i] = 3 if (ti.random() < 0.4) else 1  # Uranium / Platinum synthesis
        else:
            # --- SPHERICAL KILONOVA EJECTA ---
            # Power-law speed distribution: dM/dv ~ v^(-1.5)
            # More mass at lower speeds (physical: most ejecta is slow)
            spd_frac = ti.random() ** 1.8
            spd = exp_spd * (0.12 + spd_frac * 0.88)

            # Preserve ~20% of original angular momentum
            vel[i] = vel[i] * 0.20 + ti.Vector([spd * ti.cos(ang),
                                                  spd * ti.sin(ang)])
            # Temperature: exponential falloff from merger point
            dx = pos[i][0] - cx
            dy = pos[i][1] - cy
            dist = ti.sqrt(dx*dx + dy*dy + 1e-7)
            temp[i] = PEAK_TEMP * ti.exp(-dist * 18.0)
            temp[i] = ti.max(temp[i], 0.25)
            ptype[i] = 2
            
            # Element distribution based on ejecta expansion speed
            if spd_frac > 0.72:
                element[i] = 1 # Platinum (outer fast dynamic ejecta)
            elif spd_frac > 0.32:
                element[i] = 2 # Gold (intermediate-speed main r-process ejecta)
            else:
                element[i] = 3 # Uranium (inner slow high-density actinide ejecta)

        age[i] = 0.0


@ti.kernel
def spawn_at_cursor(n_total: ti.i32, mx: ti.f32, my: ti.f32):
    """Probabilistically move ~40 particles to cursor with random scatter."""
    for i in range(n_total):
        if ti.random() < 40.0 / ti.max(float(n_total), 1.0):
            ang = ti.random() * 2.0 * PI
            r   = ti.random() * 0.018
            pos[i]  = ti.Vector([mx + r*ti.cos(ang), my + r*ti.sin(ang)])
            vel[i]  = ti.Vector([(ti.random()-0.5)*0.008, (ti.random()-0.5)*0.008])
            temp[i] = 0.35 + ti.random() * 0.40
            age[i]  = 0.0
            alive[i] = 1


@ti.kernel
def activate_new(start: ti.i32, end: ti.i32, cx: ti.f32, cy: ti.f32):
    """Activate additional particles (spawned near center)."""
    for i in range(start, end):
        alive[i] = 1
        ang = ti.random() * 2.0 * PI
        r   = 0.015 + ti.random() * 0.07
        pos[i]  = ti.Vector([cx + r*ti.cos(ang), cy + r*ti.sin(ang)])
        vel[i]  = ti.Vector([(ti.random()-0.5)*0.003, (ti.random()-0.5)*0.003])
        temp[i] = 0.08 + ti.random() * 0.15
        age[i]  = 0.0
        ptype[i] = 3


@ti.kernel
def deactivate_range(start: ti.i32, end: ti.i32):
    """Deactivate particles beyond new count."""
    for i in range(start, end):
        alive[i] = 0


# =============================================================================
# RENDERING KERNELS
# =============================================================================

@ti.kernel
def decay_pixels(decay: ti.f32):
    """
    Image persistence: multiply all pixels by decay factor each frame.
    This creates glowing motion trails behind moving particles.
    Also enforces minimum background color (near-black with blue tint).
    """
    for i, j in pixels:
        pixels[i, j] *= decay
        # Enforce dark background floor
        for c in ti.static(range(3)):
            if pixels[i, j][c] < BG[c]:
                pixels[i, j][c] = BG[c]


@ti.kernel
def render_particles(n: ti.i32, color_mode: ti.i32, bright: ti.f32):
    """
    Splat each particle as a soft Gaussian glow onto the pixel buffer.
    Additive blending via atomic_add for overlapping glow regions.
    Glow radius scales with temperature (hotter = larger, brighter glow).
    """
    for idx in range(n):
        if alive[idx] == 0:
            continue

        px_f = pos[idx][0] * float(WIDTH)
        py_f = pos[idx][1] * float(HEIGHT)
        px_i = int(px_f)
        py_i = int(py_f)

        col = get_particle_color(idx, color_mode)

        # Glow radius: 2-5 pixels depending on temperature
        gr = GLOW_R_BASE + ti.cast(temp[idx] * 3.0, ti.i32)
        sigma2 = float(gr) * 0.55
        sigma2 = sigma2 * sigma2 + 0.5

        # Brightness scales with temperature
        base_b = bright * (0.25 + temp[idx] * 0.75)

        for dx in range(-gr, gr + 1):
            for dy in range(-gr, gr + 1):
                ix = px_i + dx
                iy = py_i + dy
                if 0 <= ix < WIDTH and 0 <= iy < HEIGHT:
                    d2 = float(dx * dx + dy * dy)
                    glow = base_b * ti.exp(-d2 / (2.0 * sigma2))
                    ti.atomic_add(pixels[ix, iy][0], col[0] * glow)
                    ti.atomic_add(pixels[ix, iy][1], col[1] * glow)
                    ti.atomic_add(pixels[ix, iy][2], col[2] * glow)


@ti.kernel
def render_star_glow(sx: ti.f32, sy: ti.f32, radius: ti.i32,
                     cr: ti.f32, cg: ti.f32, cb: ti.f32, bright: ti.f32):
    """
    Neutron star glow: bright 1/r^2 core + softer exponential halo.
    Core is near-white; halo tinted by star color.
    """
    px = int(sx * float(WIDTH))
    py = int(sy * float(HEIGHT))
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            ix = px + dx
            iy = py + dy
            if 0 <= ix < WIDTH and 0 <= iy < HEIGHT:
                d = ti.sqrt(float(dx*dx + dy*dy) + 0.8)
                # Sharp bright core (1/r^2 falloff)
                core = bright * 2.5 / (d * d + 0.8)
                # Soft halo (exponential falloff)
                halo = bright * 0.4 * ti.exp(-d / (float(radius) * 0.28 + 1.0))
                g = core + halo
                ti.atomic_add(pixels[ix, iy][0], cr * g)
                ti.atomic_add(pixels[ix, iy][1], cg * g)
                ti.atomic_add(pixels[ix, iy][2], cb * g)


@ti.kernel
def render_effects(cx: ti.f32, cy: ti.f32, sim_t: ti.f32,
                   shock_r: ti.f32, shock_a: ti.f32,
                   gw_amp: ti.f32, gw_wl: ti.f32,
                   show_gw: ti.i32, show_shock: ti.i32,
                   flash_a: ti.f32,
                   is_black_hole: ti.i32, pulsar_angle: ti.f32,
                   remnant_active: ti.i32, grb_a: ti.f32):
    """
    Screen-space effects pass:
      1. Collision flash (brief white overlay)
      2. Gamma-Ray Burst (collimated vertical jet beam of energy)
      3. Gravitational wave ripples (inspiral concentric rings)
      4. Shockwave rings (blue-white fast ejecta & gold-orange slow kilonova ejecta)
      5. Black Hole Remnant (event horizon overlay + Einstein lensing ring)
      6. Magnetar Remnant (bipolar rotating pulsar beams + poloidal dipole loops)
      7. Tone mapping / clamping
    """
    aspect = float(WIDTH) / float(HEIGHT)
    for i, j in pixels:
        fx = float(i) / float(WIDTH)
        fy = float(j) / float(HEIGHT)
        
        # Aspect-corrected distance relative to center
        ddx = (fx - cx) * aspect
        ddy = fy - cy
        dist = ti.sqrt(ddx * ddx + ddy * ddy + 1e-10)

        # ---- Collision flash ----
        if flash_a > 0.005:
            pixels[i, j][0] += flash_a * 0.95
            pixels[i, j][1] += flash_a * 0.97
            pixels[i, j][2] += flash_a * 1.00

        # ---- Gamma-Ray Burst (Collimated Bipolar Jet Beam) ----
        if grb_a > 0.005:
            dx_c = ti.abs(fx - cx) * aspect
            # Relativistic beam along vertical center (x = cx)
            beam = grb_a * ti.exp(-dx_c * dx_c / 0.00016)
            pixels[i, j][0] += beam * 0.70
            pixels[i, j][1] += beam * 0.85
            pixels[i, j][2] += beam * 1.00

        # ---- Gravitational wave ripples (during inspiral) ----
        if show_gw == 1 and gw_amp > 0.002:
            wave = ti.sin(dist * gw_wl - sim_t * GW_ANIM_SPEED)
            wave = wave * wave                       # Energy density (always positive)
            amp = gw_amp * wave / (dist * 10.0 + 0.4)
            amp *= ti.exp(-dist * 2.0)               # Exponential envelope
            # Subtle purple-indigo tint (artistic choice for GW visualization)
            pixels[i, j][0] += amp * 0.22
            pixels[i, j][1] += amp * 0.10
            pixels[i, j][2] += amp * 0.42

        # ---- Shockwave rings (kilonova) ----
        if show_shock == 1 and shock_r > 0.005:
            # PRIMARY: blue-white (fast, lanthanide-free "blue kilonova")
            rd1 = ti.abs(dist - shock_r)
            if rd1 < SHOCK_THICK * 4.5:
                ring1 = shock_a * ti.exp(-rd1*rd1 / (2.0*SHOCK_THICK*SHOCK_THICK + 1e-7))
                pixels[i, j][0] += ring1 * 0.45
                pixels[i, j][1] += ring1 * 0.65
                pixels[i, j][2] += ring1 * 1.00

            # SECONDARY: orange-gold (slow, lanthanide-rich "red kilonova")
            # This is where gold/platinum emission dominates
            shock2_r = shock_r * SHOCK2_RATIO
            shock2_a = shock_a * 0.55
            thk2 = SHOCK_THICK * 1.4
            rd2 = ti.abs(dist - shock2_r)
            if rd2 < thk2 * 4.5:
                ring2 = shock2_a * ti.exp(-rd2*rd2 / (2.0*thk2*thk2 + 1e-7))
                pixels[i, j][0] += ring2 * 1.00
                pixels[i, j][1] += ring2 * 0.65
                pixels[i, j][2] += ring2 * 0.15

        # ---- Black Hole Remnant ----
        if remnant_active == 1 and is_black_hole == 1:
            if dist < EH_RADIUS:
                # Swallowed space (event horizon)
                pixels[i, j] = ti.Vector([0.001, 0.001, 0.003])
            elif dist < EH_RADIUS * 1.6:
                # Einstein Ring / gravitational lensing accretion boundary
                f = (dist - EH_RADIUS) / (EH_RADIUS * 0.6)
                lensing = ti.exp(-f * f * 2.5) * 1.35 * (0.35 + 0.65 * shock_a)
                pixels[i, j][0] += lensing * 1.0
                pixels[i, j][1] += lensing * 0.55
                pixels[i, j][2] += lensing * 0.15

        # ---- Magnetar Remnant (rotating pulsar beams and dipole field loops) ----
        if remnant_active == 1 and is_black_hole == 0:
            # Rotate coordinates to align with magnetic axis (pulsar_angle)
            ca = ti.cos(-pulsar_angle)
            sa = ti.sin(-pulsar_angle)
            rx = ddx * ca - ddy * sa
            ry = ddx * sa + ddy * ca
            r_p = ti.sqrt(rx * rx + ry * ry + 1e-12)
            theta_p = ti.atan2(ry, rx)

            # 1. Narrow Bipolar searchlight beams (along x-axis in rotated frame)
            diff1 = ti.abs(theta_p)
            diff2 = ti.abs(theta_p - PI)
            diff3 = ti.abs(theta_p + PI)
            min_diff = ti.min(ti.min(diff1, diff2), diff3)
            
            if min_diff < 0.06 and r_p < 0.45:
                beam_factor = ti.exp(-min_diff * min_diff / 0.0006) * (1.0 - r_p / 0.45)
                # Rapidly spinning light searchlight
                beam_glow = beam_factor * 0.85 * (0.8 + 0.2 * ti.sin(sim_t * 24.0)) * (0.4 + 0.6 * shock_a)
                pixels[i, j][0] += beam_glow * 0.65
                pixels[i, j][1] += beam_glow * 0.82
                pixels[i, j][2] += beam_glow * 1.00

            # 2. Poloidal dipole field loops (r_p = L * sin^2(theta_p))
            sin_sq = ti.sin(theta_p)
            sin_sq = sin_sq * sin_sq
            for L in ti.static([0.08, 0.16, 0.26]):
                diff_l = ti.abs(r_p - L * sin_sq)
                if diff_l < 0.0035 * (1.0 + r_p * 4.0):
                    loop_glow = 0.25 * ti.exp(-diff_l / 0.002) * (1.0 - r_p / 0.38) * (0.4 + 0.6 * shock_a)
                    pixels[i, j][0] += loop_glow * 0.22
                    pixels[i, j][1] += loop_glow * 0.58
                    pixels[i, j][2] += loop_glow * 1.00

        # ---- Tone mapping ----
        for c in ti.static(range(3)):
            pixels[i, j][c] = ti.min(pixels[i, j][c], 1.0)


# =============================================================================
# SLIDER UI SYSTEM (manual implementation — no dependencies)
# =============================================================================

class Slider:
    """On-screen horizontal slider with label, track, handle, and value display."""
    def __init__(self, label, y_pos, min_v, max_v, default, fmt=".2f"):
        self.label   = label
        self.y       = y_pos
        self.min_v   = min_v
        self.max_v   = max_v
        self.value   = default
        self.fmt     = fmt
        self.x0      = 0.028     # Track start (normalized x)
        self.x1      = 0.158     # Track end   (normalized x)

    def handle_x(self):
        frac = (self.value - self.min_v) / (self.max_v - self.min_v + 1e-12)
        return self.x0 + frac * (self.x1 - self.x0)

    def hit(self, mx, my):
        hx = self.handle_x()
        return abs(mx - hx) < 0.018 and abs(my - self.y) < 0.016

    def drag(self, mx):
        frac = (mx - self.x0) / (self.x1 - self.x0 + 1e-12)
        frac = max(0.0, min(1.0, frac))
        self.value = self.min_v + frac * (self.max_v - self.min_v)

    def draw(self, gui):
        hx = self.handle_x()
        # Track background
        gui.line(begin=(self.x0, self.y), end=(self.x1, self.y),
                 radius=2, color=0x2A2A3F)
        # Filled portion
        gui.line(begin=(self.x0, self.y), end=(hx, self.y),
                 radius=2, color=0x4466AA)
        # Handle
        gui.circle(pos=(hx, self.y), color=0x99BBEE, radius=6)
        # Label
        gui.text(self.label, pos=(self.x0, self.y + 0.018),
                 font_size=12, color=0x7799CC)
        # Value
        gui.text(f"{self.value:{self.fmt}}", pos=(self.x1 + 0.007, self.y - 0.004),
                 font_size=11, color=0x6688AA)


# =============================================================================
# MAIN SIMULATION LOOP
# =============================================================================

def main():
    # ---- Startup console output ----
    print()
    print("=" * 52)
    print("   * NEUTRON STAR COLLISION SIMULATION *")
    print("=" * 52)
    print()
    print("  ON-SCREEN SLIDERS: Adjust parameters in real-time")
    print("  SPACE: Pause/Resume | R: Reset | Q: Quit")
    print("  C: Cycle color mode")
    print("  LEFT-CLICK: Spawn particles | HOLD+DRAG: Apply force")
    print("  FPS: [live counter in window]")
    print("=" * 52)
    print()

    # ---- GUI Window ----
    gui = ti.GUI("* Neutron Star Collision * | FPS: --",
                 res=(WIDTH, HEIGHT), background_color=0x000003)

    # ---- Simulation state (Python side) ----
    phase       = 0        # 0=inspiral, 1=flash, 2=kilonova, 3=aftermath
    paused      = False
    sim_time    = 0.0
    separation  = INIT_SEPARATION
    theta       = 0.0       # Orbital angle [radians]
    m1          = NS_MASS_1
    m2          = NS_MASS_2
    m_total     = m1 + m2
    omega       = 0.0       # Current orbital angular velocity

    # Remnant & TOV Limit state
    is_black_hole  = 0     # 1 if total mass > TOV limit, collapses to BH
    pulsar_angle   = 0.0   # Current spin angle of pulsar beams
    grb_a          = 0.0   # GRB collimated beam intensity
    remnant_active = 0     # Whether the central remnant is active

    # Merger state
    merge_x, merge_y = 0.5, 0.5
    shock_r   = 0.0
    shock_a   = 0.0
    flash_a   = 0.0
    flash_tmr = 0.0

    # Star positions (computed analytically from orbital params)
    s1x = s1y = s2x = s2y = 0.5
    s1vx = s1vy = s2vx = s2vy = 0.0

    # FPS tracking
    fps_cnt   = 0
    fps_timer = time.time()
    fps_val   = 0.0

    # Active particle count
    n_particles = DEFAULT_N

    # Color mode
    color_modes = ["Temperature", "Velocity", "Age", "Field Strength", "Element Synthesis"]
    color_mode  = 0

    # ---- Sliders ----
    sl_gravity   = Slider("Gravity",          0.93, 0.1, 5.0, 1.0)
    sl_time      = Slider("Time Speed",       0.865, 0.1, 5.0, 1.0)
    sl_particles = Slider("Particles (k)",    0.80, 5.0, 15.0, 10.0, ".1f")
    sl_damping   = Slider("Energy Damping",   0.735, 0.0, 1.0, 0.15, ".2f")
    sl_trail     = Slider("Trail Length",      0.67, 0.80, 0.99, TRAIL_DECAY_DEF, ".3f")
    sl_massratio = Slider("Mass Ratio m1/m2", 0.605, 0.5, 2.0, 1.0)
    sl_explosion = Slider("Explosion Power",  0.54, 0.5, 3.0, 1.0)

    all_sliders = [sl_gravity, sl_time, sl_particles, sl_damping,
                   sl_trail, sl_massratio, sl_explosion]

    # ---- Helper: compute star state from orbital parameters ----
    def star_state(sep, angle, mm1, mm2):
        mt = mm1 + mm2
        om = math.sqrt(G * mt / (sep ** 3 + 1e-18))
        r1 = (mm2 / mt) * sep
        r2 = (mm1 / mt) * sep
        sx1 = 0.5 + r1 * math.cos(angle)
        sy1 = 0.5 + r1 * math.sin(angle)
        sx2 = 0.5 - r2 * math.cos(angle)
        sy2 = 0.5 - r2 * math.sin(angle)
        v1  = om * r1
        v2  = om * r2
        vx1 = -v1 * math.sin(angle)
        vy1 =  v1 * math.cos(angle)
        vx2 =  v2 * math.sin(angle)
        vy2 = -v2 * math.cos(angle)
        return sx1, sy1, sx2, sy2, vx1, vy1, vx2, vy2, om

    # ---- Reset function ----
    def reset():
        nonlocal phase, sim_time, separation, theta, m1, m2, m_total, omega
        nonlocal shock_r, shock_a, flash_a, flash_tmr, merge_x, merge_y
        nonlocal n_particles, s1x, s1y, s2x, s2y, s1vx, s1vy, s2vx, s2vy
        nonlocal is_black_hole, pulsar_angle, grb_a, remnant_active

        phase      = 0
        sim_time   = 0.0
        separation = INIT_SEPARATION
        theta      = 0.0
        m1         = NS_MASS_1 * sl_massratio.value
        m2         = NS_MASS_2
        m_total    = m1 + m2
        shock_r    = 0.0
        shock_a    = 0.0
        flash_a    = 0.0
        flash_tmr  = 0.0
        merge_x    = 0.5
        merge_y    = 0.5
        is_black_hole  = 0
        pulsar_angle   = 0.0
        grb_a          = 0.0
        remnant_active = 0
        n_particles = int(sl_particles.value * 1000)
        n_particles = min(n_particles, MAX_P)

        s1x, s1y, s2x, s2y, s1vx, s1vy, s2vx, s2vy, omega = \
            star_state(separation, theta, m1, m2)
        init_particles(n_particles,
                       s1x, s1y, s2x, s2y,
                       s1vx, s1vy, s2vx, s2vy,
                       m1, m2)
        # Clear pixel buffer
        decay_pixels(0.0)

    reset()

    # ---- Mouse state ----
    active_slider  = None
    canvas_pressed = False

    # ================================================================
    # MAIN LOOP
    # ================================================================
    while gui.running:

        # ---- EVENT HANDLING ----
        for e in gui.get_events(ti.GUI.PRESS, ti.GUI.RELEASE):
            if e.type == ti.GUI.PRESS:
                if e.key == ti.GUI.SPACE:
                    paused = not paused
                elif e.key == 'r':
                    reset()
                elif e.key == 'q' or e.key == ti.GUI.ESCAPE:
                    gui.running = False
                elif e.key == 'c':
                    color_mode = (color_mode + 1) % len(color_modes)
                elif e.key == ti.GUI.LMB:
                    mx, my = gui.get_cursor_pos()
                    # Check sliders first
                    sl_hit = False
                    for s in all_sliders:
                        if s.hit(mx, my):
                            active_slider = s
                            sl_hit = True
                            break
                    if not sl_hit:
                        # Color mode button region
                        if 0.02 <= mx <= 0.17 and 0.47 <= my <= 0.51:
                            color_mode = (color_mode + 1) % len(color_modes)
                        else:
                            canvas_pressed = True
                            spawn_at_cursor(n_particles, mx, my)

            elif e.type == ti.GUI.RELEASE:
                if e.key == ti.GUI.LMB:
                    active_slider  = None
                    canvas_pressed = False

        # Continuous mouse tracking
        mx, my = gui.get_cursor_pos()
        is_lmb = gui.is_pressed(ti.GUI.LMB)

        if is_lmb and active_slider is not None:
            active_slider.drag(mx)
        if not is_lmb:
            canvas_pressed = False
            active_slider  = None

        # ---- READ SLIDER VALUES ----
        g_mult    = sl_gravity.value
        time_spd  = sl_time.value
        new_n     = min(int(sl_particles.value * 1000), MAX_P)
        damp_val  = 1.0 - sl_damping.value * 0.035
        trail_d   = sl_trail.value
        expl_mult = sl_explosion.value

        if phase == 0:
            m1      = NS_MASS_1 * sl_massratio.value
            m_total = m1 + m2

        # Handle particle count changes
        if new_n > n_particles:
            activate_new(n_particles, new_n, 0.5, 0.5)
        elif new_n < n_particles:
            deactivate_range(new_n, n_particles)
        n_particles = new_n

        dt = DT_BASE * time_spd
        mouse_active = 1 if (canvas_pressed and is_lmb) else 0

        # ---- PHYSICS UPDATE ----
        if not paused:
            sim_time += dt

            # Substepping loop (5 steps per frame) for fast, stable orbits
            substeps = 5
            sub_dt = dt / substeps

            for _ in range(substeps):
                if phase == 0:
                    # ======= INSPIRAL =======
                    # Peters formula orbital decay: da/dt = -C / a^3
                    # We add a small extra chirp factor near contact for faster acceleration
                    chirp_acc = 1.0 + max(0.0, (INIT_SEPARATION - separation) / INIT_SEPARATION) * 1.5
                    separation -= GW_DECAY_COEFF * g_mult * chirp_acc / (separation**3 + 1e-18) * time_spd / substeps
                    separation = max(separation, 0.001)

                    # Kepler's third law: omega^2 = G * M_total / a^3
                    omega = math.sqrt(G * m_total * g_mult / (separation**3 + 1e-18))
                    theta += omega * sub_dt

                    s1x, s1y, s2x, s2y, s1vx, s1vy, s2vx, s2vy, _ = \
                        star_state(separation, theta, m1, m2)

                    step_binary(n_particles, s1x, s1y, s2x, s2y, m1, m2,
                                sub_dt, damp_val, g_mult, mx, my, mouse_active)

                    # Check merger trigger
                    if separation < MERGE_DISTANCE:
                        phase   = 1
                        merge_x = (s1x * m2 + s2x * m1) / m_total  # Center of mass
                        merge_y = (s1y * m2 + s2y * m1) / m_total
                        flash_a   = 1.0
                        flash_tmr = 0.0
                        shock_r   = 0.0
                        shock_a   = 0.85
                        grb_a     = 1.0  # Launch Gamma-Ray Burst vertical beam
                        
                        # Check TOV Mass Limit: collapses to BH if mass ratio > limit
                        is_black_hole = 1 if (m_total > TOV_LIMIT) else 0
                        remnant_active = 1

                        trigger_explosion(n_particles, merge_x, merge_y,
                                          EXPLODE_SPEED * expl_mult, JET_FRACTION)
                        print()
                        print("  >>> MERGER! Kilonova initiated!")
                        print(f"      Collision at ({merge_x:.3f}, {merge_y:.3f})")
                        print(f"      Total mass: {m_total:.2f} sim units (TOV Limit: {TOV_LIMIT:.2f})")
                        if is_black_hole == 1:
                            print("  >>> MASS EXCEEDS TOV LIMIT! Collapsing into a BLACK HOLE!")
                        else:
                            print("  >>> MASS BELOW TOV LIMIT! Settling into a rapidly spinning MAGNETAR!")
                        print()
                        break # Break out of substeps to enter Phase 1 immediately

                elif phase == 1:
                    # ======= COLLISION FLASH =======
                    flash_tmr += sub_dt
                    flash_a = max(0.0, 1.0 - flash_tmr * 45.0)
                    # GRB decays rapidly
                    grb_a = max(0.0, grb_a * math.exp(-sub_dt * 15.0))

                    shock_r += SHOCK_SPEED * time_spd / substeps
                    shock_a = max(0.0, 0.85 - shock_r * 1.3)

                    step_merged(n_particles, merge_x, merge_y, m_total,
                                sub_dt, damp_val, g_mult, mx, my, mouse_active, is_black_hole)

                    if flash_a <= 0.005:
                        phase = 2
                        print("  >>> Kilonova expansion: r-process nucleosynthesis active")
                        print("      (Heavy elements — gold, platinum — being forged)")

                elif phase == 2:
                    # ======= KILONOVA EXPANSION =======
                    # GRB decays slightly slower
                    grb_a = max(0.0, grb_a * math.exp(-sub_dt * 8.0))
                    
                    shock_r += SHOCK_SPEED * time_spd / substeps
                    shock_a = max(0.0, 0.65 - shock_r * 0.9)

                    step_merged(n_particles, merge_x, merge_y, m_total,
                                sub_dt, damp_val, g_mult, mx, my, mouse_active, is_black_hole)

                    if shock_r > 0.85:
                        phase = 3
                        print("  >>> Remnant settling (hypermassive NS or black hole)")

                elif phase == 3:
                    # ======= AFTERMATH =======
                    grb_a = max(0.0, grb_a * math.exp(-sub_dt * 4.0))
                    if shock_a > 0.005:
                        shock_r += SHOCK_SPEED * time_spd * 0.4 / substeps
                        shock_a *= 0.994

                    step_merged(n_particles, merge_x, merge_y, m_total,
                                sub_dt, damp_val, g_mult, mx, my, mouse_active, is_black_hole)

        # Update pulsar rotation if magnetar is active and simulation is running
        if remnant_active == 1 and is_black_hole == 0 and not paused:
            pulsar_angle += 85.0 * dt
            if pulsar_angle > 2.0 * PI:
                pulsar_angle -= 2.0 * PI

        # ---- RENDERING ----

        # 1) Decay pixel buffer (creates trails)
        decay_pixels(trail_d)

        # 2) Render particle glow
        render_particles(n_particles, color_mode, P_BRIGHT)

        # 3) Render star / remnant glow
        if phase == 0:
            # Two orbiting neutron stars
            # Subtle brightness pulse at twice orbital frequency (tidal bulge)
            pulse = 0.92 + 0.08 * math.sin(theta * 2.0)
            render_star_glow(s1x, s1y, STAR_GLOW_R,
                             0.55, 0.72, 1.0, S_BRIGHT * pulse)
            render_star_glow(s2x, s2y, STAR_GLOW_R,
                             0.65, 0.78, 1.0, S_BRIGHT * (m2/m1) * pulse)
        elif phase == 1:
            # Bright merger flash remnant
            render_star_glow(merge_x, merge_y, STAR_GLOW_R + 12,
                             0.95, 0.97, 1.0, S_BRIGHT * 1.8)
        else:
            # Phase 2 & 3: Remnant core settling
            if is_black_hole == 1:
                # Black Hole: render accretion disk glow (event horizon itself is drawn in effects)
                rb = S_BRIGHT * 0.65 * max(0.15, 1.0 - shock_r * 0.9)
                render_star_glow(merge_x, merge_y, STAR_GLOW_R + 8,
                                 1.0, 0.52, 0.12, rb)
            else:
                # Magnetar: render hot blue-white core
                rb = S_BRIGHT * 1.40 * max(0.40, 1.0 - shock_r * 0.9)
                render_star_glow(merge_x, merge_y, STAR_GLOW_R - 4,
                                 0.50, 0.75, 1.0, rb)

        # 4) Screen-space effects (GW ripples, shockwave, flash, tone map)
        do_gw    = 1 if phase == 0 else 0
        do_shock = 1 if phase >= 1 else 0

        # GW amplitude increases as separation shrinks (physical: amplitude ~ 1/r)
        gw_a = 0.0
        gw_wl = GW_BASE_WAVELEN
        if phase == 0 and separation > 0.001:
            gw_a  = GW_MAX_ALPHA * (INIT_SEPARATION / (separation + 0.005))
            gw_a  = min(gw_a, 0.14)
            # GW frequency chirp: wavelength decreases as omega increases
            gw_wl = max(20.0, GW_BASE_WAVELEN * 0.5 / (omega + 0.5))

        render_effects(0.5, 0.5, sim_time,
                       shock_r, shock_a,
                       gw_a, gw_wl,
                       do_gw, do_shock,
                       flash_a if phase == 1 else 0.0,
                       is_black_hole, pulsar_angle, remnant_active, grb_a)

        # 5) Set image from pixel buffer
        gui.set_image(pixels)

        # ---- UI OVERLAY ----

        # Slider panel edge lines (subtle frame)
        gui.line((0.005, 0.50), (0.005, 0.96), radius=1, color=0x1A1A2E)
        gui.line((0.175, 0.50), (0.175, 0.96), radius=1, color=0x1A1A2E)

        for s in all_sliders:
            s.draw(gui)

        # Color mode display + click area
        gui.text(f"Color: {color_modes[color_mode]}",
                 pos=(0.028, 0.50), font_size=12, color=0x88AADD)
        gui.text("[click / C]",
                 pos=(0.028, 0.48), font_size=10, color=0x445566)

        # Element Synthesis Legend (Only shows when Element Synthesis mode is selected)
        if color_mode == 4:
            gui.text("Synthesis Zone:", pos=(0.028, 0.44), font_size=11, color=0x7799CC)
            # Platinum (Silver/Blue-white)
            gui.rect((0.028, 0.41), (0.040, 0.425), color=0xDCEEFA)
            gui.text("Platinum (Pt)", pos=(0.048, 0.41), font_size=10, color=0xCCDDF8)
            # Gold (Au)
            gui.rect((0.028, 0.38), (0.040, 0.395), color=0xFFC70D)
            gui.text("Gold (Au)", pos=(0.048, 0.38), font_size=10, color=0xFFDD44)
            # Uranium (U)
            gui.rect((0.028, 0.35), (0.040, 0.365), color=0x38F266)
            gui.text("Uranium (U)", pos=(0.048, 0.35), font_size=10, color=0x58FF88)
            # Iron / ISM
            gui.rect((0.028, 0.32), (0.040, 0.335), color=0x1A73BF)
            gui.text("ISM / Iron (Fe)", pos=(0.048, 0.32), font_size=10, color=0x3F9FEF)

        # Phase indicator
        pnames  = ["INSPIRAL", "COLLISION!", "KILONOVA", "AFTERMATH"]
        pcolors = [0x4488FF, 0xFFFF55, 0xFF8844, 0x777788]
        gui.text(pnames[phase], pos=(0.42, 0.965),
                 font_size=18, color=pcolors[phase])
                 
        if phase >= 2:
            rname = "BLACK HOLE remnant" if is_black_hole == 1 else "MAGNETAR remnant"
            rcolor = 0xFF5555 if is_black_hole == 1 else 0x55FFFF
            gui.text(rname, pos=(0.39, 0.935), font_size=12, color=rcolor)

        # Orbital info during inspiral
        if phase == 0:
            gui.text(f"Separation: {separation:.4f}",
                     pos=(0.38, 0.030), font_size=12, color=0x5577AA)
            # Physical scaling for the frequency chirp (from ~30 Hz to 1000+ Hz)
            freq_hz = omega * 480.0 / (2.0 * PI)
            gui.text(f"GW Freq: {freq_hz:.1f} Hz (chirp)",
                     pos=(0.38, 0.010), font_size=12, color=0x5577AA)

        # Pause indicator
        if paused:
            gui.text("|| PAUSED", pos=(0.46, 0.50),
                     font_size=22, color=0xFFAA33)

        # FPS counter
        fps_cnt += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            fps_val   = fps_cnt / elapsed
            fps_cnt   = 0
            fps_timer = time.time()
            try:
                gui.name = (f"* Neutron Star Collision * | "
                            f"FPS: {fps_val:.0f} | "
                            f"Particles: {n_particles}")
            except Exception:
                pass  # Some Taichi versions don't support title update

        fps_color = 0x44FF44 if fps_val >= 24 else (0xFFFF44 if fps_val >= 15 else 0xFF4444)
        gui.text(f"FPS: {fps_val:.0f}", pos=(0.91, 0.965),
                 font_size=14, color=fps_color)
        gui.text(f"N: {n_particles}", pos=(0.91, 0.940),
                 font_size=11, color=0x6688AA)

        # Controls reminder
        gui.text("SPACE:Pause  R:Reset  Q:Quit  C:Color  Click:Spawn  Drag:Force",
                 pos=(0.22, 0.004), font_size=10, color=0x334455)

        # 7) Show frame
        gui.show()

    print()
    print("Simulation ended.")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
