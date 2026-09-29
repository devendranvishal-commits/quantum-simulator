"""
BUILD-YOUR-OWN MOLECULAR DOCKING SIMULATOR
===============================================
Built for Vishal, OSU Tech Innovation / BuildX prep project.

What this is:
    Real drug discovery uses "molecular docking" to predict whether a small
    molecule (a potential drug) will bind tightly to a target protein (often
    something involved in a disease). Companies like Isomorphic Labs,
    Insilico Medicine, and Schrodinger build massive, industrial versions of
    exactly this. This file is a genuine, working, from-scratch version of
    the same core idea, built with nothing but numpy/scipy so it runs on
    any laptop with no special installs.

The physics (simplified but real):
    Atoms attract each other a little at medium range and repel strongly if
    they get too close (this is the Lennard-Jones potential, a real,
    widely-used approximation of how atoms interact). A molecule "binds
    well" to a protein pocket when it can find a position/rotation where
    it fits snugly - close enough to attract, not so close that atoms
    clash - i.e. the LOWEST possible energy score.

Three parts:
    PART 1: Represent atoms/molecules as 3D point sets with simple types.
    PART 2: The scoring function (Lennard-Jones interaction energy).
    PART 3: The "docking" search - try many positions/rotations of a
            candidate molecule against a fixed protein pocket, and find
            the best-fitting pose (lowest energy) using scipy optimization.
            Then screen several candidate molecules and rank them - a tiny,
            real version of "virtual screening" used in early drug discovery.

Run it with:  python3 docking_demo.py
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

np.set_printoptions(precision=3, suppress=True)

# ======================================================================
# PART 1: ATOMS AND MOLECULES
# ======================================================================
# Every atom is (x, y, z, radius, well_depth). radius/well_depth are
# simplified Lennard-Jones parameters (roughly: bigger radius = bigger
# atom, deeper well = stronger attraction at ideal distance). Real force
# fields (AMBER, CHARMM) use looked-up values per element; we use
# representative simplified values so the physics stays real.

CARBON = dict(radius=1.7, well=0.10)
OXYGEN = dict(radius=1.52, well=0.20)
NITROGEN = dict(radius=1.55, well=0.16)


def atom(x, y, z, kind):
    return np.array([x, y, z, kind["radius"], kind["well"]])


# A tiny, fixed "protein binding pocket" - a handful of atoms arranged to
# form a small cavity, standing in for a real protein active site.
PROTEIN_POCKET = np.array([
    atom(0.0, 0.0, 0.0, OXYGEN),
    atom(3.0, 0.0, 0.0, CARBON),
    atom(0.0, 3.0, 0.0, CARBON),
    atom(0.0, 0.0, 3.0, NITROGEN),
    atom(2.0, 2.0, 2.0, CARBON),
    atom(-2.0, 1.0, 1.0, OXYGEN),
])

# Candidate "drug" molecules - small sets of atoms with a shape and
# chemistry of their own. In real drug discovery these would come from a
# chemical database of thousands of real compounds; we use 3 toy examples
# to demonstrate ranking/screening.
CANDIDATES = {
    "Molecule A (small, oxygen-rich)": np.array([
        atom(0.0, 0.0, 0.0, OXYGEN),
        atom(1.4, 0.0, 0.0, CARBON),
        atom(0.0, 1.4, 0.0, OXYGEN),
    ]),
    "Molecule B (bulky, carbon chain)": np.array([
        atom(0.0, 0.0, 0.0, CARBON),
        atom(1.5, 0.0, 0.0, CARBON),
        atom(3.0, 0.0, 0.0, CARBON),
        atom(4.5, 0.0, 0.0, CARBON),
    ]),
    "Molecule C (nitrogen-containing)": np.array([
        atom(0.0, 0.0, 0.0, NITROGEN),
        atom(1.4, 0.0, 0.0, CARBON),
        atom(1.4, 1.4, 0.0, CARBON),
    ]),
}


# ======================================================================
# PART 2: THE SCORING FUNCTION (Lennard-Jones interaction energy)
# ======================================================================

def lj_energy(protein, ligand):
    """
    Sum the Lennard-Jones interaction energy between every protein atom
    and every ligand (candidate molecule) atom. Lower (more negative) =
    better fit. This is a real, standard approximation used in
    computational chemistry, simplified here for clarity.
    """
    total = 0.0
    for p in protein:
        px, py, pz, pr, pwell = p
        for l in ligand:
            lx, ly, lz, lr, lwell = l
            dx, dy, dz = px - lx, py - ly, pz - lz
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            if dist < 1e-6:
                dist = 1e-6
            sigma = (pr + lr) / 2.0            # combined ideal contact distance
            epsilon = np.sqrt(pwell * lwell)   # combined attraction strength
            ratio = sigma / dist
            # Standard 12-6 Lennard-Jones potential:
            total += 4 * epsilon * (ratio**12 - ratio**6)
    return total


# ======================================================================
# PART 3: DOCKING - SEARCH FOR THE BEST-FITTING POSE
# ======================================================================

def transform_ligand(ligand, params):
    """
    Move and rotate the candidate molecule as a rigid body:
    params = [tx, ty, tz, rx, ry, rz] (translation + rotation angles).
    This is what a real docking algorithm searches over - where and how
    the molecule sits inside the pocket.
    """
    tx, ty, tz, rx, ry, rz = params
    coords = ligand[:, :3]
    rot = Rotation.from_euler("xyz", [rx, ry, rz])
    rotated = rot.apply(coords)
    translated = rotated + np.array([tx, ty, tz])
    new_ligand = ligand.copy()
    new_ligand[:, :3] = translated
    return new_ligand


def docking_objective(params, protein, ligand):
    """The number the optimizer minimizes: the binding energy of a given
    pose. Minimizing this = finding the tightest, most stable fit."""
    posed = transform_ligand(ligand, params)
    return lj_energy(protein, posed)


def dock(protein, ligand, seed=0):
    """
    Run the docking search: try an initial guess near the pocket center,
    then let scipy's optimizer explore translations/rotations to find the
    lowest-energy (best-fitting) pose. Real docking software (AutoDock,
    Glide) does a much more exhaustive version of exactly this search.
    """
    rng = np.random.default_rng(seed)
    best_result = None
    for _ in range(8):  # multiple random starting poses, keep the best
        x0 = np.concatenate([
            rng.uniform(-1, 1, size=3),      # starting position guess
            rng.uniform(0, 2 * np.pi, size=3)  # starting rotation guess
        ])
        result = minimize(docking_objective, x0, args=(protein, ligand),
                           method="Nelder-Mead",
                           options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4})
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    return best_result.fun, best_result.x


def demo_docking():
    print("=" * 70)
    print("MOLECULAR DOCKING SIMULATOR - virtual drug screening")
    print("=" * 70)
    print(f"Protein binding pocket: {len(PROTEIN_POCKET)} atoms\n")
    print("Screening candidate molecules against the pocket...")
    print("(lower / more negative energy score = tighter, better-fitting bind)\n")

    scores = {}
    for name, ligand in CANDIDATES.items():
        energy, pose = dock(PROTEIN_POCKET, ligand, seed=hash(name) % 1000)
        scores[name] = energy
        print(f"  {name}")
        print(f"      best binding energy: {energy:8.3f}")

    print("\n" + "-" * 70)
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    print("VIRTUAL SCREENING RESULT (best fit first):")
    for i, (name, energy) in enumerate(ranked, start=1):
        print(f"  #{i}: {name}  (energy = {energy:.3f})")

    print(f"\n-> {ranked[0][0]} is predicted to bind most tightly to this")
    print("   pocket, and would be the top candidate to test further")
    print("   (in real drug discovery: the next step would be lab testing,")
    print("   not a final answer - this simulation just narrows the field).")


if __name__ == "__main__":
    demo_docking()
