#!/usr/bin/env python3
"""
Magnetospheric structure of LSR J1835 (and any similar rotator) as a function
of emission latitude.

For a given latitude on the star it returns:

    L-shell            L = sec^2(lat), the equatorial crossing of the dipole
                       field line rooted at that latitude
    Mdot_required      the stellar mass-loss rate needed to put the
                       Hill-Pontius co-rotation-breakdown radius AT that
                       L-shell -- i.e. what you must believe if the emission
                       at that latitude is the main auroral oval
    v_corot            co-rotation speed at L
    B(L)               dipole equatorial field at L
    rho_max            density below which the Alfven radius lies beyond L
                       (required for co-rotation to be enforced out to L)
    H                  scale height of the centrifugally confined plasma disk
    v_drift            equatorial radial drift speed implied by Mdot and rho

Usage
-----
    python magnetosphere.py 38 80
    python magnetosphere.py --scan 30 89 --n 25
    python magnetosphere.py 38 --rstar 0.117 --bstar 1000 --period 2.84
"""

import argparse

import numpy as np

# cgs constants
C_LIGHT = 2.99792458e10          # cm/s
R_SUN = 6.957e10                 # cm
KEV = 1.602176634e-9             # erg
M_PROTON = 1.67262192e-24        # g
MDOT_SUN = 2.0e12                # g/s, solar mass-loss rate
MDOT_JUPITER = 1.0e6             # g/s, Io-driven, order of magnitude

# defaults for LSR J1835+3259
DEF = dict(rstar=0.098, bstar=1000.0, period=2.84, temp_kev=1.0, mu=1.0)

# the note's Hill-Pontius anchors, used to calibrate Sigma
ANCHORS = [(3.4, 1e14), (33.8, 1e10)]


# ---------------------------------------------------------------- geometry --

def lshell(lat_deg):
    """Equatorial crossing of the dipole field line rooted at `lat_deg`."""
    return 1.0 / np.cos(np.radians(lat_deg)) ** 2


def latitude(L):
    """Inverse of `lshell`: footpoint latitude of an L-shell, in degrees."""
    return np.degrees(np.arccos(1.0 / np.sqrt(L)))


def dipole_B(L, bstar):
    """Equatorial dipole field strength at L, in Gauss."""
    return bstar / L ** 3


# ------------------------------------------------------- Hill-Pontius radius --

def hp_prefactor(sigma, rstar_cm, bstar):
    """A in L_HP = A * Mdot^(-1/4), with Mdot in g/s.

    From L_HP/R* = [pi * Sigma * R*^2 * B*^2 / (Mdot c^2)]^(1/4), Gaussian
    units throughout: Sigma in cm/s, B in G, R in cm, Mdot in g/s.
    """
    return (np.pi * sigma * rstar_cm ** 2 * bstar ** 2 / C_LIGHT ** 2) ** 0.25


def calibrate_sigma(rstar_cm, bstar, anchors=ANCHORS):
    """Recover Sigma from the note's quoted (L_HP, Mdot) pairs."""
    A = np.mean([L * m ** 0.25 for L, m in anchors])
    sigma = A ** 4 * C_LIGHT ** 2 / (np.pi * rstar_cm ** 2 * bstar ** 2)
    spread = np.ptp([L * m ** 0.25 for L, m in anchors]) / A
    return sigma, A, spread


def hp_lshell(mdot, A):
    """Hill-Pontius co-rotation breakdown radius, in R*."""
    return A * mdot ** -0.25


def mdot_for_lshell(L, A):
    """Mass-loss rate that places the Hill-Pontius radius at L. Inverse of above."""
    return (A / L) ** 4


# ------------------------------------------------------------ plasma disk --

def corotation_speed(L, rstar_cm, period_h):
    """Co-rotation speed at L, in cm/s."""
    omega = 2.0 * np.pi / (period_h * 3600.0)
    return omega * L * rstar_cm


def max_density_for_alfven(L, bstar, rstar_cm, period_h):
    """Density below which v_Alfven > v_corot at L, i.e. the Alfven radius
    lies beyond L and co-rotation can still be enforced there. In g/cm^3.

    v_A = B / sqrt(4 pi rho) > v_corot  =>  rho < B^2 / (4 pi v_corot^2)
    """
    B = dipole_B(L, bstar)
    v = corotation_speed(L, rstar_cm, period_h)
    return B ** 2 / (4.0 * np.pi * v ** 2)


def scale_height(temp_kev, mu, period_h, rstar_cm):
    """Centrifugally confined disk scale height H = sqrt(kT/m)/Omega, in R*.

    Balance of plasma thermal pressure against centrifugal confinement.
    """
    omega = 2.0 * np.pi / (period_h * 3600.0)
    H_cm = np.sqrt(temp_kev * KEV / (mu * M_PROTON)) / omega
    return H_cm / rstar_cm


def drift_velocity(mdot, rho, L, H_over_R, rstar_cm):
    """Equatorial radial drift speed, in cm/s.

    Mdot = rho * v_drift * (2 pi r H), the convention that reproduces the
    note's 650 km/s.
    """
    r = L * rstar_cm
    H = H_over_R * rstar_cm
    return mdot / (rho * 2.0 * np.pi * r * H)


# ------------------------------------------------------------------- report --

def report(lat_deg, p, A, sigma):
    rstar_cm = p["rstar"] * R_SUN
    L = lshell(lat_deg)
    mdot = mdot_for_lshell(L, A)
    v = corotation_speed(L, rstar_cm, p["period"])
    B = dipole_B(L, p["bstar"])
    rho = max_density_for_alfven(L, p["bstar"], rstar_cm, p["period"])
    H = scale_height(p["temp_kev"], p["mu"], p["period"], rstar_cm)
    vd = drift_velocity(mdot, rho, L, H, rstar_cm)

    return dict(lat=lat_deg, L=L, mdot=mdot,
                mdot_solar=mdot / MDOT_SUN, mdot_jup=mdot / MDOT_JUPITER,
                v_corot_c=v / C_LIGHT, B=B, rho=rho, H=H,
                v_drift_kms=vd / 1e5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("latitudes", nargs="*", type=float,
                    help="emission latitudes in degrees")
    ap.add_argument("--scan", nargs=2, type=float, metavar=("LO", "HI"))
    ap.add_argument("--n", type=int, default=20, help="points in --scan")
    ap.add_argument("--rstar", type=float, default=DEF["rstar"],
                    help="stellar radius in Rsun (default 0.098, the value "
                         "implied by the note's co-rotation speed)")
    ap.add_argument("--bstar", type=float, default=DEF["bstar"],
                    help="equatorial surface field in Gauss")
    ap.add_argument("--period", type=float, default=DEF["period"],
                    help="rotation period in hours")
    ap.add_argument("--temp-kev", type=float, default=DEF["temp_kev"])
    ap.add_argument("--mu", type=float, default=DEF["mu"],
                    help="ion mass in proton masses")
    ap.add_argument("--sigma", type=float, default=None,
                    help="Pedersen conductance in cm/s (Gaussian). "
                         "Default: calibrated from the note's anchors.")
    ap.add_argument("--check", action="store_true",
                    help="reproduce the note's table and quoted values")
    args = ap.parse_args()

    p = dict(rstar=args.rstar, bstar=args.bstar, period=args.period,
             temp_kev=args.temp_kev, mu=args.mu)
    rstar_cm = p["rstar"] * R_SUN

    if args.sigma is None:
        sigma, A, spread = calibrate_sigma(rstar_cm, p["bstar"])
        src = f"calibrated from the note ({100*spread:.1f}% spread between anchors)"
    else:
        sigma = args.sigma
        A = hp_prefactor(sigma, rstar_cm, p["bstar"])
        src = "supplied"

    sigma_si = 1.11265e-10 * sigma * 1e-2      # cm/s -> Siemens
    print("=" * 78)
    print("  magnetospheric structure vs emission latitude")
    print("=" * 78)
    print(f"  R* = {p['rstar']:.3f} Rsun   B* = {p['bstar']:.0f} G   "
          f"P = {p['period']:.2f} h   T = {p['temp_kev']:.1f} keV   "
          f"mu = {p['mu']:.1f}")
    print(f"  Sigma = {sigma:.3e} cm/s = {sigma_si:.3f} S   ({src})")
    print(f"  Hill-Pontius prefactor A = {A:.4e}   [L_HP = A * Mdot^-1/4]")
    print()

    if args.check:
        print("  CHECK against the note")
        print(f"     {'lat':>7} {'L (script)':>11} {'L (note)':>9}")
        for lat, Ld in ((38, 1.61), (70.56, 9), (73.94, 13), (80, 32.9)):
            print(f"     {lat:7.2f} {lshell(lat):11.3f} {Ld:9.2f}")
        for Lq, mq in ANCHORS:
            print(f"     L_HP at Mdot={mq:.0e}: script {hp_lshell(mq, A):6.2f}, "
                  f"note {Lq:5.1f}")
        H = scale_height(1.0, 1.0, p["period"], rstar_cm)
        print(f"     H(1 keV, H+): script {H:5.2f} R*, note 4.71 R*"
              f"   <-- differ by {H/4.71:.2f}x, see docstring")
        v = corotation_speed(32.9, rstar_cm, p["period"]) / C_LIGHT
        print(f"     v_corot(32.9 R*): script {v:.4f} c, note 0.0046 c")
        print()

    lats = list(args.latitudes)
    if args.scan:
        lats += list(np.linspace(args.scan[0], args.scan[1], args.n))
    if not lats:
        lats = [38.0, 70.56, 73.94, 80.0]
        print("  (no latitudes given; using the four in the note)\n")

    hdr = (f"  {'lat':>6} {'L':>8} {'Mdot':>10} {'/solar':>9} {'/Jovian':>9} "
           f"{'v_cor/c':>8} {'B(L)':>10} {'rho_max':>10} {'H':>6} {'v_drift':>8}")
    print(hdr)
    print(f"  {'deg':>6} {'R*':>8} {'g/s':>10} {'':>9} {'':>9} {'':>8} "
          f"{'G':>10} {'g/cm3':>10} {'R*':>6} {'km/s':>8}")
    print("  " + "-" * (len(hdr) - 2))
    for lat in sorted(lats):
        r = report(lat, p, A, sigma)
        print(f"  {r['lat']:6.2f} {r['L']:8.2f} {r['mdot']:10.2e} "
              f"{r['mdot_solar']:9.2e} {r['mdot_jup']:9.2e} "
              f"{r['v_corot_c']:8.4f} {r['B']:10.3e} {r['rho']:10.2e} "
              f"{r['H']:6.2f} {r['v_drift_kms']:8.0f}")

    print()
    print("  Mdot is the rate REQUIRED for co-rotation breakdown at that L, i.e.")
    print("  what must be true if the emission there is the main auroral oval.")
    print("  Because L_HP ~ Mdot^-1/4, a low-latitude oval demands a huge wind:")
    print(f"     solar Mdot  = {MDOT_SUN:.0e} g/s     Jovian Mdot ~ {MDOT_JUPITER:.0e} g/s")
    print("  rho_max is an upper limit: exceed it and the Alfven radius falls")
    print("  inside L, so co-rotation cannot be enforced out to that distance.")


if __name__ == "__main__":
    main()
