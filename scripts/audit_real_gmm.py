#!/usr/bin/env python3
"""Phase 4 data audit of the real decoder GMM output (run gmm_era5_32ep_v3, me31).

Reads the converted .npz (see scripts/convert_real_gmm_pt_to_npz.py) with the
research venv (numpy/scipy/matplotlib only — no torch) and prints the audit
numbers quoted in research_notes/phase_4_data_audit.md:

  1. schema vs the 00b expectation, mu_channel/mu_2t alias checks
  2. sanity: weight sums, sigma positivity, NaN/Inf census, value ranges
  3. degeneracy census: near-zero weights / sigmas, near-duplicate components
  4. multimodality census: effective K, pairwise mode separation (2t marginal)
  5. grid geometry: O96 octahedral reduced Gaussian ring-structure check
  6. figures: MAP field, multimodality maps, histograms, ring structure
  7. cross-channel component structure (joint-GMM regime check, feeds review E)
  8. iid-source companion file comparison
  9. feasibility numbers: k-NN edge count, density-underflow headroom

Usage:  .venv/bin/python scripts/audit_real_gmm.py
"""

import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "data"
FIGS = ROOT / "outputs" / "figures"
MAIN_STEM = "phase_4_real_2t"
IID_STEM = "phase_4_real_2t_iidsource"

EARTH_RADIUS_KM = 6371.0


def section(title):
    print(f"\n{'=' * 72}\n## {title}\n{'=' * 72}")


def load(stem):
    arrays = dict(np.load(DATA / f"{stem}.npz"))
    meta = json.loads((DATA / f"{stem}_meta.json").read_text())
    return arrays, meta


def describe(name, a):
    print(
        f"  {name:14s} {str(tuple(a.shape)):16s} {str(a.dtype):8s} "
        f"min={np.nanmin(a):+.4g} max={np.nanmax(a):+.4g}"
    )


def main():
    arrays, meta = load(MAIN_STEM)
    FIGS.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- schema
    section("1. Schema vs expectation (00b: N=25847, K=4, C=72)")
    for k in sorted(arrays):
        describe(k, arrays[k])
    print(f"  meta keys: {sorted(meta)}")
    print(f"  target_datetime={meta['target_datetime']}  run={meta['from_run_id']} "
          f"me={meta['mini_epoch']}  channel={meta['channel_of_interest']} (ch_idx={meta['ch_idx']})")

    pi = arrays["pi"].astype(np.float64)            # [N, K]
    mu_all = arrays["mu"].astype(np.float64)        # [N, K, C]
    sigma_all = arrays["sigma"].astype(np.float64)  # [N, K, C]
    latlons = arrays["latlons"].astype(np.float64)  # [N, 2]
    ch = int(meta["ch_idx"])
    channels = meta["target_channels"]
    n, k_comp = pi.shape
    n_ch = mu_all.shape[2]
    print(f"  N={n}  K={k_comp}  C={n_ch}  (00b expected N=25847 — MISMATCH)" if n != 25847
          else f"  N={n}  K={k_comp}  C={n_ch}")

    same_mu_alias = np.array_equal(arrays["mu_channel"], arrays["mu_2t"])
    same_sg_alias = np.array_equal(arrays["sigma_channel"], arrays["sigma_2t"])
    mu_slice_ok = np.array_equal(arrays["mu_channel"], arrays["mu"][:, :, ch])
    sg_slice_ok = np.array_equal(arrays["sigma_channel"], arrays["sigma"][:, :, ch])
    print(f"  mu_channel == mu_2t (bitwise): {same_mu_alias};  sigma_channel == sigma_2t: {same_sg_alias}")
    print(f"  mu_channel == mu[:,:,{ch}]: {mu_slice_ok};  sigma_channel == sigma[:,:,{ch}]: {sg_slice_ok}")

    mu = arrays["mu_2t"].astype(np.float64)      # [N, K] 2t marginal
    sigma = arrays["sigma_2t"].astype(np.float64)

    # ---------------------------------------------------------------- sanity
    section("2. Sanity checks")
    for name, a in arrays.items():
        bad = int(np.sum(~np.isfinite(a)))
        if bad:
            print(f"  !! {name}: {bad} non-finite entries")
    print("  non-finite census: clean unless flagged above")
    sums = pi.sum(axis=1)
    print(f"  pi row sums: max|sum-1| = {np.abs(sums - 1).max():.3e};  min(pi) = {pi.min():.3e}")
    print(f"  sigma (2t): min={sigma.min():.4e}  p1={np.percentile(sigma, 1):.4f}  "
          f"median={np.median(sigma):.4f}  p99={np.percentile(sigma, 99):.4f}  max={sigma.max():.4f}")
    print(f"  sigma (all 72 ch): min={sigma_all.min():.4e}  median={np.median(sigma_all):.4f}  "
          f"max={sigma_all.max():.4f}  (count < 1e-3: {int((sigma_all < 1e-3).sum())})")
    print(f"  mu (2t): min={mu.min():+.3f}  max={mu.max():+.3f}  "
          f"(standardised units; plausible z-score range)")
    print(f"  latlons: lat [{latlons[:, 0].min():.3f}, {latlons[:, 0].max():.3f}]  "
          f"lon [{latlons[:, 1].min():.3f}, {latlons[:, 1].max():.3f}]")
    max_w = pi.max(axis=1)
    print(f"  pi softness (max weight per cell): median={np.median(max_w):.3f}  "
          f"p95={np.percentile(max_w, 95):.3f}  max={max_w.max():.3f}  "
          f"(phase_1.md calibration target: median≈0.35, p95≈0.41, max≈0.49 on e3fz467m)")

    # ------------------------------------------------------------ degeneracy
    section("3. Degeneracy census (2t marginal)")
    near_zero_w = pi < 0.01
    print(f"  near-zero weights (pi<0.01): {int(near_zero_w.sum())} slots "
          f"({near_zero_w.mean():.2%});  cells with >=1: {np.any(near_zero_w, axis=1).mean():.2%}")
    for thr in (1e-3, 1e-2):
        cnt = int((sigma < thr).sum())
        print(f"  near-zero sigma_2t (<{thr:g}): {cnt} slots "
              f"({cnt / sigma.size:.3%});  cells with >=1: {np.any(sigma < thr, axis=1).mean():.3%}")

    pair_idx = list(combinations(range(k_comp), 2))
    pi_a = np.stack([pi[:, i] for i, _ in pair_idx], axis=1)
    pi_b = np.stack([pi[:, j] for _, j in pair_idx], axis=1)
    d_mu = np.stack([np.abs(mu[:, i] - mu[:, j]) for i, j in pair_idx], axis=1)   # [N, P]
    s_max = np.stack([np.maximum(sigma[:, i], sigma[:, j]) for i, j in pair_idx], axis=1)
    sep = d_mu / s_max                                                            # [N, P]
    min_sep = sep.min(axis=1)
    dup = sep < 0.1
    print(f"  near-duplicate pairs (|mu_i-mu_j| < 0.1*max(sig)): {int(dup.sum())} pairs "
          f"({dup.mean():.2%});  cells with >=1 duplicate pair: {np.any(dup, axis=1).mean():.2%}")
    active_dup = dup & (pi_a >= 0.01) & (pi_b >= 0.01)
    print(f"  ... restricted to both weights >= 0.01: cells affected "
          f"{np.any(active_dup, axis=1).mean():.2%}")

    # --------------------------------------------------------- multimodality
    section("4. Multimodality census (2t marginal)")
    p_safe = np.clip(pi, 1e-12, 1.0)
    k_eff = np.exp(-(p_safe * np.log(p_safe)).sum(axis=1))
    print(f"  effective K (weight entropy): median={np.median(k_eff):.3f}  "
          f"p5={np.percentile(k_eff, 5):.3f}  p95={np.percentile(k_eff, 95):.3f}")
    print(f"  fraction K_eff > 1.5: {np.mean(k_eff > 1.5):.2%}")
    print(f"  fraction K_eff > 3.0: {np.mean(k_eff > 3.0):.2%}")

    max_sep = sep.max(axis=1)
    print(f"  min pairwise separation: median={np.median(min_sep):.3f}  "
          f"frac > 2: {np.mean(min_sep > 2):.3%}  frac > 1: {np.mean(min_sep > 1):.3%}")
    print(f"  max pairwise separation: median={np.median(max_sep):.3f}  "
          f"frac > 2: {np.mean(max_sep > 2):.3%}  frac > 1: {np.mean(max_sep > 1):.3%}")

    active_pair = (pi_a >= 0.1) & (pi_b >= 0.1)
    sep_active = np.where(active_pair, sep, 0.0)
    sep_active_max = sep_active.max(axis=1)
    bimodal = sep_active_max > 2.0
    bimodal_1 = sep_active_max > 1.0
    print(f"  practically bimodal (a pair with both pi>=0.1 separated > 2 sigma): "
          f"{bimodal.mean():.3%} of cells ({int(bimodal.sum())})")
    print(f"  ... separated > 1 sigma: {bimodal_1.mean():.3%} of cells ({int(bimodal_1.sum())})")
    lat_band = np.abs(latlons[bimodal, 0]) if bimodal.any() else np.array([np.nan])
    print(f"  |lat| of bimodal cells: median={np.median(lat_band):.1f} deg  "
          f"p10={np.percentile(lat_band, 10):.1f}  p90={np.percentile(lat_band, 90):.1f}")

    # -------------------------------------------------------- grid geometry
    section("5. Grid geometry — O96 octahedral hypothesis")
    lats = latlons[:, 0]
    lons = latlons[:, 1]
    ring_lats, ring_counts = np.unique(np.round(lats, 4), return_counts=True)
    ring_lats = ring_lats[::-1]          # north to south
    ring_counts = ring_counts[::-1]
    n_rings = len(ring_lats)
    print(f"  unique latitude rings: {n_rings} (O96 expects 192)")
    half = n_rings // 2
    expected = 20 + 4 * np.arange(half)  # octahedral: 20, 24, ... toward equator
    octa_north = bool(np.array_equal(ring_counts[:half], expected))
    octa_sym = bool(np.array_equal(ring_counts, ring_counts[::-1]))
    print(f"  ring counts (north pole down): first 6 {ring_counts[:6].tolist()}  "
          f"equator pair {ring_counts[half - 1:half + 1].tolist()}  last {ring_counts[-1]}")
    print(f"  octahedral pattern 20+4i toward equator: {octa_north};  N-S symmetric: {octa_sym}")
    print(f"  total points: {ring_counts.sum()} (= 4*96*105 = 40320: {ring_counts.sum() == 40320})")
    print(f"  first Gaussian latitude: {ring_lats[0]:.4f} deg (O96 reference ~ 89.284)")
    # longitude regularity on the polar and equator rings
    for tag, ridx in (("polar", 0), ("equator", half - 1)):
        sel = np.isclose(np.round(lats, 4), ring_lats[ridx])
        ring_lons = np.sort(lons[sel])
        dlons = np.diff(ring_lons)
        print(f"  {tag} ring: n={sel.sum()}, lon spacing {np.median(dlons):.4f} deg "
              f"(max dev {np.abs(dlons - np.median(dlons)).max():.2e}), first lon {ring_lons[0]:.4f}")
    # start longitude of EVERY ring (establishes the "-180 start" claim for all 192)
    lats_r = np.round(lats, 4)
    first_lons = np.array([lons[lats_r == rl].min() for rl in ring_lats])
    print(f"  all {n_rings} rings: first lon in [{first_lons.min():.4f}, {first_lons.max():.4f}] "
          f"(all == -180: {bool(np.allclose(first_lons, -180.0))})")

    # neighbour spacing for graph feasibility
    xyz = np.stack(
        [
            np.cos(np.radians(lats)) * np.cos(np.radians(lons)),
            np.cos(np.radians(lats)) * np.sin(np.radians(lons)),
            np.sin(np.radians(lats)),
        ],
        axis=1,
    )
    tree = cKDTree(xyz)
    dist, idx = tree.query(xyz, k=9)  # self + 8 neighbours
    chord = dist[:, 1:]
    arc_km = 2 * np.arcsin(np.clip(chord / 2, 0, 1)) * EARTH_RADIUS_KM
    edges = set()
    for i in range(n):
        for j in idx[i, 1:]:
            edges.add((i, j) if i < j else (j, i))
    print(f"  k=8 NN graph: |E| = {len(edges)} undirected edges "
          f"(union-symmetrised from {n * 8} directed; mutual-intersection would be smaller)")
    print(f"  1st-neighbour distance km: min={arc_km[:, 0].min():.1f}  "
          f"median={np.median(arc_km[:, 0]):.1f}  max={arc_km[:, 0].max():.1f}")
    print(f"  8th-neighbour distance km: median={np.median(arc_km[:, -1]):.1f}  "
          f"max={arc_km[:, -1].max():.1f}")

    # ------------------------------------------------------------- figures
    section("6. Figures")
    comp_density = pi / sigma                       # highest component-centre density
    mode_idx = np.argmax(comp_density, axis=1)
    map_field = mu[np.arange(n), mode_idx]
    mean_field = (pi * mu).sum(axis=1)

    lons_r = np.radians(lons)
    lats_r = np.radians(lats)

    def moll(ax, values, title, cmap="RdBu_r", vmin=None, vmax=None):
        sc = ax.scatter(lons_r, lats_r, c=values, s=0.5, cmap=cmap,
                        vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(title, fontsize=10)
        ax.grid(True, lw=0.3, alpha=0.4)
        return sc

    vmin, vmax = np.percentile(np.concatenate([map_field, mean_field]), [2, 98])
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), subplot_kw={"projection": "mollweide"})
    sc = moll(axes[0], map_field, "Per-cell MAP field (argmax-component mean, 2t standardised)",
              vmin=vmin, vmax=vmax)
    fig.colorbar(sc, ax=axes[0], orientation="horizontal", pad=0.05, shrink=0.8)
    sc = moll(axes[1], mean_field, "Mixture-mean field (over-smooth anchor)", vmin=vmin, vmax=vmax)
    fig.colorbar(sc, ax=axes[1], orientation="horizontal", pad=0.05, shrink=0.8)
    fig.suptitle(f"2t GMM fields — {meta['from_run_id']} me{meta['mini_epoch']} "
                 f"@ {meta['target_datetime']}", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "phase_4_audit_map_field.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), subplot_kw={"projection": "mollweide"})
    sc = moll(axes[0], k_eff, "Effective K (weight entropy)", cmap="viridis", vmin=1, vmax=4)
    fig.colorbar(sc, ax=axes[0], shrink=0.7)
    sc = moll(axes[1], np.clip(sep_active_max, 0, 4),
              "Max pairwise mode separation (pairs with both pi>=0.1), clipped at 4",
              cmap="magma", vmin=0, vmax=4)
    fig.colorbar(sc, ax=axes[1], shrink=0.7)
    sc = moll(axes[2], bimodal.astype(float), "Practically bimodal cells (sep > 2)",
              cmap="Reds", vmin=0, vmax=1)
    fig.colorbar(sc, ax=axes[2], shrink=0.7)
    fig.tight_layout()
    fig.savefig(FIGS / "phase_4_audit_multimodality_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(max_w, bins=80, color="steelblue")
    axes[0, 0].set_title("max mixture weight per cell")
    axes[0, 1].hist(k_eff, bins=80, color="seagreen")
    axes[0, 1].axvline(1.5, color="k", ls="--", lw=1)
    axes[0, 1].set_title("effective K (exp weight entropy)")
    axes[1, 0].hist(sep_active_max, bins=100, color="indianred", log=True)
    axes[1, 0].axvline(2.0, color="k", ls="--", lw=1)
    axes[1, 0].set_title("max active-pair separation (log count)")
    axes[1, 1].hist(np.log10(sigma.reshape(-1)), bins=100, color="goldenrod")
    axes[1, 1].set_title("log10 sigma_2t (all components)")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle("Phase 4 audit — 2t marginal censuses")
    fig.tight_layout()
    fig.savefig(FIGS / "phase_4_audit_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ring_counts, ".", ms=3, label="observed")
    axes[0].plot(np.concatenate([expected, expected[::-1]]), lw=1, alpha=0.7,
                 label="octahedral 20+4i")
    axes[0].set_xlabel("ring index (N to S)")
    axes[0].set_ylabel("points per ring")
    axes[0].legend()
    axes[0].set_title(f"Ring structure ({n_rings} rings)")
    axes[1].plot(ring_lats, ".", ms=3)
    axes[1].set_xlabel("ring index (N to S)")
    axes[1].set_ylabel("ring latitude (deg)")
    axes[1].set_title("Ring latitudes (Gaussian)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "phase_4_audit_grid_rings.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote 4 figures to {FIGS}/phase_4_audit_*.png")

    # ------------------------------------------- cross-channel structure (E)
    section("7. Cross-channel component structure (joint GMM regime check)")
    if bimodal.any():
        sel = np.flatnonzero(bimodal)
        best_pair = np.argmax(sep_active, axis=1)[sel]            # index into pair_idx
        pa = np.array([pair_idx[p][0] for p in best_pair])
        pb = np.array([pair_idx[p][1] for p in best_pair])
        probe = {"10u": 0, "msl": 4, "skt": 18, "t_850": channels.index("t_850"),
                 "z_500": channels.index("z_500")}
        print(f"  for the {sel.size} bimodal cells, separation of the SAME component pair")
        print(f"  in other channels (|d_mu|/max(sigma)):")
        for name, c in probe.items():
            d = np.abs(mu_all[sel, pa, c] - mu_all[sel, pb, c])
            s = np.maximum(sigma_all[sel, pa, c], sigma_all[sel, pb, c])
            sep_c = d / s
            print(f"    {name:6s} (ch {c:2d}): median={np.median(sep_c):.2f}  "
                  f"frac>1: {np.mean(sep_c > 1):.2%}  frac>2: {np.mean(sep_c > 2):.2%}")
        # channel-averaged separation of the 2t-selected pair (profile distinctness)
        d_all = np.abs(mu_all[sel, pa, :] - mu_all[sel, pb, :])
        s_all = np.maximum(sigma_all[sel, pa, :], sigma_all[sel, pb, :])
        mean_sep_all = (d_all / s_all).mean(axis=1)
        print(f"  channel-averaged separation of that pair over all 72 channels: "
              f"median={np.median(mean_sep_all):.2f}  p90={np.percentile(mean_sep_all, 90):.2f}")
    else:
        print("  no practically bimodal cells — cross-channel regime check moot")

    rng = np.random.default_rng(0)
    sub = rng.choice(n, size=min(4000, n), replace=False)
    d_prof = np.abs(mu_all[sub][:, :, None, :] - mu_all[sub][:, None, :, :])     # [n,K,K,C]
    s_prof = np.maximum(sigma_all[sub][:, :, None, :], sigma_all[sub][:, None, :, :])
    z_prof = (d_prof / s_prof).mean(axis=-1)                                     # [n,K,K]
    iu = np.triu_indices(k_comp, k=1)
    z_pairs = z_prof[:, iu[0], iu[1]]
    print(f"  ALL cells (n={sub.size} sample), channel-averaged pairwise component separation: "
          f"median={np.median(z_pairs):.2f}  p90={np.percentile(z_pairs, 90):.2f}  "
          f"frac of pairs >0.5: {np.mean(z_pairs > 0.5):.2%}")

    # ---------------------------------------------------- iid-source compare
    section("8. iid-source companion comparison")
    arrays_iid, _ = load(IID_STEM)
    for key in ("pi", "mu_2t", "sigma_2t"):
        a = arrays[key].astype(np.float64)
        b = arrays_iid[key].astype(np.float64)
        identical = np.array_equal(a, b)
        max_d = np.abs(a - b).max()
        corr = np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1]
        print(f"  {key:9s} bitwise-identical={identical}  max|diff|={max_d:.4g}  corr={corr:.4f}")
    pi_iid = arrays_iid["pi"].astype(np.float64)
    p_safe_iid = np.clip(pi_iid, 1e-12, 1.0)
    k_eff_iid = np.exp(-(p_safe_iid * np.log(p_safe_iid)).sum(axis=1))
    print(f"  iidsource: frac K_eff > 1.5 = {np.mean(k_eff_iid > 1.5):.2%} "
          f"(main: {np.mean(k_eff > 1.5):.2%})")

    # -------------------------------------------------- numerics feasibility
    section("9. Numerics / feasibility probes")
    z_mm = np.min(np.abs(mean_field[:, None] - mu) / sigma, axis=1)
    print(f"  mixture-mean field, z to nearest component: median={np.median(z_mm):.2f}  "
          f"max={z_mm.max():.2f}  (normal pdf exp(-z^2/2) underflows ~ z>37.6 in "
          f"float64, ~ z>13.2 in float32 (14.4 with subnormals) — linear-space NLL "
          f"code at risk if any field strays)")
    dens_mm = np.sum(pi * np.exp(-0.5 * ((mean_field[:, None] - mu) / sigma) ** 2)
                     / (sigma * np.sqrt(2 * np.pi)), axis=1)
    print(f"  mixture density at mixture mean: min={dens_mm.min():.3e} "
          f"(zeros: {int((dens_mm == 0).sum())})")
    print(f"  ICM cost scale: N={n}, k=8 edges={len(edges)}; python-loop ICM sweep "
          f"touches {n} cells x ~8 neighbours; Joint MAP grad eval = O(N*K + |E|) vectorised")


if __name__ == "__main__":
    main()
