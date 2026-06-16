"""Plotting helpers for the existing research notebooks."""

import numpy as np


def plot_field_pair(
    field_a,
    field_b,
    titles,
    *,
    cmap: str = "viridis",
):
    import matplotlib.pyplot as plt

    vmin = min(np.nanmin(field_a), np.nanmin(field_b))
    vmax = max(np.nanmax(field_a), np.nanmax(field_b))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, field, title in zip(axes, [field_a, field_b], titles):
        im = ax.imshow(field, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig, axes


def plot_component_fields(component_fields, peak_offsets, extent):
    import matplotlib.pyplot as plt

    k = component_fields.shape[-1]
    vmin, vmax = component_fields.min(), component_fields.max()
    fig, axes = plt.subplots(1, k, figsize=(3 * k, 3.4))
    for idx, ax in enumerate(np.ravel(axes)):
        im = ax.imshow(
            component_fields[:, :, idx],
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        a_k, b_k = peak_offsets[idx]
        ax.plot(-a_k, -b_k, "r+", markersize=12, markeredgewidth=2)
        ax.set_title(f"component {idx}\n(a, b) = ({a_k:g}, {b_k:g})")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.suptitle("Four component fields (peak location varied), peak marked +", y=1.02)
    return fig, axes


def plot_mu_slices(mu):
    import matplotlib.pyplot as plt

    k = mu.shape[-1]
    vmin, vmax = mu.min(), mu.max()
    fig, axes = plt.subplots(1, k, figsize=(3 * k, 3.2))
    for idx, ax in enumerate(np.ravel(axes)):
        im = ax.imshow(mu[:, :, idx], origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"mu[:, :, {idx}]")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025)
    return fig, axes


# Shared field -> (colour, marker, linestyle) map so the variogram (real space)
# and the spectrum (spectral space) read as the same diagnostic two ways: the iid
# white floor and the mixture-mean over-smooth field are the bracket; the sampler
# siblings sit between; the ERA5 reference is a thin grey dashed line, drawn last
# so it reads as a reference and not a series (direction-of-realism, NOT a target).
_FIELD_STYLE = {
    "iid_seed0": ("tab:red", "o", "-"),          # over-noisy white floor (upper)
    "mixture_mean": ("tab:purple", "v", "-"),    # over-smooth extreme (lower)
    "mode_map": ("0.45", "s", "--"),
    "smoothed_map_n10": ("tab:orange", "^", "--"),
    "m1_star": ("tab:blue", "D", "-"),
    "m4_beta1": ("tab:green", "P", "--"),
}
ERA5_KEY = "era5"
_ERA5_STYLE = ("0.35", None, ":")
_FALLBACK_CYCLE = [
    ("tab:cyan", "o", "-"), ("tab:brown", "s", "--"),
    ("tab:olive", "^", "-"), ("tab:pink", "d", "--"),
]


def field_style(name, idx=0):
    """(colour, marker, linestyle) for a field, stable across variogram/spectrum."""
    if name == ERA5_KEY:
        return _ERA5_STYLE
    if name in _FIELD_STYLE:
        return _FIELD_STYLE[name]
    return _FALLBACK_CYCLE[idx % len(_FALLBACK_CYCLE)]


def _ordered_items(series):
    """Series in a stable order with the ERA5 reference drawn LAST (on top)."""
    keys = [k for k in series if k != ERA5_KEY]
    if ERA5_KEY in series:
        keys.append(ERA5_KEY)
    return [(k, series[k]) for k in keys]


def plot_variograms(
    centres,
    variograms,
    *,
    xlabel: str,
    title: str,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for idx, (label, values) in enumerate(_ordered_items(variograms)):
        colour, marker, ls = field_style(label, idx)
        if label == ERA5_KEY:
            ax.plot(centres, values, ls, color=colour, lw=1.0, label="ERA5 (reference)")
        else:
            ax.plot(centres, values, marker=marker, ls=ls, color=colour,
                    ms=3.5, lw=1.5, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("semivariance")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, ax


def plot_spectra(ell, spectra, *, title, era5_key=ERA5_KEY):
    """Log-log angular power spectrum C_l vs degree l for the headline fields.

    RUNG-3 BRACKET DIAGNOSTIC (not a skill metric, not an optimisation target):
    the iid draw is the over-noisy white floor (upper at high l), the mixture-mean
    field is the over-smooth extreme (lower), the sampler siblings sit between and
    below iid at high l. The ERA5 series (if present) is drawn as a THIN grey
    dashed reference line LAST -- direction-of-realism only, never a target. There
    is no GMM-derivable target spectrum (large_notes open problem). Returns
    ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for idx, (label, values) in enumerate(_ordered_items(spectra)):
        colour, marker, ls = field_style(label, idx)
        if label == era5_key:
            ax.plot(ell, values, ls, color=colour, lw=1.0, label="ERA5 (reference)")
        else:
            ax.plot(ell, values, marker=marker, ls=ls, color=colour,
                    ms=2.8, lw=1.4, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("angular degree $\\ell$")
    ax.set_ylabel("$C_\\ell$ (resolved band)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, ax


def plot_single_point_marginal(
    samples,
    x,
    total_pdf,
    component_pdfs,
    pi,
    *,
    point_index: int,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(samples, bins=100, density=True, alpha=0.55, color="steelblue", label=f"{len(samples):,} samples")
    ax.plot(x, total_pdf, "r-", lw=2, label="GMM PDF")
    for idx, component_pdf in enumerate(component_pdfs):
        ax.plot(x, component_pdf, "--", lw=1, alpha=0.55, label=f"comp {idx} (pi={pi[idx]:.2f})")
    ax.set_xlabel("2t (normalised)")
    ax.set_ylabel("density")
    ax.set_title(f"Single-point marginal (point {point_index})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, ax


def plot_pooled_distribution(sample_iid, mean_field):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(sample_iid, bins=120, density=True, alpha=0.5, color="indianred", label="IID samples")
    ax.hist(mean_field, bins=120, density=True, alpha=0.5, color="steelblue", label="Mean field")
    ax.set_xlabel("2t (normalised)")
    ax.set_ylabel("density")
    ax.set_title("Pooled distribution across all grid points")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_mollweide_fields(
    latlons_deg,
    fields,
    *,
    suptitle: str,
    cmap: str = "RdBu_r",
):
    import matplotlib.pyplot as plt

    latlons = np.asarray(latlons_deg, dtype=float)
    lats_rad = np.radians(latlons[:, 0])
    lons_rad = np.radians(latlons[:, 1])
    values_for_scale = np.concatenate([np.asarray(values).reshape(-1) for values in fields.values()])
    vmin, vmax = np.percentile(values_for_scale, [2, 98])

    fig, axes = plt.subplots(
        1,
        len(fields),
        figsize=(8 * len(fields), 5),
        subplot_kw={"projection": "mollweide"},
    )
    axes = np.atleast_1d(axes)
    for ax, (title, values) in zip(axes, fields.items()):
        sc = ax.scatter(
            lons_rad,
            lats_rad,
            c=np.asarray(values).reshape(-1),
            s=0.5,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.set_title(title, pad=10)
        ax.grid(True, lw=0.3, alpha=0.4)
        fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, label="2t (normalised)", shrink=0.8)
    fig.suptitle(suptitle, y=1.02, fontsize=13)
    fig.tight_layout()
    return fig, axes
