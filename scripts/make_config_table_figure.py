"""App A training-configuration table, rendered as an image (ruling 7, 2 Jul PM).

The two-regime training configuration for the mixture-emitting models ships as
a figure (outputs/figures/app_a_training_configs.png) so the automated word
count excludes it; the counting caveat is flagged on the PROTOCOL §6 checklist.
Every constant below is a config constant, verified against the pinned config
and notes (NOT results -- no macros involved):

  shared recipe   ~/WeatherGenerator/config/gmm_forecast_config.yml:101 (freeze),
                  163--193 (data window / cadence / AdamW / LR schedule),
                  244--245 (validation window); K=4 default at
                  ~/WeatherGenerator/src/weathergen/model/model.py:434; GMM head
                  per-stream at config/streams/era5_1deg/era5.yml:37
  forecast deltas gmm_forecast_config.yml:10--24 (header) + 209 (masking),
                  232--234 (forecast block), load_chkpt block
  run lengths     research_notes/log.md:262 (AE 32 mini-epochs, me31 kept),
                  log.md:199 (forecast 6 + 8 across two runs)

    .venv/bin/python scripts/make_config_table_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "outputs" / "figures" / "app_a_training_configs.png"

SHARED_ROWS = [
    ("Trainable weights", "whole model, nothing frozen"),
    ("Loss", "Gaussian-mixture negative log-likelihood on masked tokens"),
    ("Mixture components", "$K = 4$ weighted components per location"),
    ("Training window", "ERA5 1979–2022, 6-hour cadence"),
    ("Validation window", "October–December 2023"),
    ("Optimiser", r"AdamW ($\beta_1{=}0.975$, $\beta_2{=}0.9875$, $\epsilon{=}2\times10^{-8}$)"),
    ("Weight decay / grad. clip", "0.1 / 0.1"),
    ("Learning rate", r"$10^{-6}\rightarrow10^{-5}$; 512-step cosine warmup,"
                      "\n512-step linear cooldown"),
]

REGIME_HEADER = ("", "Autoencoder", "Forecast model")
REGIME_ROWS = [
    ("Initialisation", "from scratch", "warm start from the\nautoencoder checkpoint"),
    ("Masking", "random, rate 0.6", "forecast (full current\nfield as source)"),
    ("Forecast steps", "none (step-0 reconstruction)", "eight 6-hour steps,\nto +48 hours"),
    ("Mini-epochs", "32 (checkpoint 31 used)", "6 + 8 across two runs\n(run record in text)"),
]


def _style(table, header_rows=(), fontsize=9.5):
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.75")
        cell.set_linewidth(0.6)
        cell.PAD = 0.03
        if row in header_rows:
            cell.set_facecolor("0.92")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("0.975")
        if col == 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"
        else:
            cell._loc = "left"


def main():
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.4, 4.9), height_ratios=[9.2, 6.8],
    )
    for ax in (ax_top, ax_bot):
        ax.axis("off")

    ax_top.set_title("Shared training recipe (both regimes)", fontsize=10.5,
                     weight="bold", loc="left", pad=4)
    t1 = ax_top.table(
        cellText=[[k, v] for k, v in SHARED_ROWS],
        colWidths=[0.30, 0.70], cellLoc="left", loc="upper center",
    )
    _style(t1)
    for row, (_, v) in enumerate(SHARED_ROWS):
        t1[row, 1].set_height(t1[row, 1].get_height() * (1.9 if "\n" in v else 1.25))
        t1[row, 0].set_height(t1[row, 1].get_height())

    ax_bot.set_title("Per-regime settings", fontsize=10.5, weight="bold",
                     loc="left", pad=4)
    t2 = ax_bot.table(
        cellText=[list(REGIME_HEADER)] + [list(r) for r in REGIME_ROWS],
        colWidths=[0.22, 0.39, 0.39], cellLoc="left", loc="upper center",
    )
    _style(t2, header_rows=(0,))
    for row in range(1, len(REGIME_ROWS) + 1):
        tall = any("\n" in c for c in REGIME_ROWS[row - 1])
        for col in range(3):
            t2[row, col].set_height(t2[row, col].get_height() * (1.9 if tall else 1.25))

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
