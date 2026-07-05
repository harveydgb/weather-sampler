# Research notes: sampling coherent fields from independent per-location Gaussian mixtures

These notes summarise the reasoning behind the project — why the problem exists, how the
approach was built up, why the evaluation looks the way it does, and which papers shaped the
choices. Exact numbers, tables, and figures live in the report; nothing here supersedes it.

## 1. Problem and motivation

The WeatherGenerator decoder branch studied here attaches a Gaussian-mixture head to every
target location: each grid point receives mixture weights, means, and scales for its own
marginal distribution, trained with a pointwise negative log-likelihood. Nothing in the head or
the loss specifies a joint distribution across locations — spatial covariance is discarded by
construction, for computational reasons. Drawing one value independently from each local
mixture therefore produces salt-and-pepper fields that no atmospheric state resembles, even
though every individual value is locally plausible. Yet most uses of a weather field are
spatial: catchment rainfall, wind-farm aggregation, and event extent all depend on joint
structure, which is exactly what independent draws destroy. The question the project asks is
narrow and post hoc: given only the emitted per-location mixtures, can we select a single
spatially coherent field that remains faithful to every local marginal? A complication that
shapes everything downstream is that component labels are not aligned across space — component
k at one cell has no relationship to component k at its neighbour — so any coupling must act on
values, never on label indices.

## 2. Approach

The samplers were developed on a controlled toy before touching real output: an 8×8 grid of
synthetic K=4 mixtures whose means come from smooth overlapping surfaces, deliberately
scrambled per cell so that neither label smoothness nor a hidden dominance field can be
exploited, with mixture weights drawn independently at each cell. The toy has no ground-truth
field; the only "truth" is the emitted mixture, which mirrors the real setting.

Simple baselines bracket the problem: the independent draw (the noisy failure mode), the
per-cell MAP (jagged but maximally faithful), the mixture mean (smooth but often sitting in
low-density valleys between modes), and — critically — the smoothed MAP, a graph-Laplacian blur
of the per-cell MAP, which is the obvious method a skeptic would try first and which any
proposed sampler must out-argue.

Two samplers were carried forward. The first, the Joint MAP, minimises the summed mixture
negative log-likelihood plus a λ-weighted graph-Laplacian smoothness penalty; λ is treated as a
declared coherence prior, never tuned to a Pareto knee. The second, the Mode-selection MRF,
first extracts each cell's mixture modes by mean-shift and then selects one mode per cell by
minimising a unary-likelihood plus value-space pairwise energy, so the field is smooth in the
selected values while every value stays on a genuine local mode. The toy comparison showed the
two are complementary: the Joint MAP reaches lower roughness but pays for it by smearing values
into the low-density gaps between modes, while the mode-selection route cannot smear at all but
is capped in how smooth it can get. Two ablation branches were opened and closed. Swapping the
quadratic penalty for total variation, checked against an exact min-cut solution of the
discretised objective, showed the smearing is a property of the continuous objective rather
than the optimiser, and that the TV route collapses to a constant field before it reaches the
Joint MAP's operating range. An annealed-Langevin search over the same objective found no
better basins than the multi-restart Adam optimiser, validating the Joint MAP's optimisation
without changing any result.

## 3. Evaluation reasoning

Because the decoder discarded spatial covariance, no quantity computable from the mixtures can
say what the "correct" coherence level is — coherence is asserted as a prior, not recovered.
The evaluation is built around that honesty. Marginal faithfulness is measured as
per-cell negative log-likelihood, with a per-cell excess-NLL-to-nearest-mode tail as the
sharper diagnostic, since boundary smearing is invisible in an averaged score. Roughness is
reported scale-free (edge-averaged squared differences normalised by field variance), because
the raw penalty can be gamed by shrinking the field toward its mean. Comparisons against the
smoothed-MAP baseline are made at matched roughness: λ is selected by a declared rule — the
sweep value whose roughness matches the blur's — so the two methods are compared at the same
asserted coherence rather than at cherry-picked operating points. Scores are stratified by
latitude band, by whether a cell is practically bimodal (well-separated modes with
non-negligible weights), and by effective component count, because the globally averaged score
dilutes the small multimodal subset where the methods actually differ. Finally, marginal
faithfulness of the stochastic baseline is checked directly: the independent draw should
reproduce the emitted marginals essentially exactly (a do-no-harm check via PIT and CRPS),
while the deterministic fields are reported as marginal position, not calibration. A native
spherical power spectrum brackets the fields between the independent-draw white floor and the
over-smoothed floor, with ERA5 as a direction-of-realism reference only, never a target.

## 4. Key findings

Two results carry the report. First, the mixtures themselves change character with forecast
lead. The canonical reconstruction-regime artifact is near-one-hot — the expected signature of
the lowest-uncertainty task — but in the forecast regime the weights soften monotonically from
+6 h to +48 h: the one-hot fraction falls steeply, second-mode mass grows, and the share of
cells with a ≥1σ-separated second mode rises substantially, with further training deepening
rather than reversing the effect. This is what makes a mode-aware sampler worth having at all.
Second, at matched roughness the Joint MAP beats the smoothed-MAP baseline on both likelihood
and the off-mode smear tail, at every lead and in both regimes, and the advantage is stable
across twelve forecast initialisations spanning all four seasons of the held-out year. The
residual failure modes differ in kind: the blur smears broadly and shallowly, while the Joint
MAP's rare deep excursions concentrate on the bimodal cells — the real-data echo of the
structural quadratic-penalty smearing the toy predicted. See the report (Chapters 4–5 and the
associated figures) for the exact tables.

## 5. Related work

The framing owes most to the statistical post-processing literature, which named this problem
long before ML weather models did: calibrated marginals plus a separately imposed dependence
structure is exactly the Schaake shuffle and ensemble copula coupling setting (Clark et al.
2004; Schefzik, Thorarinsdottir and Gneiting 2013), and those methods explain why a dependence
template must come from somewhere outside the marginals. Among ML weather models, AtmoRep
(Lessig et al. 2023) shows the alternative of building stochasticity into the model itself, and
its use of spectra as a realism diagnostic motivated the spectral bracket here; Aurora (Bodnar
et al. 2024) anchors the deterministic foundation-model contrast, and GenCast (Price et al.
2024) the generative-ensemble one. The treatment of mixture modes rests on the Q-FAT
mode-sampling idea (Sheebaelhamd et al. 2025) — components are modes worth preserving, and
variance-scaling tricks can destroy them — together with the classical mode-finding results of
Carreira-Perpiñán (2000) and Carreira-Perpiñán and Williams (2003). The exact TV certificate
uses Ishikawa's (2003) construction for MRFs with convex priors, with the TV penalty itself
from Rudin, Osher and Fatemi (1992); optimisation is Adam (Kingma and Ba 2015). The
faithfulness checks draw on the proper-scoring literature: Gneiting and Raftery (2007) for
scoring rules, Grimit et al. (2006) for the closed-form mixture CRPS, and Gneiting, Balabdaoui
and Raftery (2007) for calibration via PIT. The data context is ERA5 (Hersbach et al. 2020)
through the WeatherGenerator project's ERA5-trained GMM branch.
