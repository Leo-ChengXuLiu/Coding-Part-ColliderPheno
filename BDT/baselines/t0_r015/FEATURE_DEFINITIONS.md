# Formal T0 ordered reconstructed features

Definitions below follow the formal R015 producer, including its denominator
guards. Angles are three-dimensional opening angles in radians, not rapidity-phi
distances. Every constituent of each jet is used; no leading-constituent cap,
grooming, truth selection, or track-only substructure selection is applied.

## Objects, roles, and notation

Input objects are the complete persisted Delphes `EFlowTrack`, `EFlowPhoton`, and
`EFlowNeutralHadron` collections before isolation removal. Tracks use
`px=PT*cos(phi)`, `py=PT*sin(phi)`, `pz=PT*sinh(eta)` and
`E=sqrt(px²+py²+pz²+m_pi²)` with `m_pi=0.13957039 GeV` for every track.
Photons and neutral hadrons use `ET` in place of `PT` and `E=ET*cosh(eta)`
(massless convention). These are reconstructed object types, not truth species.

FastJet 3.4.0 inclusive `ee_genkt`, `p=1`, R=0.15, E-scheme clusters the complete
object set. Jets are sorted descending by `(E, px, py, pz)` and assigned sequential
indices. Accept jets only if `E>500 GeV`; select events only if exactly two are
accepted and `Mjj>8000 GeV`. HJ has larger mass; an exact mass tie assigns HJ to
the smaller jet index. There is no reconstructed-jet eta cut. All formulas below
use the HJ or LJ constituents after this role assignment.

Let `E_J` and `p_J` be the E-scheme jet energy and momentum, `E_sum=sum_i E_i`,
and `theta(a,b)=acos(clip(p_a dot p_b / (|p_a||p_b|), -1, 1))`.
Substructure code defines this angle as zero when either momentum norm is zero;
the girth code instead treats such an angle as undefined. This difference is
preserved below. No feature is pre-imputed: undefined values become native NaN.

## Exact order, units, and missing values

| # | Input | Definition | Units | Undefined/missing rule |
|---|---|---|---|---|
| 1 | `M_HJ` | HJ `max(0, FastJet m())`; for timelike jets, `sqrt(E_J²-|p_J|²)` | GeV | No producer missing sentinel; negative signed mass clamped to zero |
| 2 | `M_LJ` | Same mass prescription for LJ | GeV | Same as HJ |
| 3 | `D2_LJ` | LJ `e3/e2³`, all constituents, beta=1 (below) | dimensionless | NaN for fewer than two constituents, `E_sum<=0`, or `e2==0` |
| 4 | `E_HJ` | HJ E-scheme energy `sum_i E_i` | GeV | No producer missing sentinel for selected jets |
| 5 | `E_LJ` | LJ E-scheme energy `sum_i E_i` | GeV | Same as HJ |
| 6 | `abs_cos_theta_HJ` | HJ `abs(pz_J/|p_J|)` relative to beam axis | dimensionless | NaN if `|p_J|==0` |
| 7 | `abs_cos_theta_LJ` | Same polar-angle formula for LJ | dimensionless | Same as HJ |
| 8 | `D2_HJ` | HJ `e3/e2³`, all constituents, beta=1 | dimensionless | Same guard as LJ D2 |
| 9 | `constituent_multiplicity_HJ` | Number of all HJ EFlow constituents | count | Integer count, including zero; no missing sentinel |
| 10 | `constituent_multiplicity_LJ` | Number of all LJ EFlow constituents | count | Same as HJ |
| 11 | `track_multiplicity_HJ` | Number of HJ constituents of type `EFlowTrack` | count | Zero when no tracks; no truth or charge predicate |
| 12 | `track_multiplicity_LJ` | Number of LJ constituents of type `EFlowTrack` | count | Same as HJ |
| 13 | `charged_energy_fraction_HJ` | `sum_{i in EFlowTrack} E_i/E_J` in HJ | dimensionless | NaN if `E_J==0`; zero if there are no tracks |
| 14 | `charged_energy_fraction_LJ` | Same track-energy fraction for LJ | dimensionless | Same as HJ |
| 15 | `jet_girth_HJ` | `sum_i (E_i/E_J)*theta(i,J)` for all HJ constituents | radians | NaN for empty constituents or any zero constituent/jet momentum norm |
| 16 | `jet_girth_LJ` | Same energy-weighted opening-angle sum for LJ | radians | Same as HJ |
| 17 | `tau21_HJ` | HJ `tau2/tau1` using the axes below | dimensionless | NaN if `tau1` is zero or nonfinite |
| 18 | `tau21_LJ` | Same N-subjettiness ratio for LJ | dimensionless | Same as HJ |

For D2, define `z_i=E_i/E_sum` and

```text
e2 = sum_{i<j} z_i z_j theta(i,j)
e3 = sum_{i<j<k} z_i z_j z_k theta(i,j) theta(i,k) theta(j,k)
D2 = e3 / e2^3
```

There is no R normalization in these correlators. With exactly two constituents
and nonzero e2, e3 and D2 are zero, not missing. Substructure zero-momentum angles
use the zero convention above.

For tau21, recluster the jet's complete constituent set using the same `ee_genkt`
definition (p=1, R=0.15, E-scheme), and obtain exclusive one- and two-jet axes.
For one constituent both axis sets are that constituent; for two constituents
the two-axis set is the two inputs themselves. There is no axis minimization.

```text
tau_N = sum_i E_i min_{a in axes_N} theta(i,a) / (E_sum * R)
```

Tau is NaN for empty constituents, empty axes, or `E_sum<=0`. A nonfinite or
exactly zero tau1 makes tau21 NaN. No epsilon denominator replacement is used.
The girth denominator is `E_J`; in the selected domain `E_J>500 GeV`.
Counts and all other columns enter the native DMatrix via float64 arrays;
XGBoost's internal storage is implementation-defined. NaNs are passed with
`missing=np.nan`; infinite inputs are rejected by the public wrapper.
