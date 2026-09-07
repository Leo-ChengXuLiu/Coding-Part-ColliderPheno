# Feature definitions for the 10 TeV tc analysis

The exact ordered feature lists are in
[`configs/tc_10tev_plot_contract.yaml`](configs/tc_10tev_plot_contract.yaml).
All model inputs are reconstructed quantities. Truth flavor is used only to
supervise the three Stage-1 jet taggers.

## Objects and roles

Particle-flow candidates are clustered with FastJet `ee_genkt`, `p=1`, and
`R=0.10`. Jets must have `E>500 GeV`. `j1` and `j2` are the two highest-energy
jets. The larger-mass member is `HJ`; the other is `LJ`.

## Event features

| Group | Definition |
|---|---|
| `njet_all`, `njet_e500` | Number of reconstructed jets before and after the 500 GeV energy requirement. |
| `nlepton_prompt_iso` | Number of reconstructed Delphes electrons or muons with energy above 100 GeV. |
| `n_visible`, `n_charged`, `n_neutral` | Particle-flow candidate multiplicities. |
| `visible_*` | Four-vector sum of all visible particle-flow candidates. |
| `missing_*` | Missing four-vector defined from the 10 TeV initial state and the visible sum. |
| `thrust`, `thrust_major`, `thrust_minor` | Standard momentum-weighted event-axis observables. |
| `sphericity`, `aplanarity`, `c_parameter`, `d_parameter` | From eigenvalues `lambda1>=lambda2>=lambda3` of the normalized momentum tensor: `S=3/2(lambda2+lambda3)`, `A=3/2 lambda3`, `C=3 sum_{i<j} lambda_i lambda_j`, `D=27 lambda1 lambda2 lambda3`. |
| `j1_*`, `j2_*`, `hj_*`, `lj_*` | `mass`, `energy`, `pt`, `eta`, rapidity, `phi`, and `cos(theta)` for the four named jet roles. |
| `mjj`, `ejj`, `ptjj`, `yjj` | Invariant mass, energy, transverse momentum, and rapidity of `HJ+LJ`. |
| `delta_phi_jj`, `delta_eta_jj`, `delta_y_jj`, `delta_r_jj` | Signed angular/rapidity separations; `delta_phi` is wrapped to `[-pi,pi]`. |
| `opening_angle_jj` | Three-dimensional opening angle between HJ and LJ momenta. |
| `acoplanarity_jj` | `abs(pi-delta_phi_jj)`. |
| `energy_asymmetry`, `pt_asymmetry`, `mass_asymmetry` | `(HJ-LJ)/(HJ+LJ)` for the named quantity. |
| `abs_cos_theta_hj` | Absolute HJ polar-angle cosine. |
| `roe_n_candidates`, `roe_energy` | Number and energy of particle-flow candidates outside HJ and LJ. |
| `roe_*_fraction` | ROE quantity divided by the corresponding event total. |
| `hj_lj_candidate_fraction`, `hj_lj_energy_fraction` | Fraction of visible candidates or energy assigned to HJ and LJ. |

## Per-jet features

| Group | Definition |
|---|---|
| `mass`, `energy`, `pt`, `eta`, `rapidity`, `cos_theta` | Reconstructed jet kinematics. |
| `n_constituents`, `n_charged`, `n_neutral`, `n_photon`, `n_electron`, `n_muon` | Constituent multiplicities by detector object type. |
| `charged_energy_fraction`, `neutral_energy_fraction` | Charged or neutral constituent energy divided by total constituent energy. |
| `em_energy_fraction`, `had_energy_fraction` | Delphes electromagnetic or hadronic calorimeter energy divided by total constituent energy. |
| `lead_constituent_fraction` | Largest constituent energy divided by total constituent energy. |
| `ptd` | `sqrt(sum_i pt_i^2)/sum_i pt_i`. |
| `girth` | `sum_i pt_i DeltaR(i,jet)/sum_i pt_i`. |
| `width` | `sqrt(sum_i pt_i DeltaR(i,jet)^2/sum_i pt_i)`. |
| `jet_charge_k03/k05/k10` | `sum_i q_i pt_i^kappa / pt_jet^kappa`, for `kappa=0.3,0.5,1.0`. |
| `tau1`, `tau2`, `tau3` | Energy-weighted N-subjettiness using exclusive `ee_genkt` axes, normalized by `sum_i E_i R`. |
| `tau21`, `tau32` | `tau2/tau1` and `tau3/tau2`. |
| `ecf1`, `ecf2`, `ecf3` | Energy-correlation functions from the 30 highest-energy constituents; angular exponent one. |
| `c2`, `d2`, `n2` | `ecf3/ecf2^2`, `ecf3/ecf2^3`, and the generalized two-smallest-angle ECF3 divided by `ecf2^2`. |
| `softdrop_mass`, `softdrop_zg`, `softdrop_rg` | Cambridge/Aachen declustering (`p=0`) with `zcut=0.1`, `beta=0`; groomed mass and accepted splitting symmetry/angle. |
| `n_soft_electrons`, `n_soft_muons` | Electron and muon track constituents. |
| `soft_lepton_ptrel_max` | Largest lepton momentum transverse to the jet axis. |

## Impact-parameter features

For a valid track, `Sd0=d0/sigma(d0)`, `Sz0=dz/sigma(dz)`, and
`S3D=hypot(Sd0,Sz0)`. The uncertainties are the Delphes TrackSmearing outputs,
not constants fitted from the analysis sample.

| Feature | Definition |
|---|---|
| `n_ip_tracks` | Number of constituents with finite positive `sigma(d0)` and `sigma(dz)`. |
| `n_ip2d_sig_gt2/gt3` | Counts with `abs(Sd0)>2` or `>3`. |
| `n_ip3d_sig_gt2/gt3` | Counts with `abs(S3D)>2` or `>3`. |
| `ip2d_sig_abs_max/second/mean/median/sum` | Summary statistics of `abs(Sd0)`. |
| `ipz_sig_abs_max/mean` | Summary statistics of `abs(Sz0)`. |
| `ip3d_sig_abs_max/second/mean/sum` | Summary statistics of `abs(S3D)`. |
| `displaced_track_pt_fraction_sig2` | Constituent-`pt` fraction carried by tracks with `abs(Sd0)>2`. |
| `displaced_track_energy_fraction_sig2` | Constituent-energy fraction carried by tracks with `abs(Sd0)>2`. |

## OOF role features

The first three are five-fold out-of-fold Stage-1 scores. With an epsilon clip
of `1e-6`, the remaining inputs are

```text
lj_role_c_over_b_log_score = log(P_LJ(c) / P_LJ(b))
tc_role_score_product      = P_HJ(b) * P_LJ(c)
tc_role_score_contrast     = P_HJ(b) + P_LJ(c) - P_LJ(b)
```

The scores use balanced training priors and are not calibrated physical
probabilities.
