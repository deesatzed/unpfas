# PFAS Enzymatic Defluorination Triage Pipeline (`DESIGN.md`)

## 1. Scientific Objective & Scope

Cleaving aliphatic $\text{C–F}$ bonds ($\text{BDE} \approx 480\text{–}540\text{ kJ/mol}$) in per- and polyfluoroalkyl substances (PFAS) is limited by both transition-state electrostatics and steric accommodation of rigid perfluoroalkyl chains. Because classical molecular mechanics cannot form or break covalent $\text{C–F}$ bonds directly, this pipeline uses **structure prediction + explicit-solvent all-atom MD** where its physical leverage is strongest: **eliminating fold-destabilizing or non-productive variants early and ranking designs by ground-state catalytic pre-organization, local electric-field polarization, and environmental matrix robustness** prior to QM/MM barrier refinement and wet-lab expression.

### Anchor Enzyme Scaffold
* **Enzyme**: Fluoroacetate dehalogenase **RPA1163** from *Rhodopseudomonas palustris* (UniProt `Q6NAM1`, $\alpha/\beta$ hydrolase core + $\alpha$-helical cap domain; obligate asymmetric homodimer).
* **Structural Templates**:
  * **PDB `3R3V`** (`data/structures/pdb_00003r3v.cif`): `D110N` Michaelis complex co-crystallized with fluoroacetate (`FAH 305` bound in Protomer A; Protomer B empty, capturing native half-of-the-sites asymmetry). Reversion `Asn110 -> Asp110` establishes the wild-type holo coordinates.
  * **PDB `3R3U`** (`data/structures/pdb_00003r3u.cif`): Wild-type apo homodimer reference for `Asp110` rotamer and apo cap-domain baseline.

### 3-Step "Stepping-Stone" Substrate Ladder
All substrates have $\text{p}K_a < 1$ and are modeled exclusively in their deprotonated **anionic carboxylate ($-1$)** state across environmental pH (`4.5–8.5`):

| Tier | Substrate | Formula | Role in Pipeline | Expected Turnovers / Stoichiometry |
| :--- | :--- | :--- | :--- | :--- |
| **Control 0** | **Monofluoroacetate (`MFA`)** | $\text{CH}_2\text{FCOO}^-$ | Fast native positive control; calibrates WT NAC occupancy and $\vec{E}_{\text{proj}}$ | $1\ \text{F}^-$ via $\text{S}_\text{N}2$ ester + hydrolysis -> glycolate |
| **Bridge 1** | **Difluoroacetate (`DFA`)** | $\text{CHF}_2\text{COO}^-$ | Known slow positive control; quantifies gem-difluoro steric/electronic penalty | $2\ \text{F}^-$ ($1$ enzymatic $\text{S}_\text{N}2$ + $1$ spontaneous $\alpha$-fluoroalcohol elimination -> glyoxylate) |
| **Target 2** | **Perfluorobutanoate (`PFBA`)** | $\text{CF}_3\text{CF}_2\text{CF}_2\text{COO}^-$ | Short-chain ($\text{C}_4$) perfluoroalkyl stretch target | $2\ \text{F}^-$ if Step 1 ($\text{S}_\text{N}2$) & Step 2 (ester hydrolysis) both complete -> $\text{CF}_3\text{CF}_2\text{COCOO}^-$ |

---

## 2. Crystallographic Vector Analysis of `3R3V` (Why `Ile253` Is Gatekeeper #1)

Direct 3D geometric decomposition around bound fluoroacetate (`FAH 305`, Chain A) in `3R3V` establishes sub-Ångström reference distances and reveals the exact steric bottleneck for perfluoroalkyl expansion:

### Native Active-Site Reference Contacts (`3R3V` Chain A)
* **Halide-Stabilizing Cradle (to scissile `FAH:F`)**:
  * `His155:NE2` — **`2.98 Å`**
  * `Tyr219:OH` — **`3.28 Å`**
  * `Trp156:NE1` — **`3.32 Å`**
* **Carboxylate Clamp (to `FAH:O` / `FAH:OXT`)**:
  * `Arg111:NH1` — **`2.69 Å`**
  * `Arg114:NE` / `NH1` — **`2.81 Å`** / **`2.98 Å`**
* **Catalytic Triad**:
  * `Asp110` (`Asn110` in `3R3V`, `d(ND2···Cα) = 3.28 Å`), `His280` (`d(NE2···Cα) = 4.30 Å`), `Asp134` (charge-relay acid).

### Tetrahedral Prochiral Vector Projection from $\text{C}_\alpha$
Fixing `FAH:C` (carboxylate carbon) in the `Arg111`/`Arg114` clamp and `FAH:F` in the `His155`–`Trp156`–`Tyr219` halide cradle leaves exactly two tetrahedral $sp^3$ vectors ($\hat{v}_1, \hat{v}_2$) at $\text{C}_\alpha$:

1. **Prochiral Vector $\hat{v}_1$ (Catalytic Cleft — Dead Wall for Alkyl Chains)**
   * Projects directly into **`Asp110`/`Asn110` (`2.05 Å` at $1.54\text{ \AA}$ $\text{C}_\beta$ bond length)** and **`His280` (`1.61 Å` at $2.90\text{ \AA}$ $\text{C}_\gamma$ distance)**.
   * *Implication*: A perfluoroethyl tail ($-\text{CF}_2\text{CF}_3$) **cannot** occupy $\hat{v}_1$. In both `DFA` and `PFBA`, $\hat{v}_1$ must hold the single **non-scissile $\alpha$-fluorine** (`1.36 Å` bond length, `2.17 Å` from `Asn110:ND2`), explaining the intrinsic lone-pair repulsion against the incoming `Asp110` carboxylate.
2. **Prochiral Vector $\hat{v}_2$ (Fluorophilic Alcove Vector — Primary Engineering Axis)**
   * Projects away from `Asp110`/`His280` directly into **`Ile253`** (`d(Cα···CD1) = 3.82 Å`).
   * Extending `PFBA` along $\hat{v}_2$ places the $\text{C}_\gamma$ zone (`2.80 Å` from $\text{C}_\alpha$) **`2.60 Å`** from `Ile253:CG2` and the terminal $-\text{CF}_3$ zone (`4.10 Å` from $\text{C}_\alpha$) at a severe **`2.11 Å` steric clash with `Ile253:CG2`**.
   * Secondary shell framing this alcove and pocket rim: **`Ile135`** (`5.33 Å`, backside `172.3°`), **`Trp185`** (`5.10 Å`, cap wall/gate), **`Ile153`** (`6.31 Å`), **`Phe40`** (`6.37 Å`), and **`Tyr141`** (`7.18 Å`, distal alcove rim).

---

## 3. Zoned Variant Design Rules & v0.1 Benchmark Panel

### Mutational Zones
* **Zone 0 — Immutable Catalytic Core (Frozen)**: `Asp110`, `His280`, `Asp134`, `His155`, `Trp156`, `Tyr219`, `Arg111`, `Arg114`.
* **Zone 1 — Primary $\hat{v}_2$ Alcove Wall (Steric Relief + Fluorophilic Packing)**: `Ile253` (primary steric gatekeeper), `Ile135`, `Ile153`, `Phe40`, `Tyr141`. *Negative design rule*: forbid introducing polar/anionic H-bond acceptors (`Ser`, `Thr`, `Asp`, `Glu`) facing $-\text{CF}_2\text{CF}_3$.
* **Zone 2 — Second-Shell Cradle & Triad Buttressing**: `Phe281` (behind `His280`), `His109` (behind `Asp110`/`Arg111`), hydrophobic packing behind `His155`–`Trp156`–`Tyr219` to maximize scissile $\vec{E}_{\text{proj}}$.
* **Zone 3 — Cap-Domain Channel & Dimer Allostery**: `Lys181`–`Trp185` allosteric gate controlling the ~11 Å access channel and open/closed cap dynamics.

### Curated 10-Variant Benchmark Panel (`v0.1`)

| ID | Variant | Zone(s) | Design Hypothesis / Control Role |
| :--- | :--- | :---: | :--- |
| `V01` | **WT (`RPA1163`)** | — | Native reference: high NAC on `MFA`, reduced NAC on `DFA`, steric rejection/tilting on `PFBA` |
| `V02` | **`D110N`** | `0` | Negative catalytic control (`3R3V` crystal baseline; binds headgroup/F without anionic nucleophile) |
| `V03` | **`K181M/W185Y`** | `3` | Evolved cap-domain allosteric gate benchmark (cf. PDB `9U5X`); tests channel breathing alone |
| `V04` | **`I253V`** | `1` | Conservative `-CH2` trim along $\hat{v}_2$; tests `DFA` optimization vs. partial `PFBA` relief |
| `V05` | **`I253A`** | `1` | Primary $\hat{v}_2$ alcove carve eliminating the `2.11 Å` `Ile253:CG2` clash with `PFBA` $-\text{CF}_3$ |
| `V06` | **`I253G`** | `1` | Over-carving boundary control; tests whether excessive cavity volume floods the pocket with water or destabilizes $\beta$-strand packing |
| `V07` | **`I253A/I135V`** | `1` | Dual $\hat{v}_2$ + backside (`172°`) steric relief to prevent $\text{C}_\alpha$ recoil away from `Asp110` |
| `V08` | **`I253A/Y141F`** | `1` | Alcove carve + de-hydroxylation of distal `Tyr141` (`7.18 Å`) to create an apolar aromatic face for $-\text{CF}_3$ |
| `V09` | **`I253A/W185F`** | `1 + 3` | Alcove carve + cap-wall slimming while retaining quadrupole aromatic packing |
| `V10` | **`I253A/K181M/W185Y`** | `1 + 3` | Synergistic $\hat{v}_2$ fluorophilic alcove + evolved allosteric cap-gate triple variant |

---

## 4. Two-Tier Compute Funnel & Matrix Protocol

1. **Force-Field & Ligand Validation (`OpenFF Sage 2.2` + AM1-BCC)**
   * Parameterize `MFA`, `DFA`, and `PFBA` (`-1` carboxylates) with `OpenFF Sage`.
   * Run a short explicit-water ligand sanity simulation (`5 ns`) on `PFBA` verifying fluorocarbon chain stiffness ($\text{C–C–C–C}$ helical twist $\sim 160^\circ\text{–}170^\circ$, no unphysical collapsed hairpins).
2. **Fold Sanity Pre-Filter (AlphaFold2 / Boltz-2)**
   * Predict apo/holo variant structures; enforce **Stage A Gate 1** before MD allocation.
3. **Tier 1 — Fast Restrained-Monomer Screen (All 10 Variants × 3 Substrates)**
   * **System**: Protomer A (`Amber ff14SB` / `ff19SB` + `TIP3P`/`OPC`, `0.15 M NaCl`, `pH 7.5` tautomer assignment: `His155` $\text{HID}/\text{HIE}$ donating $\text{N}_{\varepsilon 2}\text{–H}$ to $\text{F}$; `His280` neutral $\text{N}_{\delta 1}\text{–H}$ leaving $\text{N}_{\varepsilon 2}$ lone pair toward water/`Asp110`).
   * **Boundary Restraint**: Weak harmonic restraint (`0.5 kcal/mol/Å²`) applied **only** to peripheral dimer-interface $\text{C}_\alpha$ atoms `> 15 Å` from `FAH:Cα`—preserving monomer fold integrity without restricting active-site or cap-domain (`Lys181–Trp185`) breathing.
   * **Sampling**: `3 independent replicates × 50 ns` (`150 ns` per variant–substrate pair).
4. **Tier 2 — Asymmetric Homodimer + Groundwater Matrix Stress Test (Top 20% Hits + WT)**
   * **System**: Full unrestrained biological homodimer (`Protomer A: holo`, `Protomer B: apo`), `3 × 200 ns`.
   * **Environmental Matrix Grid**:
     * *pH shift*: `pH 7.5` (standard neutral `His155`/`His280`) vs. `pH 5.5` (protonated `His+` sensitivity check).
     * *Temperature*: `288 K` (`15 °C` subsurface groundwater) vs. `303 K` (`30 °C` engineered bioreactor).
     * *Ionic matrix*: `+ 10 mM CaCl2` to test divalent cation bridging/competition at the PFAS $-\text{COO}^-$ headgroup.

---

## 5. Two-Stage Filtering & Quantitative Scoring Rubric

### Stage A: Hard Veto Gates (Pass/Fail)
1. **Structural Integrity Veto**: Active-site + cap-interface `pLDDT < 85` OR Zone-0 heavy-atom `RMSD > 0.8 Å` vs. `3R3V`.
2. **Dead-End / Unbound Pose Veto**: Reject if terminal $-\text{CF}_3$ fluorines occupy the `His155`–`Trp156`–`Tyr219` cradle (`d(tail-F ··· cradle) < d(α-F ··· cradle)` for `> 50%` of equilibration frames) OR headgroup unbinds (`min d(Arg111/114-Nη ··· O_COO⁻) > 4.5 Å` for `> 30%` of frames).

### Stage B: Simultaneous 4-Criteria NAC Indicator
For frame $t$, $\text{NAC}(t) = 1$ if **either** $\alpha$-fluorine ($F_{\alpha,1}$ or $F_{\alpha,2}$) simultaneously satisfies:
1. **Nucleophile attack distance**: $\min\bigl(d(\text{Asp110-O}_{\delta 1}, \text{C}_\alpha),\, d(\text{Asp110-O}_{\delta 2}, \text{C}_\alpha)\bigr) \le 3.3\text{ \AA}$
2. **Backside $\text{S}_\text{N}2$ colinearity**: $\theta(\text{Asp110-O}_\delta \text{–} \text{C}_\alpha \text{–} \text{F}_{\text{scissile}}) \in [155^\circ, 180^\circ]$
3. **Halide cradle engagement**: $\ge 2$ of $\{\text{His155-HE2},\, \text{Trp156-HE1},\, \text{Tyr219-HH}\}$ within $\le 2.5\text{ \AA}$ of $\text{F}_{\text{scissile}}$ (heavy-atom donor–F $\le 3.35\text{ \AA}$)
4. **Carboxylate clamp lock**: $\min d(\text{Arg111/Arg114-N}_\eta \cdots \text{O}_{\text{COO}^-}) \le 3.4\text{ \AA}$

### Composite Ranking Score ($S_{\text{total}} \in [0, 100]$)

$$S_{\text{total}} = 35\,\hat{P}_{\text{NAC}} + 25\,\hat{E}_{\text{proj}} + 20\,\hat{R}_{\text{pose}} + 20\,\hat{M}_{\text{robust}}$$

Each sub-metric $\hat{X} \in [0, 1]$ is min–max scaled across the panel (anchored at `0` = dead threshold, `1` = WT-on-`MFA` catalytic ceiling or panel maximum):

| Sub-Metric | Weight | Exact Computation from MD Trajectory | Mechanistic Failure Caught |
| :--- | :---: | :--- | :--- |
| **$\hat{P}_{\text{NAC}}$** | **35 pts** | $\frac{1}{N_{\text{frames}}} \sum_t \text{NAC}(t)$ across production replicates | Steric tilting of $\text{C}_\alpha$ or loss of backside attack angle |
| **$\hat{E}_{\text{proj}}$** | **25 pts** | Mean protein+solvent electric field $\langle \vec{E}(\mathbf{r}_{\text{mid}}) \cdot \hat{u}_{\text{C}_\alpha\to\text{F}} \rangle_{\text{NAC}}$ (MV/cm) at the scissile $\text{C–F}$ midpoint | Insufficient electrostatic stabilization to polarize a perfluoro $\text{C–F}$ bond |
| **$\hat{R}_{\text{pose}}$** | **20 pts** | $0.5\,(1 - \text{clip}(\text{RMSF}_{\text{C}_\alpha}/2.0\text{ \AA})) + 0.5\,f_{\text{wat,Step2}}$, where $f_{\text{wat,Step2}}$ is fraction of frames with water $\text{O}$ within $3.5\text{ \AA}$ of `His280:NE2` and $4.2\text{ \AA}$ of `Asp110:CG` | Non-productive sliding OR **Step-2 covalent acyl-enzyme suicide trap** |
| **$\hat{M}_{\text{robust}}$** | **20 pts** | Tier 1: $\text{AF2 active-site pLDDT}/100 \times (1 - \text{SE}(P_{\text{NAC}})/\mu)$; Tier 2: $\min_{\text{matrix}}(P_{\text{NAC}}) / P_{\text{NAC, std}}$ across pH/temp/$\text{Ca}^{2+}$ | Variants that collapse in cold/acidic/hard groundwater |

---

## 6. Modular Open-Source Repository Architecture

```text
unpfas/
├── DESIGN.md                        # Architecture, 3R3V vector analysis, zones & rubric
├── configs/
│   └── benchmark_v01.yaml           # 10 variants, 3 substrates, MD & rubric cutoffs
├── data/
│   ├── structures/                  # pdb_00003r3v.cif, pdb_00003r3u.cif
│   └── ligands/                     # MFA, DFA, PFBA protonated SDFs & OpenFF XML/prmtop
├── src/pfas_bench/
│   ├── prep_receptor.py             # D110N->WT reversion, in-silico mutagenesis, His tautomers
│   ├── prep_ligands.py              # SMILES -> pH 7.4 carboxylate -> OpenFF Sage + torsion check
│   ├── place_substrate.py           # Template-aligned placement onto FAH 305 clamp + cradle
│   ├── run_md.py                    # OpenMM Tier-1 monomer & Tier-2 homodimer runner
│   ├── analyze_nac_efield.py        # 4-rule NAC detector, E_proj calculator, Step-2 water check
│   └── score_and_plot.py            # Stage-A vetoes, S_total ranking table & diagnostic plots
└── results/
    └── summary_ranking.csv          # Transparent per-variant breakdown
```
