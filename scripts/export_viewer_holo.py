"""Exports a PBC-unwrapped, clean holo PDB (protein + PFBA + 6 A pocket waters) for Mol* 3D viewing
and prints exact active-site distances, angles, and steric contacts blocking the SN2 angle."""

from pathlib import Path
import math
import numpy as np


def parse_pdb(pdb_path: Path):
    box = None
    prot_lines = []
    wat_residues = {}
    lig_lines = []

    with open(pdb_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                box = np.array([float(line[6:15]), float(line[15:24]), float(line[24:33])], dtype=float)
            elif line.startswith(("ATOM  ", "HETATM")):
                resname = line[17:20].strip()
                chain = line[21].strip()
                resseq = line[22:26].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                rec = {
                    "raw": line,
                    "serial": int(line[6:11]),
                    "name": line[12:16].strip(),
                    "resname": resname,
                    "chain": chain,
                    "resseq": resseq,
                    "xyz": np.array([x, y, z], dtype=float),
                    "elem": line[76:78].strip(),
                }
                if resname == "LIG":
                    lig_lines.append(rec)
                elif resname in ("HOH", "WAT"):
                    key = (chain, resseq, len(wat_residues) if (chain, resseq) in wat_residues and rec["name"] == "O" else -1)
                    # Group every 3 atoms (O, H1, H2) as one water molecule
                    if rec["name"] == "O":
                        cur_wid = len(wat_residues) + 1
                        wat_residues[cur_wid] = [rec]
                    else:
                        wat_residues[len(wat_residues)].append(rec)
                elif resname not in ("NA", "CL", "CA"):
                    prot_lines.append(rec)
    return box, prot_lines, wat_residues, lig_lines


def mic_shift(pt: np.ndarray, ref_pt: np.ndarray, box: np.ndarray) -> np.ndarray:
    return pt - box * np.round((pt - ref_pt) / box)


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u = (a - b) / np.linalg.norm(a - b)
    v = (c - b) / np.linalg.norm(c - b)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u, v))))))


def main():
    in_pdb = Path("/Users/o2satz/antigravity/pfas/results/head_to_head_tip3p/runs/V12_V10_I153A_Y141L__PFBA/rep02_minimized.pdb")
    out_artifact = Path("/Users/o2satz/.gemini/antigravity/brain/1e66644c-6bfc-43f0-b691-dcbc01afc00e/structures/V12_PFBA_holo.pdb")
    out_workspace = Path("/Users/o2satz/antigravity/pfas/results/head_to_head_tip3p/V12_PFBA_holo_unwrapped.pdb")

    box, prot, wats, lig = parse_pdb(in_pdb)

    # Anchor on Asp110 CG in Chain A
    asp110_cg = [r["xyz"] for r in prot if r["resseq"] == "110" and r["name"] == "CG"][0]

    # Unwrap LIG into pocket frame
    lig_center = np.mean([r["xyz"] for r in lig], axis=0)
    lig_target_center = mic_shift(lig_center, asp110_cg, box)
    lig_delta = lig_target_center - lig_center
    for r in lig:
        r["xyz"] = r["xyz"] + lig_delta

    lig_pts = np.array([r["xyz"] for r in lig])

    # Keep structured pocket waters within 5.8 A of LIG or Asp110/His280
    his280_ne2 = [r["xyz"] for r in prot if r["resseq"] == "280" and r["name"] == "NE2"][0]
    kept_wats = []
    for wid, w_atoms in wats.items():
        o_pt = mic_shift(w_atoms[0]["xyz"], asp110_cg, box)
        w_delta = o_pt - w_atoms[0]["xyz"]
        d_lig = float(np.min(np.linalg.norm(lig_pts - o_pt[None, :], axis=1)))
        d_cat = min(float(np.linalg.norm(o_pt - asp110_cg)), float(np.linalg.norm(o_pt - his280_ne2)))
        if d_lig <= 5.5 or d_cat <= 5.8:
            shifted = []
            for wa in w_atoms:
                cp = dict(wa)
                cp["xyz"] = wa["xyz"] + w_delta
                shifted.append(cp)
            kept_wats.append(shifted)

    # Write clean PDB with sequential serials, PFBA as chain A resseq 305 (resname PFB), and CONECT records
    out_lines = ["REMARK   1 V12 (I253A/K181M/W185Y/I153A/Y141L) + PFBA EXPLICIT-WATER MINIMIZED HOLO COMPLEX\n"]
    serial = 1
    lig_old_to_new = {}

    for r in prot:
        x, y, z = r["xyz"]
        out_lines.append(
            f"ATOM  {serial:5d} {r['name']:<4s} {r['resname']:>3s} A{int(r['resseq']):4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {r['elem']:>2s}\n"
        )
        serial += 1
    out_lines.append(f"TER   {serial:5d}      {prot[-1]['resname']:>3s} A{int(prot[-1]['resseq']):4d}\n")
    serial += 1

    # Ligand on Chain A residue 305 so Mol* displays it right in Chain A's pocket
    for r in lig:
        x, y, z = r["xyz"]
        lig_old_to_new[r["name"]] = serial
        out_lines.append(
            f"HETATM{serial:5d} {r['name']:<4s} PFB A 305    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {r['elem']:>2s}\n"
        )
        serial += 1

    for w_idx, w_atoms in enumerate(kept_wats, start=401):
        for wa in w_atoms:
            x, y, z = wa["xyz"]
            out_lines.append(
                f"HETATM{serial:5d} {wa['name']:<4s} HOH A{w_idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {wa['elem']:>2s}\n"
            )
            serial += 1

    # Add explicit CONECT bonds for PFB so Mol* renders exact C-C, C-O, and C-F single/double bonds
    bonds = [
        ("C1x", "O1x"), ("C1x", "O2x"), ("C1x", "C2x"),
        ("C2x", "F1x"), ("C2x", "F2x"), ("C2x", "C3x"),
        ("C3x", "F3x"), ("C3x", "F4x"), ("C3x", "C4x"),
        ("C4x", "F5x"), ("C4x", "F6x"), ("C4x", "F7x"),
    ]
    adj = {}
    for a, b in bonds:
        ia, ib = lig_old_to_new[a], lig_old_to_new[b]
        adj.setdefault(ia, []).append(ib)
        adj.setdefault(ib, []).append(ia)
    for ia in sorted(adj.keys()):
        nbrs = "".join(f"{ib:5d}" for ib in sorted(adj[ia]))
        out_lines.append(f"CONECT{ia:5d}{nbrs}\n")
    out_lines.append("END\n")

    for dst in (out_artifact, out_workspace):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("".join(out_lines), encoding="utf-8")

    # Now print exact 3D geometric analysis of V10 + PFBA!
    lig_by_name = {r["name"]: r["xyz"] for r in lig}
    ca = lig_by_name["C2x"]
    c1 = lig_by_name["C1x"]
    cb = lig_by_name["C3x"]
    cg = lig_by_name["C4x"]
    f1, f2 = lig_by_name["F1x"], lig_by_name["F2x"]

    od1 = [r["xyz"] for r in prot if r["resseq"] == "110" and r["name"] == "OD1"][0]
    od2 = [r["xyz"] for r in prot if r["resseq"] == "110" and r["name"] == "OD2"][0]
    od_best = od1 if np.linalg.norm(od1 - ca) < np.linalg.norm(od2 - ca) else od2

    print("=== V12 (I253A/K181M/W185Y/I153A/Y141L) + PFBA Unwrapped Pocket Geometry ===")
    print(f"Kept {len(kept_wats)} structured pocket waters; wrote {out_artifact}")
    for fn, fp in [("F1x", f1), ("F2x", f2)]:
        print(
            f"Alpha-{fn}: d(Asp110-OD..Ca)={np.linalg.norm(od_best - ca):.2f} A | "
            f"theta(OD-Ca-{fn})={angle_deg(od_best, ca, fp):.1f} deg"
        )
        for seq, aname in [("155", "HE2"), ("156", "HE1"), ("219", "HH")]:
            hp = [r["xyz"] for r in prot if r["resseq"] == seq and r["name"] == aname][0]
            print(f"   d(Res{seq}:{aname} .. {fn}) = {np.linalg.norm(hp - fp):.2f} A")

    print("\n=== Closest Protein Heavy-Atom Contacts to PFBA C_beta / C_gamma / Tail-Fs ===")
    tail_Check = [r for r in lig if r["name"] not in ("C1x", "O1x", "O2x")]
    res_min = {}
    for pr in prot:
        if pr["elem"] == "H":
            continue
        for lr in tail_Check:
            d = float(np.linalg.norm(pr["xyz"] - lr["xyz"]))
            key = f"{pr['resname']}{pr['resseq']}"
            if d < 4.3 and (key not in res_min or d < res_min[key][0]):
                res_min[key] = (d, pr["name"], lr["name"])
    for key, (d, pa, la) in sorted(res_min.items(), key=lambda x: x[1][0]):
        print(f"  {key:8s}:{pa:4s} <-> PFBA:{la:4s} = {d:.2f} A")


if __name__ == "__main__":
    main()
