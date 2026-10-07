#!/usr/bin/env python3
"""
Independent verification of a Task 6 run directory.

Checks (each is written to verification.json and summarised in run_summary.json):

A. RMSE provenance - TRAIN/TEST RMSE are recomputed from the generated script's
   saved freerun_prediction.npy and the real data, and compared with the
   TRAIN_RMSE / TEST_RMSE lines that script printed.

B. Continuous u-only rollout (dynamic proof) - using the kept network saved by
   the script (final_accel.pt) and the centered input u ONLY, every one of the
   4095 RK4 steps is recomputed from the saved state trajectory
   freerun_states.npy:  rk4(state[k], u[k], u[k+1]) == state[k+1]  for all k.
   This shows that the state is initialised once at sample 0, that one
   continuous integration crosses the 3071 -> 3072 train/test boundary, that
   there is no state reset anywhere, and that measured y never enters the
   dynamics. The boundary step is also reported explicitly.

C. Source audit (static proof) - the free_run(accel, u, y0, v0) function of the
   generated script is parsed: one loop, no boundary branch, no reference to
   measured y, the state assigned only at initialisation and by the RK4 update.

D. Test-y mutation - the script is re-run unchanged, and re-run on a copy of
   SNLS80mV.mat whose measured test output is replaced by a mean-preserving
   mirror image plus an alternating offset. Keeping the version with the best
   test score is allowed, so the kept version may legitimately change; but if
   the kept weights are identical the free-run prediction must be identical,
   and check B must pass for the perturbed run too.

Usage:
    python task6/verify_run.py <run_dir> [--data path/to/SNLS80mV.mat] [--skip-reruns]
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

START, STOP, N_TRAIN, N = 42650, 46746, 3072, 4096
DT = 1.0 / 610.35
TOL_Y = 1e-6     # V   per-step agreement (float round-off only)
TOL_V = 1e-3     # V/s per-step agreement (float round-off only)
TOL_PRED = 1e-5  # V   prediction agreement between two runs


def excerpt(mat_path: Path):
    m = loadmat(mat_path)
    u = np.asarray(m["V1"], dtype=np.float64).reshape(-1)[START:STOP]
    y = np.asarray(m["V2"], dtype=np.float64).reshape(-1)[START:STOP]
    return u - u.mean(), y - y.mean()


def printed(stdout: str, key: str) -> float | None:
    hits = re.findall(rf"^\s*{key}\s*=\s*([0-9.eE+-]+)\s*$", stdout, re.MULTILINE)
    return float(hits[-1]) if hits else None


# ---------------------------------------------------------------- check B
def rk4_batch(accel, y, v, u0, u1, convention):
    um = 0.5 * (u0 + u1) if convention == "linear" else u0
    u4 = u1 if convention == "linear" else u0
    a1 = accel(y, v, u0)
    a2 = accel(y + 0.5 * DT * v, v + 0.5 * DT * a1, um)
    a3 = accel(y + 0.5 * DT * (v + 0.5 * DT * a1), v + 0.5 * DT * a2, um)
    a4 = accel(y + DT * (v + 0.5 * DT * a2), v + DT * a3, u4)
    k1y, k2y = v, v + 0.5 * DT * a1
    k3y, k4y = v + 0.5 * DT * a2, v + DT * a3
    y_next = y + DT / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
    v_next = v + DT / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
    return y_next, v_next


def replay_check(run_dir: Path, u: np.ndarray, y: np.ndarray) -> dict:
    """Recompute every RK4 step of the saved trajectory from the saved model and u only."""
    states = np.load(run_dir / "freerun_states.npy").astype(np.float64)
    pred = np.load(run_dir / "freerun_prediction.npy").reshape(-1).astype(np.float64)
    accel = torch.jit.load(str(run_dir / "final_accel.pt"), map_location="cpu").eval()
    params = list(accel.parameters())
    dtype = params[0].dtype if params else torch.float32
    boundary = {}
    if (run_dir / "boundary_state.json").exists():
        boundary = json.loads((run_dir / "boundary_state.json").read_text(encoding="utf-8"))

    out = {
        "states_shape": list(states.shape),
        "prediction_equals_states_y": bool(states.shape == (N, 2) and np.array_equal(pred, states[:, 0])),
        "model_dtype": str(dtype),
        "declared_u_convention": boundary.get("rk4_u_convention"),
    }
    if states.shape != (N, 2):
        out["passed"] = False
        return out

    t = lambda a: torch.tensor(a, dtype=dtype)
    with torch.no_grad():
        per_conv = {}
        for conv in ("linear", "zoh"):
            yn, vn = rk4_batch(accel, t(states[:-1, 0]), t(states[:-1, 1]), t(u[:-1]), t(u[1:]), conv)
            ey = np.abs(yn.double().numpy() - states[1:, 0])
            ev = np.abs(vn.double().numpy() - states[1:, 1])
            per_conv[conv] = {
                "max_step_err_y_V": float(ey.max()),
                "max_step_err_v_Vps": float(ev.max()),
                "boundary_step_err_y_V": float(ey[N_TRAIN - 1]),
                "boundary_step_err_v_Vps": float(ev[N_TRAIN - 1]),
                "steps_failing": int(np.sum((ey > TOL_Y) | (ev > TOL_V))),
                "y_next_3072": float(yn[N_TRAIN - 1]),
                "v_next_3072": float(vn[N_TRAIN - 1]),
            }
    conv = min(per_conv, key=lambda c: per_conv[c]["steps_failing"])
    best = per_conv[conv]

    v_fd = {
        "forward (y1-y0)/dt": (y[1] - y[0]) / DT,
        "2nd-order one-sided (-3y0+4y1-y2)/(2dt)": (-3 * y[0] + 4 * y[1] - y[2]) / (2 * DT),
    }
    out.update({
        "u_convention_matched": conv,
        "per_convention": per_conv,
        "rk4_steps_checked": N - 1,
        "steps_failing": best["steps_failing"],
        "initial_state": {"y0": float(states[0, 0]), "v0": float(states[0, 1])},
        "y0_equals_measured_y0": bool(abs(states[0, 0] - y[0]) < TOL_Y),
        "v0_finite_difference_candidates": {k: float(v) for k, v in v_fd.items()},
        "v0_matches_finite_difference": [k for k, v in v_fd.items() if abs(states[0, 1] - v) < 1e-3],
        "boundary": {
            "state_k3071_saved": [float(states[N_TRAIN - 1, 0]), float(states[N_TRAIN - 1, 1])],
            "state_k3072_saved": [float(states[N_TRAIN, 0]), float(states[N_TRAIN, 1])],
            "state_k3072_recomputed_from_k3071_and_u_only": [best["y_next_3072"], best["v_next_3072"]],
            "boundary_step_err_y_V": best["boundary_step_err_y_V"],
            "boundary_step_err_v_Vps": best["boundary_step_err_v_Vps"],
            "measured_y_3072_for_comparison": float(y[N_TRAIN]),
            "script_reported": boundary,
        },
    })
    out["passed"] = bool(
        out["prediction_equals_states_y"]
        and best["steps_failing"] == 0
        and out["y0_equals_measured_y0"]
        and out["v0_matches_finite_difference"]
    )
    return out


# ---------------------------------------------------------------- check C
def source_audit(script: Path) -> dict:
    src = script.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "free_run"),
        None,
    )
    if fn is None:
        return {"free_run_found": False, "passed": False}

    params = [a.arg for a in fn.args.args]
    loops = [n for n in ast.walk(fn) if isinstance(n, (ast.For, ast.While))]
    ifs = [n for n in ast.walk(fn) if isinstance(n, ast.If)]
    assigned = {}
    for n in ast.walk(fn):
        targets = []
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            targets = [n.target]
        for tgt in targets:
            for name in ast.walk(tgt):
                if isinstance(name, ast.Name):
                    assigned.setdefault(name.id, []).append(n.lineno)
    loaded = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    free_names = sorted(loaded - set(params) - set(assigned) - set(dir(builtins)))

    boundary_words = re.compile(r"train|test|split|boundary|3072|n_tr", re.IGNORECASE)
    boundary_ifs = [
        {"line": n.lineno, "test": ast.unparse(n.test)} for n in ifs
        if boundary_words.search(ast.unparse(n.test))
    ]
    has_3072 = any(isinstance(n, ast.Constant) and n.value in (3072, 3071) for n in ast.walk(fn))
    measured_y_like = re.compile(r"^(y|ys|y_?(c|meas|measured|data|true|target|test|train|all|full|centered|raw|np|t))$", re.IGNORECASE)
    suspicious = sorted(
        nm for nm in set(params) | set(free_names)
        if measured_y_like.match(nm) and nm not in ("y0",)
    )

    lines = src.splitlines()
    out = {
        "free_run_found": True,
        "signature": f"free_run({', '.join(params)})",
        "free_run_source": "\n".join(lines[fn.lineno - 1: fn.end_lineno]),
        "num_loops": len(loops),
        "loops": [ast.unparse(l).splitlines()[0] for l in loops],
        "if_branches": [{"line": n.lineno, "test": ast.unparse(n.test)} for n in ifs],
        "boundary_if_branches": boundary_ifs,
        "contains_constant_3071_or_3072": has_3072,
        "assignments_by_name": assigned,
        "external_names_read": free_names,
        "measured_y_like_names": suspicious,
    }
    out["passed"] = bool(
        params[:4] == ["accel", "u", "y0", "v0"]
        and len(loops) == 1
        and not boundary_ifs
        and not has_3072
        and not suspicious
    )
    return out


# ---------------------------------------------------------------- check D
def rerun(script: Path, mat: Path, label: str) -> tuple[Path, dict]:
    tmp = Path(tempfile.mkdtemp(prefix=f"verify_{label}_"))
    shutil.copy(script, tmp / script.name)
    env = dict(os.environ, SILVERBOX_MAT_PATH=str(mat), MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, script.name], cwd=tmp, env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=300,
    )
    return tmp, {
        "dir": str(tmp),
        "exit_code": proc.returncode,
        "TRAIN_RMSE": printed(proc.stdout, "TRAIN_RMSE"),
        "TEST_RMSE": printed(proc.stdout, "TEST_RMSE"),
        "stderr_tail": proc.stderr[-1500:],
    }


def same_weights(a: Path, b: Path) -> bool:
    sa = torch.jit.load(str(a / "final_accel.pt")).state_dict()
    sb = torch.jit.load(str(b / "final_accel.pt")).state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def mutation_check(script: Path, data: Path, y: np.ndarray, run_pred: np.ndarray) -> dict:
    base_dir, base = rerun(script, data, "original")
    base_pred = np.load(base_dir / "freerun_prediction.npy").reshape(-1)
    base["max_abs_pred_diff_vs_agent_run_V"] = float(np.max(np.abs(base_pred - run_pred)))

    m = loadmat(data)
    v2 = np.asarray(m["V2"], dtype=np.float64).reshape(-1).copy()
    seg = slice(START + N_TRAIN, STOP)
    yt = v2[seg]
    v2[seg] = 2.0 * yt.mean() - yt + 0.02 * np.where(np.arange(yt.size) % 2 == 0, 1.0, -1.0)
    m_out = {k: v for k, v in m.items() if not k.startswith("__")}
    m_out["V2"] = v2.reshape(np.asarray(m["V2"]).shape)
    tmp_mat = Path(tempfile.mkdtemp(prefix="verify_mat_")) / "SNLS80mV.mat"
    savemat(tmp_mat, m_out)
    u_p, y_p = excerpt(tmp_mat)

    pert_dir, pert = rerun(script, tmp_mat, "perturbed")
    pert_pred = np.load(pert_dir / "freerun_prediction.npy").reshape(-1)
    weights_same = same_weights(base_dir, pert_dir)
    pred_diff = float(np.max(np.abs(pert_pred - base_pred)))
    pert_replay = replay_check(pert_dir, u_p, y_p)

    out = {
        "perturbation": {
            "max_abs_change_test_y_V": float(np.max(np.abs(y_p[N_TRAIN:] - y[N_TRAIN:]))),
            "max_abs_change_train_y_V": float(np.max(np.abs(y_p[:N_TRAIN] - y[:N_TRAIN]))),
        },
        "rerun_original": base,
        "rerun_perturbed": pert,
        "script_read_perturbed_file": pert["TEST_RMSE"] != base["TEST_RMSE"],
        "kept_weights_identical": weights_same,
        "max_abs_pred_diff_perturbed_vs_original_V": pred_diff,
        "perturbed_run_replay_passed": pert_replay["passed"],
        "perturbed_run_replay_steps_failing": pert_replay.get("steps_failing"),
    }
    if weights_same:
        out["interpretation"] = (
            "Same kept weights; prediction must be identical although measured test y changed."
        )
        consistent = pred_diff < TOL_PRED
    else:
        out["interpretation"] = (
            "A different version was kept (test-score selection, allowed by the assignment); "
            "test-y independence of the dynamics rests on the u-only replay of the perturbed run."
        )
        consistent = True
    out["passed"] = bool(
        base["exit_code"] == 0 and pert["exit_code"] == 0
        and out["script_read_perturbed_file"] and consistent and pert_replay["passed"]
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--data", default=None)
    ap.add_argument("--skip-reruns", action="store_true", help="Skip check D (two re-runs).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data = Path(args.data or summary["data"]).resolve()
    script = run_dir / summary["final_script"]
    # The kept script's artifacts sit next to it (its own attempt directory, or
    # the run directory itself for a flat single-attempt run).
    art_dir = script.parent
    stdout = Path(str(script.with_suffix("")) + ".stdout.txt").read_text(encoding="utf-8")
    u, y = excerpt(data)

    report: dict = {"script": summary["final_script"], "data": str(data)}

    # A. RMSE provenance
    pred = np.load(art_dir / "freerun_prediction.npy").reshape(-1).astype(np.float64)
    rec_train = float(np.sqrt(np.mean((pred[:N_TRAIN] - y[:N_TRAIN]) ** 2)))
    rec_test = float(np.sqrt(np.mean((pred[N_TRAIN:] - y[N_TRAIN:]) ** 2)))
    p_train, p_test = printed(stdout, "TRAIN_RMSE"), printed(stdout, "TEST_RMSE")
    report["A_rmse_provenance"] = {
        "printed_TRAIN_RMSE": p_train,
        "printed_TEST_RMSE": p_test,
        "recomputed_TRAIN_RMSE": rec_train,
        "recomputed_TEST_RMSE": rec_test,
        "passed": bool(
            pred.shape == (N,) and p_train is not None and p_test is not None
            and abs(rec_train - p_train) < 1e-6 and abs(rec_test - p_test) < 1e-6
        ),
    }

    # B. step-by-step u-only replay of the agent's own run
    report["B_continuous_u_only_rollout"] = replay_check(art_dir, u, y)

    # C. static audit of free_run
    report["C_free_run_source_audit"] = source_audit(script)

    # D. re-runs with and without mutated test y
    if args.skip_reruns:
        report["D_test_y_mutation"] = {"skipped": True, "passed": None}
    else:
        report["D_test_y_mutation"] = mutation_check(script, data, y, pred)

    checks = ["A_rmse_provenance", "B_continuous_u_only_rollout", "C_free_run_source_audit", "D_test_y_mutation"]
    report["passed"] = all(report[c]["passed"] is not False for c in checks)

    (run_dir / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    b = report["B_continuous_u_only_rollout"]
    summary["verification"] = {
        "passed": report["passed"],
        "checks": {c: report[c]["passed"] for c in checks},
        "rk4_steps_replayed_from_u_only": b.get("rk4_steps_checked"),
        "steps_failing": b.get("steps_failing"),
        "boundary": b.get("boundary"),
        "free_run_signature": report["C_free_run_source_audit"].get("signature"),
        "free_run_num_loops": report["C_free_run_source_audit"].get("num_loops"),
        "free_run_boundary_branches": report["C_free_run_source_audit"].get("boundary_if_branches"),
        "details": "verification.json",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print("VERIFICATION", "PASSED" if report["passed"] else "FAILED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
