# PhotonicsAI Homework — Tasks 5 and 6

This repository contains the code for the two coding tasks.

## Results at a glance

| Result | Produced by | Train free-run RMSE | Test free-run RMSE |
|---|---|---:|---:|
| **Task 5 direct solution** | hand-written script `task5/task5_silverbox_pinn.py` | `0.0019888468 V` | **`0.0015988781 V`** |
| **Task 6 agent-generated validated solution** | executor-generated script in `task6/validated_run/` | `0.05250201921 V` | **`0.05948953519 V`** |

The two numbers come from different sources and must not be mixed: the Task 5
value was **not** produced by the Task 6 agentic system. The Task 6 value is
the real RMSE printed by the script the agent wrote and ran, independently
verified by `task6/verify_run.py`. It is only slightly better than the
approximately 0.06 V test RMS of a zero predictor.

## Structure

- `task5/task5_silverbox_pinn.py` — direct PyTorch/RK4 solution for Task 5.
- `task5/task5_results.json` — recorded result from the kept Task 5 run.
- `task5/task5_test_freerun.png` — measured vs. predicted test free run.
- `task6/agentic_system.py` — four-node LangGraph agentic system.
- `task6/task5_assignment.txt` — Task 5 prompt given to the Task 6 orchestrator.
- `task6/verify_run.py` — independent verifier for a Task 6 run directory.
- `task6/validated_run/`, `task6/validated_run_v2/` — evidence from the two real
  Task 6 runs (dialogue, scripts, logs, plots, summaries, verifier reports).
- `requirements.txt` — Python dependencies.

The benchmark data file `SNLS80mV.mat` is intentionally not committed. Download
it from the Silverbox benchmark source and pass its local path to the scripts.

## Task 5

Example:

```bash
python task5/task5_silverbox_pinn.py /path/to/SNLS80mV.mat --outdir task5_run --best-only
```

Verified Task 5 runs:

| Architecture | Optimizer | Train free-run RMSE | Test free-run RMSE |
|---|---|---:|---:|
| `3 -> 16 -> 16 -> 1` | Adam | `0.0047041010 V` | `0.0067999880 V` |
| `3 -> 32 -> 32 -> 1` | Adam | `0.0027092390 V` | `0.0029287520 V` |
| **`3 -> 32 -> 32 -> 32 -> 1`** | **AdamW** | **`0.0019888468 V`** | **`0.0015988781 V`** |

The three-hidden-layer Tanh model is the kept version.

See `task5/task5_results.json` for the complete recorded configurations.

## Task 6

The graph has exactly four role nodes:

1. `orchestrator`
2. `scientist`
3. `general`
4. `executor`

The shared state is a dialogue. The orchestrator uses an LLM call to choose the
next role. Specialist nodes append their turn and return control. The executor is
the only role that receives the coding tool.

The coding tool both writes a new `.py` file and runs it with a hard 300-second
timeout, returning exit code, stdout and stderr. The program refuses to accept
`finish` until a successfully executed script has printed a numeric `TEST_RMSE`.

Quality gate: a zero predictor scores about 0.06 V test RMSE. While the best
real `TEST_RMSE` is not below 0.02 V and at least two of the 8 orchestrator
decisions would remain, a `finish` choice is rejected and sent back to the
orchestrator LLM with the reason; the LLM then decides itself whether the
scientist (diagnose and re-plan) or the executor (implement in a new script)
goes next. With fewer decisions left, it may finish with the best real result.
The kept attempt is the successful one with the lowest complete free-run
`TEST_RMSE`, which the assignment allows.

### Install

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### API key

Do **not** put an API key in this repository.

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
```

macOS/Linux:

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

The model defaults to `gpt-6-luna`; override with `--model <name>` or the
`AGENT_MODEL` environment variable.

### Validate Task 6 on Task 5

Put the Silverbox files in `SilverboxFiles/` (not committed). The script finds
`SNLS80mV.mat` there automatically; `--data` overrides it.

```bash
python task6/agentic_system.py --workdir task6_runs/my_run
python task6/verify_run.py task6_runs/my_run
```

Without `--workdir`, a fresh `task6_runs/run_<timestamp>/` directory is used.

The program streams the role-by-role dialogue to the terminal and writes into
the work directory:

- `dialogue.txt`, `dialogue.json` — the full shared dialogue (saved after every turn)
- `silverbox_agent_attempt_NN/` — one directory per executor attempt (never
  overwritten), in which the script runs, containing:
- `silverbox_agent_attempt_NN.py` — the executor-generated script
- `silverbox_agent_attempt_NN.stdout.txt` / `.stderr.txt` / `.traceback.txt`
- files written by a successful generated script, per the executor's free-run contract:
  - `freerun_states.npy` — the (4096, 2) state trajectory (y, v) of the final free run
  - `freerun_prediction.npy` — its y column
  - `final_accel.pt` — the kept acceleration network a_theta(y, v, u), TorchScript
  - `boundary_state.json` — state at samples 3071 and 3072, number of state
    initialisations, number of RK4 steps
  - the measured-vs-predicted test plot
- `run_summary.json` — turn sequence, orchestrator decisions, executor runs with
  their printed RMSE values, kept script, exit code, `TRAIN_RMSE`, `TEST_RMSE`,
  boundary state; after verification also a `verification` block

(`task6/validated_run/` predates the per-attempt directories; its single
attempt's files sit directly in the run directory. The verifier handles both.)

The generated script must do its final evaluation with a function
`free_run(accel, u, y0, v0)`. That function never receives measured y, sets the
state once, and runs one loop over all 4095 RK4 steps with no train/test branch.
As the assignment allows, the script may compare complete free runs by test RMSE
to choose which version to keep. What it may not do is read measured y inside
the free-run dynamics or reset the state at the boundary.

`task6/verify_run.py` independently checks the run and writes `verification.json`:

- **A. RMSE provenance:** recomputes TRAIN/TEST RMSE from `freerun_prediction.npy`
  and the real data and compares them with the printed `TRAIN_RMSE` / `TEST_RMSE`.
- **B. Continuous, u-only rollout:** using `final_accel.pt` and the input u only,
  it recomputes every one of the 4095 RK4 steps of `freerun_states.npy`
  (`rk4(state[k], u[k], u[k+1]) == state[k+1]`). It also checks that `state[0]` is
  measured y[0] with a finite-difference velocity, and reports the 3071 → 3072
  boundary step explicitly. A reset or any use of measured y at any step breaks
  that step's agreement.
- **C. Source audit:** parses `free_run` and reports its signature and source,
  its loops, its `if` branches, every state assignment, and every external name it
  reads. It fails on more than one loop, a boundary branch, the constants
  3071/3072, or a name that looks like measured y.
- **D. Test-y mutation:** re-runs the script unchanged, then on a copy of
  `SNLS80mV.mat` whose measured test output is replaced by a mean-preserving
  mirror image plus a ±0.02 V alternating offset. If the same weights are kept,
  the prediction must be identical. If test-score selection keeps a different
  version, which is allowed, check B must still pass on the mutated run.

Check B was validated with negative controls: a script with an explicit
`if k + 1 == 3072` reset, and one with a hidden branch-free reset from measured y.
Both are caught, with exactly one failing step, the boundary step.

For the homework PDF, use the real dialogue log and the `FINAL_TEST_RMSE` from
your actual Task 6 run. Do not copy a value from Task 5 and present it as an
agent-generated Task 6 result.

### Task 6 validation results

Both runs used model `gpt-6-luna`, the unchanged Task 5 prompt
(`task6/task5_assignment.txt`) and `SNLS80mV.mat`. Model checkpoints (`*.pt`) and
NumPy arrays (`*.npy`) are not committed; everything else in the run directories is.

**Run 1 — `task6/validated_run/` (before the quality gate)**

- Sequence: user → orchestrator → scientist → orchestrator → executor → orchestrator
  (decisions: scientist, executor, finish)
- One executor attempt, exit code 0: `TRAIN_RMSE = 0.05250201921 V`,
  **`TEST_RMSE = 0.05948953519 V`**
- `verify_run.py`: A, B, C, D all passed.
- Barely better than the ~0.06 V zero predictor, which motivated the quality gate.

**Run 2 — `task6/validated_run_v2/` (with the quality gate)**

- Sequence: user → orchestrator → scientist → orchestrator → executor →
  orchestrator → scientist → orchestrator → executor → orchestrator → scientist →
  orchestrator → executor → orchestrator → executor → orchestrator
  (decisions: scientist, executor, scientist, executor, scientist, executor,
  executor, finish — all 8 used; finish only at decision 8)

| Attempt | Exit code | Time (s) | TRAIN_RMSE (V) | TEST_RMSE (V) |
|---|---:|---:|---:|---:|
| 01 | 0 | 40.8 | 0.05646572659 | **0.06048733984** (kept) |
| 02 | 0 | 154.9 | 0.05847274015 | 0.06052074549 |
| 03 | 0 | 36.6 | nan | nan (diverged; not a valid result) |
| 04 | 0 | 24.9 | nan | nan (diverged; not a valid result) |

- Kept: attempt 01, the successful attempt with the lowest test RMSE.
- `verify_run.py` on the kept attempt: A, B, C, D all passed.
- The improvement loop did not reach the 0.02 V target; the run finished at the
  end of its decision budget.

Run 2 summary: 8 orchestrator decisions, executor attempts 01-04, best v2
`TEST_RMSE = 0.06048733984 V`, which does not beat run 1. Run 2 is supplementary
evidence that the loop reacts to a weak result (re-planning and new scripts);
run 1 remains the best Task 6 result.

**Final Task 6 result**

The best real, verified Task 6 result is run 1 (`task6/validated_run/`,
script `silverbox_agent_attempt_01.py`, exit code 0, 74.3 s):
**`TRAIN_RMSE = 0.05250201921 V`, `TEST_RMSE = 0.05948953519 V`**. Neither run
produced a model clearly better than a zero predictor. This is the honest
agent-generated result; the 0.0015988781 V model of Task 5 was produced
directly, not by the agent.

**About the committed evidence**

- Committed per run: `dialogue.txt`, `dialogue.json`, `run_summary.json`, every
  generated script with its stdout/stderr logs, `boundary_state.json`, the test
  plots, and the verifier report `verification.json`.
- Not committed: model checkpoints (`*.pt`), NumPy arrays (`*.npy`), the
  benchmark data (`SNLS80mV.mat`, `SilverboxFiles/`) and any API key.
- In `run_summary.json` and `verification.json` only the absolute local path
  strings were rewritten after the run (repository-relative paths, `<TEMP>`,
  `<PYTHON>`); no number, result or other field changed. Dialogues, scripts and
  logs are unmodified raw output.

**Reproduce / verify**

Checks A-C need the run's `*.pt` / `*.npy` files, which are not committed. The
kept scripts are deterministic (fixed seeds, one thread; check D reproduced the
recorded prediction exactly), so running the kept script inside its own
directory with `SILVERBOX_MAT_PATH` pointing to `SNLS80mV.mat` regenerates them.
Run the commands from the repository root:

```bash
# Re-verify the recorded runs:
python task6/verify_run.py task6/validated_run
python task6/verify_run.py task6/validated_run_v2

# A new run, in a fresh directory:
python task6/agentic_system.py --workdir task6_runs/repro
python task6/verify_run.py task6_runs/repro
```

LLM outputs vary, so a new run produces a different dialogue and RMSE.
`verify_run.py` re-runs the kept script twice (check D), which reproduces the
recorded prediction exactly because the scripts fix all seeds and use one thread.

---

# Homework 2 — Human-in-the-loop agents and LLM-controlled simulated focusing

Homework 2 extends the four-role LangGraph system in `task6/agentic_system.py`
(orchestrator, scientist, general, executor), which it imports without
modifying. Code is in `hw2/`, tests in `hw2/tests/`.

## Task 1 — Human-in-the-loop (`hw2/task1_hitl.py`)

- The orchestrator LLM may route to a `human` destination, using real LangGraph
  `interrupt()` calls. The paused thread resumes on the same `thread_id` with
  `Command(resume=...)`.
- The executor's coding tool runs only after a human approves the exact prepared
  action (`action_id` plus the code's SHA-256). A refused action is never executed,
  and an approved action runs exactly once.
- Demonstrations:
  - `hw2/demo_task1.py`: an ambiguous question, then a clarification.
  - `hw2/demo_task1_hard.py`: a refusal, then a fresh approval.

## Task 2 — Simulated optical focusing (`hw2/simulated_bench.py`, `hw2/focusing_optimizer.py`, `hw2/focusing_tool.py`, `hw2/task2_agent.py`)

- **Simulated optics, not physical hardware:** laser → binary-amplitude DMD →
  one fixed complex Gaussian transmission matrix *T* → camera with Poisson shot
  noise and Gaussian read noise. No hardware driver, camera SDK or DMD driver is
  used.
- **Optimizer:** a black-box binary flipping search that uses only camera
  measurements. Results are the means of independent validation measurements;
  the noisy search maxima are reported only as biased search values.
- **Agent tool:** in Task 2 the executor's only tool is the bounded
  `FocusingSession.call(arguments)`. The executor LLM chooses all 13 experiment
  settings. There is no code execution, and all experiments of one run share the
  same medium.

## Requirements and installation

Python 3.12 was used. Install the dependencies listed in `requirements.txt`
(HW2 uses `langgraph`, `openai`, `numpy` and `matplotlib`):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Offline tests (no API key, no network)

```powershell
python -m unittest discover -s hw2/tests -v
```

The tests use scripted or fake models and never call the OpenAI API. Run them
from a git checkout: a few tests check that the HW1 files are unchanged.

## Offline simulated demo (mocked LLM, real simulator)

```powershell
python -m hw2.task2_agent --offline-demo
```

The LLM replies are scripted and clearly labelled as mocked; all measured values
come from the real simulated focusing tool.

## Real LLM demos (paid OpenAI API calls)

Set `OPENAI_API_KEY` in your shell environment; never put it in a file or in
this repository:

```powershell
$env:OPENAI_API_KEY = "<your key>"
python -m hw2.demo_task2_live --model gpt-6-luna --max-api-calls 30
python -m hw2.demo_task1
python -m hw2.demo_task1_hard --data <path to SNLS80mV.mat>
```

- `demo_task2_live` is bounded:
  - session: 16×16 DMD, 32×32 camera, medium seed 7;
  - at most 3 completed experiments and 4000 camera acquisitions;
  - at most 8 orchestrator decisions and the `--max-api-calls` cap.
- **Silverbox data:** the Task 1 code-execution demo (`demo_task1_hard`) and
  `python -m hw2.task1_hitl` need the Silverbox benchmark file `SNLS80mV.mat`.
  It is **not distributed in this repository**; download it from the benchmark
  source and pass `--data`.

## Where results are stored

Every run writes to a new directory under `task6_runs/`:
- `hw2_task1_ambiguity_<timestamp>/`
- `hw2_task1_hard_<timestamp>/`
- `hw2_task2_live_<timestamp>/`
- `hw2_task2_offline_<timestamp>/`

These directories contain dialogues, run records, experiment tables and focusing
artifacts. A run refuses to overwrite an existing non-empty directory.
`task6_runs/` is git-ignored, so local run evidence is not part of this repository.

## Report and video

The Homework 2 PDF report and the screen recording are submitted separately
with the homework submission; they are not stored in this repository.
