import copy
import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
import torch.nn.functional as F


# Reproducible, CPU-only execution.
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)

FS = 610.35
DT = 1.0 / FS
N_TRAIN = 3072
N_TOTAL = 4096
DEVICE = torch.device("cpu")
START_TIME = time.time()


def find_channels(obj):
    """Find channels V1 and V2 in common scipy.io.loadmat structures."""
    if isinstance(obj, dict):
        keys = {str(key).lower(): key for key in obj}
        if "v1" in keys and "v2" in keys:
            return obj[keys["v1"]], obj[keys["v2"]]
        for value in obj.values():
            found = find_channels(value)
            if found is not None:
                return found

    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            names = {name.lower(): name for name in obj.dtype.names}
            if "v1" in names and "v2" in names:
                return obj[names["v1"]], obj[names["v2"]]
            for name in obj.dtype.names:
                found = find_channels(obj[name])
                if found is not None:
                    return found
        elif obj.dtype == object:
            for value in obj.flat:
                found = find_channels(value)
                if found is not None:
                    return found

    if hasattr(obj, "_fieldnames"):
        names = {name.lower(): name for name in obj._fieldnames}
        if "v1" in names and "v2" in names:
            return getattr(obj, names["v1"]), getattr(obj, names["v2"])
        for name in obj._fieldnames:
            found = find_channels(getattr(obj, name))
            if found is not None:
                return found
    return None


def as_float_vector(value):
    return np.asarray(value, dtype=np.float64).squeeze().reshape(-1)


mat_path = os.environ.get("SILVERBOX_MAT_PATH")
if not mat_path:
    raise RuntimeError("SILVERBOX_MAT_PATH is not set")
if not Path(mat_path).is_file():
    raise FileNotFoundError(mat_path)

mat = scipy.io.loadmat(mat_path, simplify_cells=True)
channels = find_channels(mat)
if channels is None:
    raise KeyError("Could not locate V1 and V2 channels in the MAT file")

# Exact requested excerpt, centered on its own complete-excerpt means.
u = as_float_vector(channels[0])[42650:46746].copy()
y = as_float_vector(channels[1])[42650:46746].copy()
if len(u) != N_TOTAL or len(y) != N_TOTAL:
    raise ValueError(f"Expected 4096 samples, got u={len(u)}, y={len(y)}")
u -= np.mean(u)
y -= np.mean(y)

u_train = u[:N_TRAIN].copy()
y_train = y[:N_TRAIN].copy()
train_zero_rmse = float(np.sqrt(np.mean(y_train**2)))
test_zero_rmse = float(np.sqrt(np.mean(y[N_TRAIN:] ** 2)))

print(
    f"Data/alignment: exact excerpt=[42650:46746], n={N_TOTAL}, "
    f"train=[0:{N_TRAIN}], test=[{N_TRAIN}:{N_TOTAL}], "
    f"fs={FS:.2f} Hz, dt={DT:.12g} s; prediction index k matches measured k.",
    flush=True,
)
print(
    f"Centered signal checks: u_train RMS={np.sqrt(np.mean(u_train**2)):.8g} V, "
    f"y_train RMS={np.sqrt(np.mean(y_train**2)):.8g} V, "
    f"u_test RMS={np.sqrt(np.mean(u[N_TRAIN:]**2)):.8g} V, "
    f"y_test RMS={np.sqrt(np.mean(y[N_TRAIN:]**2)):.8g} V",
    flush=True,
)
print(
    f"Zero-predictor RMSE: train={train_zero_rmse:.10g} V, "
    f"test={test_zero_rmse:.10g} V",
    flush=True,
)

# Save a simple training-record signal/alignment diagnostic.
plt.figure(figsize=(11, 4.5))
t_train = np.arange(N_TRAIN) * DT
plt.plot(t_train, u_train, label="Centered input u", linewidth=0.8)
plt.plot(t_train, y_train, label="Centered output y", linewidth=0.8)
plt.xlabel("Time from excerpt start (s)")
plt.ylabel("Voltage (V)")
plt.title("Silverbox training excerpt: centered signals")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("training_signals.png", dpi=150)
plt.close()


# Training-only ARX diagnostic. Compare a few input lag conventions, then
# evaluate each recursively from the training record's initial history.
def arx_diagnostic():
    records = []
    ridge_grid = (1e-8, 1e-6, 1e-4, 1e-2)
    for input_lag in (0, 1, 2):
        first_k = max(2, input_lag)
        rows = []
        targets = []
        for k in range(first_k, N_TRAIN):
            rows.append(
                [
                    y_train[k - 1],
                    y_train[k - 2],
                    u_train[k - input_lag],
                    u_train[k - input_lag - 1]
                    if k - input_lag - 1 >= 0
                    else 0.0,
                ]
            )
            targets.append(y_train[k])
        X = np.asarray(rows, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        best = None
        for ridge in ridge_grid:
            coef = np.linalg.solve(
                X.T @ X + ridge * np.eye(X.shape[1]), X.T @ target
            )
            one_step = X @ coef
            one_step_rmse = float(np.sqrt(np.mean((one_step - target) ** 2)))

            pred = np.empty(N_TRAIN, dtype=np.float64)
            pred[:2] = y_train[:2]
            for k in range(2, N_TRAIN):
                j0 = k - input_lag
                j1 = j0 - 1
                input0 = u_train[j0] if j0 >= 0 else 0.0
                input1 = u_train[j1] if j1 >= 0 else 0.0
                pred[k] = (
                    coef[0] * pred[k - 1]
                    + coef[1] * pred[k - 2]
                    + coef[2] * input0
                    + coef[3] * input1
                )
                if not np.isfinite(pred[k]) or abs(pred[k]) > 1e4:
                    pred[k:] = np.nan
                    break
            recursive_rmse = (
                float(np.sqrt(np.mean((pred[2:] - y_train[2:]) ** 2)))
                if np.isfinite(pred[2:]).all()
                else float("inf")
            )
            candidate = (
                recursive_rmse,
                one_step_rmse,
                ridge,
                coef,
                input_lag,
                pred,
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
        records.append(best)

    print("Training-only ARX diagnostic (recursive RMSE; one-step RMSE):", flush=True)
    for recursive, one_step, ridge, _, lag, _ in records:
        print(
            f"  input lag={lag}: recursive={recursive:.8g} V, "
            f"one-step={one_step:.8g} V, ridge={ridge:g}",
            flush=True,
        )
    best = min(records, key=lambda item: item[0])
    np.save("arx_training_recursive_prediction.npy", best[5])
    print(
        f"Best ARX diagnostic: input lag={best[4]}, recursive training "
        f"RMSE={best[0]:.10g} V; diagnostic only, not used for test fitting.",
        flush=True,
    )


arx_diagnostic()


# Derivative targets are estimated only from the training output record.
SG_WINDOW = 31
SG_POLY = 3
y_smooth = savgol_filter(
    y_train, SG_WINDOW, SG_POLY, deriv=0, delta=DT, mode="interp"
)
v_smooth = savgol_filter(
    y_train, SG_WINDOW, SG_POLY, deriv=1, delta=DT, mode="interp"
)
a_smooth = savgol_filter(
    y_train, SG_WINDOW, SG_POLY, deriv=2, delta=DT, mode="interp"
)

# Scales are computed exclusively from training data.
sy = max(float(np.sqrt(np.mean(y_train**2))), 1e-8)
sv = max(float(np.sqrt(np.mean(v_smooth**2))), 1e-8)
su = max(float(np.sqrt(np.mean(u_train**2))), 1e-8)
sa = max(sv * sv / sy, 1e-8)

Y = y_smooth / sy
V = v_smooth / sv
U = u_train / su
A = a_smooth / sa

# Normalized Duffing initialization by derivative regression:
# A ~= -d V - k Y - q Y^3 + r U.
valid = slice(SG_WINDOW // 2, N_TRAIN - SG_WINDOW // 2)
duffing_design = np.column_stack(
    (-V[valid], -Y[valid], -(Y[valid] ** 3), U[valid])
)
duffing_target = A[valid]
duffing_coef, *_ = np.linalg.lstsq(
    duffing_design.T @ duffing_design + 1e-5 * np.eye(4),
    duffing_design.T @ duffing_target,
    rcond=None,
)
d_init, k_init, q_init, r_init = [float(x) for x in duffing_coef]
d_init = float(np.clip(d_init, 0.01, 20.0))
k_init = float(np.clip(k_init, 0.01, 100.0))
q_init = float(np.clip(q_init, -100.0, 100.0))
r_init = float(np.clip(r_init, -100.0, 100.0))

print(
    f"Training-only scales: sy={sy:.8g} V, sv={sv:.8g} V/s, "
    f"su={su:.8g} V, sa={sa:.8g} V/s^2",
    flush=True,
)
print(
    f"Derivative supervision: Savitzky-Golay window={SG_WINDOW}, "
    f"polynomial={SG_POLY}, interior valid indices "
    f"[{SG_WINDOW//2}:{N_TRAIN-SG_WINDOW//2}]",
    flush=True,
)
print(
    "Normalized Duffing derivative-fit initialization "
    f"(d,k,q,r)=({d_init:.7g}, {k_init:.7g}, {q_init:.7g}, {r_init:.7g})",
    flush=True,
)


def inverse_softplus(value):
    value = max(float(value), 1e-6)
    return float(np.log(np.expm1(value)))


class AccelerationModel(nn.Module):
    """Normalized Duffing structure plus a 2x16 tanh residual acceleration net."""

    def __init__(self):
        super().__init__()
        self.raw_d = nn.Parameter(
            torch.tensor(inverse_softplus(d_init), dtype=torch.float32)
        )
        self.raw_k = nn.Parameter(
            torch.tensor(inverse_softplus(k_init), dtype=torch.float32)
        )
        self.q = nn.Parameter(torch.tensor(q_init, dtype=torch.float32))
        self.r = nn.Parameter(torch.tensor(r_init, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def normalized_residual(self, y_value, v_value, input_value):
        features = torch.stack(
            (y_value / sy, v_value / sv, input_value / su), dim=-1
        )
        return self.net(features).squeeze(-1)

    def forward(self, y_value, v_value, input_value):
        Y_value = y_value / sy
        V_value = v_value / sv
        U_value = input_value / su
        d = F.softplus(self.raw_d)
        k = F.softplus(self.raw_k)
        normalized_accel = (
            -d * V_value
            - k * Y_value
            - self.q * Y_value**3
            + self.r * U_value
            + self.normalized_residual(y_value, v_value, input_value)
        )
        return sa * normalized_accel


def rk4_batch(model, y_state, v_state, input_value):
    """Differentiable classical RK4 step, with ZOH input at all four stages."""
    k1y = v_state
    k1v = model(y_state, v_state, input_value)

    k2y = v_state + 0.5 * DT * k1v
    k2v = model(
        y_state + 0.5 * DT * k1y,
        v_state + 0.5 * DT * k1v,
        input_value,
    )

    k3y = v_state + 0.5 * DT * k2v
    k3v = model(
        y_state + 0.5 * DT * k2y,
        v_state + 0.5 * DT * k2v,
        input_value,
    )

    k4y = v_state + DT * k3v
    k4v = model(
        y_state + DT * k3y,
        v_state + DT * k3v,
        input_value,
    )

    y_next = y_state + (DT / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    v_next = v_state + (DT / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return y_next, v_next


def free_run(accel, u, y0, v0):
    """One uninterrupted ZOH-RK4 rollout; receives no measured output array."""
    y_state, v_state = float(y0), float(v0)
    states = np.empty((len(u), 2), dtype=np.float64)
    states[0] = (y_state, v_state)
    for k in range(len(u) - 1):
        uk = float(u[k])

        k1y = v_state
        k1v = accel(y_state, v_state, uk)

        k2y = v_state + 0.5 * DT * k1v
        k2v = accel(
            y_state + 0.5 * DT * k1y,
            v_state + 0.5 * DT * k1v,
            uk,
        )

        k3y = v_state + 0.5 * DT * k2v
        k3v = accel(
            y_state + 0.5 * DT * k2y,
            v_state + 0.5 * DT * k2v,
            uk,
        )

        k4y = v_state + DT * k3y
        k4v = accel(
            y_state + DT * k3y,
            v_state + DT * k3v,
            uk,
        )

        y_state, v_state = (
            y_state + (DT / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
            v_state + (DT / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
        )
        states[k + 1] = (y_state, v_state)
    return states


class ScalarAccel:
    """Float32 CPU adapter for the scalar RK4 rollout."""

    def __init__(self, model):
        self.model = model

    def __call__(self, y_value, v_value, input_value):
        with torch.no_grad():
            yt = torch.tensor(float(y_value), dtype=torch.float32)
            vt = torch.tensor(float(v_value), dtype=torch.float32)
            ut = torch.tensor(float(input_value), dtype=torch.float32)
            return float(self.model(yt, vt, ut).item())


u_train_t = torch.tensor(u_train, dtype=torch.float32, device=DEVICE)
y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
v_smooth_t = torch.tensor(v_smooth, dtype=torch.float32, device=DEVICE)
a_smooth_t = torch.tensor(a_smooth, dtype=torch.float32, device=DEVICE)

rng = np.random.default_rng(SEED)


def continuous_train_score(candidate):
    candidate.eval()
    initial_v = float((y_train[1] - y_train[0]) / DT)
    states = free_run(
        ScalarAccel(candidate),
        u_train,
        float(y_train[0]),
        initial_v,
    )
    if not np.isfinite(states).all() or np.max(np.abs(states[:, 0])) > 10.0:
        return float("inf"), states
    score = float(np.sqrt(np.mean((states[:, 0] - y_train) ** 2)))
    return score, states


model = AccelerationModel().to(DEVICE)
best_model = copy.deepcopy(model).eval()
best_score, best_train_states = continuous_train_score(best_model)
print(
    f"Initial candidate continuous training RMSE={best_score:.10g} V",
    flush=True,
)


def consider_candidate(candidate, label):
    global best_model, best_score, best_train_states
    score, states = continuous_train_score(candidate)
    print(f"Checkpoint {label}: continuous train RMSE={score:.10g} V", flush=True)
    # Only continuous training-run results affect checkpoint selection.
    if (
        np.isfinite(score)
        and score < best_score
        and score < train_zero_rmse
        and np.isfinite(states).all()
    ):
        best_score = score
        best_model = copy.deepcopy(candidate).eval()
        best_train_states = states.copy()
        np.save("training_freerun_states.npy", best_train_states)
        np.save("training_freerun_prediction.npy", best_train_states[:, 0])
        plt.figure(figsize=(11, 4.5))
        plt.plot(t_train, y_train, label="Measured training output", linewidth=1.0)
        plt.plot(
            t_train,
            best_train_states[:, 0],
            label="Continuous training free run",
            linewidth=1.0,
        )
        plt.xlabel("Time from excerpt start (s)")
        plt.ylabel("Centered output (V)")
        plt.title(f"Training-only continuous run: RMSE={score:.6g} V")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_free_run.png", dpi=150)
        plt.close()
        print(f"Accepted improved training-only checkpoint: {score:.10g} V", flush=True)


# Pointwise acceleration supervision on training-only smoothed derivatives.
# The normalized residual is regularized so it does not needlessly replace the
# Duffing structure.
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
supervised_updates = 240
for update in range(supervised_updates):
    batch = torch.from_numpy(
        rng.integers(0, N_TRAIN, size=512, endpoint=False)
    ).long()
    pred_a = model(y_train_t[batch], v_smooth_t[batch], u_train_t[batch]) / sa
    target_a = a_smooth_t[batch] / sa
    loss_acc = torch.mean((pred_a - target_a) ** 2)
    residual = model.normalized_residual(
        y_train_t[batch], v_smooth_t[batch], u_train_t[batch]
    )
    loss = loss_acc + 1e-3 * torch.mean(residual**2)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    if (update + 1) % 60 == 0:
        print(
            f"DERIVATIVE update={update+1}/{supervised_updates} "
            f"loss={float(loss.detach()):.7g} "
            f"elapsed={time.time()-START_TIME:.1f}s",
            flush=True,
        )

consider_candidate(model, "derivative-supervised")


def sample_starts(batch_size, length, include_near_start=False):
    if include_near_start and batch_size > 0:
        starts = [0]
        if batch_size > 1:
            starts.extend(
                rng.integers(0, N_TRAIN - length + 1, size=batch_size - 1).tolist()
            )
    else:
        starts = rng.integers(0, N_TRAIN - length + 1, size=batch_size).tolist()
    return torch.tensor(starts, dtype=torch.long, device=DEVICE)


def staged_rollout_loss(candidate, starts, length, acceleration_weight):
    y_state = y_train_t[starts]
    v_state = v_smooth_t[starts]
    pred_y = [y_state]
    pred_v = [v_state]
    offsets = torch.arange(length, dtype=torch.long, device=DEVICE)
    indices = starts[:, None] + offsets[None, :]
    inputs = u_train_t[indices]

    for j in range(length - 1):
        y_state, v_state = rk4_batch(candidate, y_state, v_state, inputs[:, j])
        pred_y.append(y_state)
        pred_v.append(v_state)

    pred_y = torch.stack(pred_y, dim=1)
    pred_v = torch.stack(pred_v, dim=1)
    target_y = y_train_t[indices]
    target_v = v_smooth_t[indices]
    position_loss = torch.mean(((pred_y - target_y) / sy) ** 2)
    velocity_loss = torch.mean(((pred_v - target_v) / sv) ** 2)

    # Also retain local acceleration supervision for sampled states.
    sampled_y = y_train_t[indices]
    sampled_v = v_smooth_t[indices]
    sampled_u = inputs
    accel_pred = candidate(sampled_y, sampled_v, sampled_u) / sa
    accel_target = a_smooth_t[indices] / sa
    accel_loss = torch.mean((accel_pred - accel_target) ** 2)
    residual = candidate.normalized_residual(sampled_y, sampled_v, sampled_u)
    return (
        position_loss
        + 0.01 * velocity_loss
        + acceleration_weight * accel_loss
        + 1e-4 * torch.mean(residual**2)
    )


# Progressive training-only rollout objectives. Every sampled rollout window
# is wholly inside the training record and starts from training measurements.
rollout_stages = [
    (32, 12, 20, 0.60, 3e-4),
    (128, 4, 12, 0.35, 2e-4),
    (512, 2, 6, 0.20, 1e-4),
]
rollout_updates = 0
for length, batch_size, updates, accel_weight, learning_rate in rollout_stages:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    last_loss = float("nan")
    for update in range(updates):
        starts = sample_starts(
            batch_size, length, include_near_start=(update % 2 == 0)
        )
        loss = staged_rollout_loss(model, starts, length, accel_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        last_loss = float(loss.detach())
        rollout_updates += 1
        if (update + 1) % 4 == 0 or update + 1 == updates:
            print(
                f"ROLLOUT length={length} update={update+1}/{updates} "
                f"loss={last_loss:.7g} elapsed={time.time()-START_TIME:.1f}s",
                flush=True,
            )
    consider_candidate(model, f"staged rollout length={length}")


# Fine-tune on the prescribed continuous training trajectory (no state reset
# inside the sequence); measured test outputs are not accessed here.
full_y0 = torch.tensor(float(y_train[0]), dtype=torch.float32)
full_v0 = torch.tensor(float((y_train[1] - y_train[0]) / DT), dtype=torch.float32)
continuous_optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-5, weight_decay=1e-5
)
continuous_updates = 2
for update in range(continuous_updates):
    y_state, v_state = full_y0, full_v0
    pred_y = [y_state]
    pred_v = [v_state]
    for k in range(N_TRAIN - 1):
        y_state, v_state = rk4_batch(model, y_state, v_state, u_train_t[k])
        pred_y.append(y_state)
        pred_v.append(v_state)
    pred_y = torch.stack(pred_y)
    pred_v = torch.stack(pred_v)

    loss = torch.mean(((pred_y - y_train_t) / sy) ** 2)
    loss = loss + 0.005 * torch.mean(((pred_v - v_smooth_t) / sv) ** 2)
    # A small derivative term discourages local acceleration drift.
    accel_pred = model(y_train_t, v_smooth_t, u_train_t) / sa
    loss = loss + 0.03 * torch.mean((accel_pred - a_smooth_t / sa) ** 2)

    continuous_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    continuous_optimizer.step()
    print(
        f"CONTINUOUS update={update+1}/{continuous_updates} "
        f"loss={float(loss.detach()):.7g} "
        f"elapsed={time.time()-START_TIME:.1f}s",
        flush=True,
    )
    consider_candidate(model, f"continuous update={update+1}")


kept = copy.deepcopy(best_model).eval()
print(
    f"Selected using continuous training-only free run: "
    f"RMSE={best_score:.10g} V; train zero baseline={train_zero_rmse:.10g} V",
    flush=True,
)
print(
    "Model: state=(y,v), y_dot=v, v_dot=sa[-dV-kY-qY^3+rU+f_theta(Y,V,U)]; "
    "normalized Duffing acceleration plus 2-hidden-layer, 16-unit tanh MLP, "
    "zero-initialized residual output; positive trainable damping/stiffness.",
    flush=True,
)
print(
    "Training: training-only Savitzky-Golay acceleration supervision, "
    "staged RK4 rollout lengths 32/128/512, then continuous training rollout; "
    "AdamW, derivative lr=3e-4, weight decay=1e-5, gradient clip=0.5; "
    f"updates={supervised_updates} derivative, {rollout_updates} staged, "
    f"{continuous_updates} continuous.",
    flush=True,
)
print(
    f"Scaling: sy={sy:.8g}, sv={sv:.8g}, su={su:.8g}, sa={sa:.8g}; "
    "float32 model evaluation, RK4 ZOH input.",
    flush=True,
)


class DeploymentAccel(nn.Module):
    """Traceable acceleration module with all scaling inside its forward."""

    def __init__(self, source):
        super().__init__()
        self.core = copy.deepcopy(source).eval()

    def forward(self, y_value, v_value, input_value):
        return self.core(y_value, v_value, input_value)


deployment = DeploymentAccel(kept).eval()
trace_examples = (
    torch.zeros(3, dtype=torch.float32),
    torch.zeros(3, dtype=torch.float32),
    torch.zeros(3, dtype=torch.float32),
)
torch.jit.trace(deployment, trace_examples, check_trace=True).save("final_accel.pt")

# Exactly one final full-excerpt run of the frozen kept model. It starts at
# measured y[0] and the sample-0 finite difference, and does not reset at the
# train/test boundary. free_run receives only the input and initial state.
y0 = float(y[0])
v0 = float((y[1] - y[0]) / DT)
final_states = free_run(ScalarAccel(kept), u, y0, v0)
prediction = final_states[:, 0].copy()

train_rmse = float(np.sqrt(np.mean((prediction[:N_TRAIN] - y[:N_TRAIN]) ** 2)))
test_rmse = float(
    np.sqrt(np.mean((prediction[N_TRAIN:] - y[N_TRAIN:]) ** 2))
)

np.save("freerun_states.npy", final_states.astype(np.float64))
np.save("freerun_prediction.npy", prediction.astype(np.float64))
with open("boundary_state.json", "w", encoding="utf-8") as output_file:
    json.dump(
        {
            "k_3071": final_states[3071].tolist(),
            "k_3072": final_states[3072].tolist(),
            "state_initialisations": 1,
            "rk4_steps": 4095,
            "rk4_u_convention": "zoh",
        },
        output_file,
        indent=2,
    )

test_time = np.arange(N_TRAIN, N_TOTAL, dtype=np.float64) * DT
plt.figure(figsize=(11, 4.5))
plt.plot(test_time, y[N_TRAIN:], label="Measured output", linewidth=1.1)
plt.plot(
    test_time,
    prediction[N_TRAIN:],
    label="Continuous free-run prediction",
    linewidth=1.1,
)
plt.xlabel("Time from excerpt start (s)")
plt.ylabel("Centered output (V)")
plt.title(f"Silverbox test segment: free-run RMSE={test_rmse:.6g} V")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("test_measured_vs_predicted.png", dpi=160)
plt.close()

print(f"TRAIN_RMSE={train_rmse:.10g}", flush=True)
print(f"TEST_RMSE={test_rmse:.10g}", flush=True)
print(f"EXIT_CODE=0", flush=True)
print("ERRORS=none", flush=True)
print(
    "Saved training_signals.png, ARX diagnostic, training free-run diagnostics "
    "when improved, test_measured_vs_predicted.png, freerun_states.npy, "
    "freerun_prediction.npy, final_accel.pt, and boundary_state.json.",
    flush=True,
)
print(f"TOTAL_RUNTIME_SECONDS={time.time()-START_TIME:.3f}", flush=True)
