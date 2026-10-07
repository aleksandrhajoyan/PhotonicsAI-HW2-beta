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
DEVICE = torch.device("cpu")

FS = 610.35
DT = 1.0 / FS
N_TRAIN = 3072
N_TOTAL = 4096


def find_channels(obj):
    """Find V1 and V2 in common scipy.io.loadmat structures."""
    if isinstance(obj, dict):
        keys = {str(key).lower(): key for key in obj}
        if "v1" in keys and "v2" in keys:
            return obj[keys["v1"]], obj[keys["v2"]]
        for value in obj.values():
            result = find_channels(value)
            if result is not None:
                return result

    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            names = {name.lower(): name for name in obj.dtype.names}
            if "v1" in names and "v2" in names:
                return obj[names["v1"]], obj[names["v2"]]
            for name in obj.dtype.names:
                result = find_channels(obj[name])
                if result is not None:
                    return result
        elif obj.dtype == object:
            for value in obj.flat:
                result = find_channels(value)
                if result is not None:
                    return result

    if hasattr(obj, "_fieldnames"):
        names = {name.lower(): name for name in obj._fieldnames}
        if "v1" in names and "v2" in names:
            return getattr(obj, names["v1"]), getattr(obj, names["v2"])
        for name in obj._fieldnames:
            result = find_channels(getattr(obj, name))
            if result is not None:
                return result

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

u = as_float_vector(channels[0])[42650:46746].copy()
y = as_float_vector(channels[1])[42650:46746].copy()
if len(u) != N_TOTAL or len(y) != N_TOTAL:
    raise ValueError(f"Expected 4096 excerpt samples, got u={len(u)}, y={len(y)}")

# Center each signal using the mean of the complete required excerpt.
u -= np.mean(u)
y -= np.mean(y)
u_train_np = u[:N_TRAIN].copy()
y_train_np = y[:N_TRAIN].copy()

# All scaling and derivative estimates are training-only.
sg_window = 31
y_sg = savgol_filter(
    y_train_np, sg_window, 3, deriv=0, delta=DT, mode="interp"
)
v_sg = savgol_filter(
    y_train_np, sg_window, 3, deriv=1, delta=DT, mode="interp"
)
a_sg = savgol_filter(
    y_train_np, sg_window, 3, deriv=2, delta=DT, mode="interp"
)
design = np.column_stack((-v_sg, -y_sg, -(y_sg ** 3), u_train_np))
ls_coefficients, *_ = np.linalg.lstsq(design, a_sg, rcond=None)
a_ls, b_ls, c_ls, g_ls = [float(value) for value in ls_coefficients]

sy = max(float(np.sqrt(np.mean(y_train_np ** 2))), 1e-8)
su = max(float(np.sqrt(np.mean(u_train_np ** 2))), 1e-8)
sv = max(float(np.sqrt(np.mean(v_sg ** 2))), 1e-8)
sa = sv * sv / sy

# Normalized Duffing form: Y_dot = (sv/sy)V and
# V_dot = sa[-d V - k Y - q Y^3 + r U + residual].
d_ls = a_ls * sv / sa
k_ls = b_ls * sy / sa
q_ls = c_ls * sy**3 / sa
r_ls = g_ls * su / sa

# Damping and linear stiffness are constrained positive. The smoothed
# derivative regression supplies a starting point, not the final fit.
d_start = max(d_ls, 1e-3)
k_start = max(k_ls, 1e-3)
q_start = q_ls
r_start = r_ls

print(
    f"Data: excerpt=[42650:46746], samples={N_TOTAL}, train={N_TRAIN}, "
    f"test={N_TOTAL-N_TRAIN}, fs={FS:.2f} Hz, dt={DT:.10g} s",
    flush=True,
)
print(
    f"Training-only RMS scales: sy={sy:.7g} V, sv={sv:.7g} V/s, "
    f"su={su:.7g} V, sa={sa:.7g} V/s^2",
    flush=True,
)
print(
    "Smoothed-derivative physical Duffing LS start "
    f"(a,b,c,g)=({a_ls:.7g}, {b_ls:.7g}, {c_ls:.7g}, {g_ls:.7g})",
    flush=True,
)


def inverse_softplus(value):
    value = max(float(value), 1e-6)
    if value > 20.0:
        return value
    return float(np.log(np.expm1(value)))


class ResidualDuffing(nn.Module):
    """Normalized Duffing oscillator plus a small residual acceleration MLP."""

    def __init__(self):
        super().__init__()
        self.raw_d = nn.Parameter(
            torch.tensor(inverse_softplus(d_start), dtype=torch.float32)
        )
        self.raw_k = nn.Parameter(
            torch.tensor(inverse_softplus(k_start), dtype=torch.float32)
        )
        self.q = nn.Parameter(torch.tensor(q_start, dtype=torch.float32))
        self.r = nn.Parameter(torch.tensor(r_start, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def coefficients(self):
        return F.softplus(self.raw_d), F.softplus(self.raw_k)

    def residual_normalized(self, y_value, v_value, input_value):
        features = torch.stack(
            (y_value / sy, v_value / sv, input_value / su), dim=-1
        )
        return self.net(features).squeeze(-1)

    def forward(self, y_value, v_value, input_value):
        d, k = self.coefficients()
        normalized_acceleration = (
            -d * (v_value / sv)
            - k * (y_value / sy)
            - self.q * (y_value / sy) ** 3
            + self.r * (input_value / su)
            + self.residual_normalized(y_value, v_value, input_value)
        )
        return sa * normalized_acceleration


def rk4_batch(model, y_state, v_state, input_value):
    """Differentiable classical RK4 step with ZOH input at all stages."""
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
    """Uninterrupted ZOH-input RK4 rollout; this function has no y access."""
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

        k4y = v_state + DT * k3v
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
    """Adapter for the deployment model used by the prescribed free_run."""

    def __init__(self, model):
        self.model = model

    def __call__(self, y_value, v_value, input_value):
        with torch.no_grad():
            y_tensor = torch.tensor(float(y_value), dtype=torch.float32)
            v_tensor = torch.tensor(float(v_value), dtype=torch.float32)
            u_tensor = torch.tensor(float(input_value), dtype=torch.float32)
            return float(self.model(y_tensor, v_tensor, u_tensor).item())


def continuous_train_score(model):
    """Consistent sample-aligned training-only continuous free-run metric."""
    model.eval()
    initial_velocity = float((y_train_np[1] - y_train_np[0]) / DT)
    states = free_run(
        ScalarAccel(model),
        u_train_np,
        float(y_train_np[0]),
        initial_velocity,
    )
    if not np.isfinite(states).all():
        return float("inf")
    return float(np.sqrt(np.mean((states[:, 0] - y_train_np) ** 2)))


u_train_t = torch.tensor(u_train_np, dtype=torch.float32, device=DEVICE)
y_train_t = torch.tensor(y_train_np, dtype=torch.float32, device=DEVICE)
v_train_t = torch.tensor(v_sg, dtype=torch.float32, device=DEVICE)
torch_generator = torch.Generator(device="cpu")
torch_generator.manual_seed(SEED)
np_rng = np.random.default_rng(SEED)


def coefficient_penalty(model, reference):
    d, k = model.coefficients()
    d_ref, k_ref, q_ref, r_ref = reference
    return (
        ((d - d_ref) / max(abs(float(d_ref)), 0.1)) ** 2
        + ((k - k_ref) / max(abs(float(k_ref)), 0.1)) ** 2
        + ((model.q - q_ref) / max(abs(float(q_ref)), 0.1)) ** 2
        + ((model.r - r_ref) / max(abs(float(r_ref)), 0.1)) ** 2
    )


def rollout_loss(model, starts, length, include_velocity=True):
    """Batch training-only windows, aligned so state j predicts sample j."""
    y_state = y_train_t[starts]
    v_state = v_train_t[starts]
    predictions_y = [y_state]
    predictions_v = [v_state]

    offsets = torch.arange(length, dtype=torch.long, device=DEVICE)
    target_indices = starts[:, None] + offsets[None, :]
    inputs = u_train_t[target_indices]

    for j in range(length - 1):
        y_state, v_state = rk4_batch(model, y_state, v_state, inputs[:, j])
        predictions_y.append(y_state)
        predictions_v.append(v_state)

    pred_y = torch.stack(predictions_y, dim=1)
    target_y = y_train_t[target_indices]
    loss = torch.mean(((pred_y - target_y) / sy) ** 2)

    if include_velocity:
        pred_v = torch.stack(predictions_v, dim=1)
        target_v = v_train_t[target_indices]
        loss = loss + 0.025 * torch.mean(((pred_v - target_v) / sv) ** 2)
    return loss


def sample_starts(batch_size, length):
    return torch.tensor(
        np_rng.integers(0, N_TRAIN - length + 1, size=batch_size),
        dtype=torch.long,
        device=DEVICE,
    )


model = ResidualDuffing().to(DEVICE)
start_time = time.time()
reference = (
    F.softplus(model.raw_d.detach()).clone(),
    F.softplus(model.raw_k.detach()).clone(),
    model.q.detach().clone(),
    model.r.detach().clone(),
)

best_model = copy.deepcopy(model).eval()
best_score = continuous_train_score(best_model)
print(
    f"Initial Duffing candidate: continuous training RMSE={best_score:.8g} V",
    flush=True,
)


def consider_candidate(candidate, label):
    global best_model, best_score
    candidate.eval()
    score = continuous_train_score(candidate)
    print(
        f"Candidate {label}: continuous training free-run RMSE={score:.8g} V",
        flush=True,
    )
    if np.isfinite(score) and score < best_score:
        best_score = score
        best_model = copy.deepcopy(candidate).eval()
        print(f"Selected new training-only best: {best_score:.8g} V", flush=True)


# Duffing warm-start refinement by RK4 trajectory loss on training-only windows.
# The residual MLP is zero-initialized, so this stage fits the physical terms.
warm_optimizer = torch.optim.AdamW(
    [model.raw_d, model.raw_k, model.q, model.r],
    lr=2e-3,
    weight_decay=0.0,
)
warm_stages = [
    (128, 8, 18, 1.5e-3),
    (512, 4, 14, 1.0e-3),
    (1024, 2, 8, 5.0e-4),
]
warm_updates = 0
for length, batch_size, updates, learning_rate in warm_stages:
    for group in warm_optimizer.param_groups:
        group["lr"] = learning_rate
    last_loss = float("nan")
    for update in range(updates):
        model.train()
        starts = sample_starts(batch_size, length)
        loss = rollout_loss(model, starts, length, include_velocity=True)
        loss = loss + 1e-3 * coefficient_penalty(model, reference)
        warm_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [model.raw_d, model.raw_k, model.q, model.r], 0.75
        )
        warm_optimizer.step()
        last_loss = float(loss.detach())
        warm_updates += 1
        if (update + 1) % 5 == 0 or update + 1 == updates:
            print(
                f"WARM length={length} update={update+1}/{updates} "
                f"loss={last_loss:.7g} elapsed={time.time()-start_time:.1f}s",
                flush=True,
            )
    consider_candidate(model, f"Duffing warm-fit length={length}")

warm_reference = (
    F.softplus(model.raw_d.detach()).clone(),
    F.softplus(model.raw_k.detach()).clone(),
    model.q.detach().clone(),
    model.r.detach().clone(),
)

# Residual-network and physical-parameter training curriculum. Progressively
# longer windows emphasize free-running behavior over local derivative fit.
optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)
residual_stages = [
    (64, 8, 18, 7e-4),
    (256, 6, 14, 5e-4),
    (512, 4, 10, 3e-4),
    (1024, 2, 7, 2e-4),
    (2048, 1, 4, 1.5e-4),
]
residual_updates = 0
for length, batch_size, updates, learning_rate in residual_stages:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    last_loss = float("nan")
    for update in range(updates):
        model.train()
        starts = sample_starts(batch_size, length)
        loss = rollout_loss(model, starts, length, include_velocity=True)
        loss = loss + 7e-4 * coefficient_penalty(model, warm_reference)
        residual_penalty = sum(
            torch.sum(parameter ** 2) for parameter in model.net.parameters()
        )
        loss = loss + 1e-7 * residual_penalty

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.75)
        optimizer.step()
        last_loss = float(loss.detach())
        residual_updates += 1

        if (update + 1) % 4 == 0 or update + 1 == updates:
            print(
                f"RESIDUAL length={length} update={update+1}/{updates} "
                f"loss={last_loss:.7g} elapsed={time.time()-start_time:.1f}s",
                flush=True,
            )
    consider_candidate(model, f"residual length={length}")

# Fine-tune the whole prescribed continuous training trajectory from the
# prescribed initial state; never reset or access test outputs in training.
continuous_optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-4, weight_decay=1e-5
)
full_y0 = torch.tensor(float(y_train_np[0]), dtype=torch.float32)
full_v0 = torch.tensor(
    float((y_train_np[1] - y_train_np[0]) / DT), dtype=torch.float32
)
continuous_updates = 4
for update in range(continuous_updates):
    model.train()
    y_state, v_state = full_y0, full_v0
    predicted_y = [y_state]
    predicted_v = [v_state]
    for k in range(N_TRAIN - 1):
        y_state, v_state = rk4_batch(
            model, y_state, v_state, u_train_t[k]
        )
        predicted_y.append(y_state)
        predicted_v.append(v_state)

    pred_y = torch.stack(predicted_y)
    pred_v = torch.stack(predicted_v)
    loss = torch.mean(((pred_y - y_train_t) / sy) ** 2)
    loss = loss + 0.01 * torch.mean(((pred_v - v_train_t) / sv) ** 2)
    loss = loss + 7e-4 * coefficient_penalty(model, warm_reference)
    residual_penalty = sum(
        torch.sum(parameter ** 2) for parameter in model.net.parameters()
    )
    loss = loss + 1e-7 * residual_penalty

    continuous_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    continuous_optimizer.step()

    print(
        f"CONTINUOUS update={update+1}/{continuous_updates} "
        f"loss={float(loss.detach()):.7g} elapsed={time.time()-start_time:.1f}s",
        flush=True,
    )
    consider_candidate(model, f"continuous update={update+1}")

kept = copy.deepcopy(best_model).eval()
print(
    f"Selected checkpoint by continuous training-only free run: "
    f"RMSE={best_score:.10g} V",
    flush=True,
)
print(
    f"Training budget: Duffing rollout updates={warm_updates}, "
    f"residual rollout updates={residual_updates}, "
    f"continuous rollout updates={continuous_updates}",
    flush=True,
)
print(
    "Model: normalized Duffing acceleration plus zero-initialized residual "
    "2-hidden-layer, 32-unit tanh MLP; trainable positive damping and linear "
    "stiffness; AdamW with gradient clipping; classical RK4, ZOH input.",
    flush=True,
)


class DeploymentAccel(nn.Module):
    """Acceleration module with all input and output scaling included."""

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
traced = torch.jit.trace(deployment, trace_examples, check_trace=True)
traced.save("final_accel.pt")

# Exactly one final full-record free run for the kept checkpoint. Initialize
# once from measured y[0] and the forward finite difference; do not reset at
# the train/test boundary. free_run receives no measured-output array.
y0 = float(y[0])
v0 = float((y[1] - y[0]) / DT)
final_states = free_run(ScalarAccel(kept), u, y0, v0)
prediction = final_states[:, 0].copy()

# Sample alignment audit: state index k is compared with measured sample k.
train_rmse = float(
    np.sqrt(np.mean((prediction[:N_TRAIN] - y[:N_TRAIN]) ** 2))
)
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
plt.plot(test_time, y[N_TRAIN:], label="Measured output", linewidth=1.2)
plt.plot(
    test_time,
    prediction[N_TRAIN:],
    label="Continuous free-run prediction",
    linewidth=1.2,
)
plt.xlabel("Time from excerpt start (s)")
plt.ylabel("Centered output (V)")
plt.title(f"Silverbox test segment: free-run RMSE = {test_rmse:.6g} V")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("test_measured_vs_predicted.png", dpi=160)
plt.close()

print(f"TRAIN_RMSE={train_rmse:.10g}", flush=True)
print(f"TEST_RMSE={test_rmse:.10g}", flush=True)
print(
    "Saved freerun_states.npy, freerun_prediction.npy, final_accel.pt, "
    "boundary_state.json, and test_measured_vs_predicted.png",
    flush=True,
)
print("EXIT_CODE=0", flush=True)
print("ERRORS=none", flush=True)
