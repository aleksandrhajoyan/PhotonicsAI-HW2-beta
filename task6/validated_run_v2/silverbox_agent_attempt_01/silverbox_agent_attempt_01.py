import os
import random
import json
import time
import copy
from pathlib import Path

import numpy as np
import scipy.io
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Reproducibility and CPU-only execution.
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
    """Recursively locate V1/V2 channel arrays in common scipy MAT formats."""
    if isinstance(obj, dict):
        lower_keys = {str(k).lower(): k for k in obj.keys()}
        if "v1" in lower_keys and "v2" in lower_keys:
            return obj[lower_keys["v1"]], obj[lower_keys["v2"]]
        for value in obj.values():
            try:
                result = find_channels(value)
                if result is not None:
                    return result
            except (TypeError, ValueError, RecursionError):
                pass

    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            lower_names = {name.lower(): name for name in obj.dtype.names}
            if "v1" in lower_names and "v2" in lower_names:
                return obj[lower_names["v1"]], obj[lower_names["v2"]]
            for name in obj.dtype.names:
                try:
                    result = find_channels(obj[name])
                    if result is not None:
                        return result
                except (TypeError, ValueError, RecursionError):
                    pass
        if obj.dtype == object:
            for value in obj.flat:
                try:
                    result = find_channels(value)
                    if result is not None:
                        return result
                except (TypeError, ValueError, RecursionError):
                    pass

    if hasattr(obj, "_fieldnames"):
        names = {name.lower(): name for name in obj._fieldnames}
        if "v1" in names and "v2" in names:
            return getattr(obj, names["v1"]), getattr(obj, names["v2"])
        for name in obj._fieldnames:
            try:
                result = find_channels(getattr(obj, name))
                if result is not None:
                    return result
            except (TypeError, ValueError, RecursionError):
                pass
    return None


def as_float_vector(x):
    return np.asarray(x, dtype=np.float64).squeeze().reshape(-1)


def inverse_softplus(x):
    x = max(float(x), 1e-6)
    if x > 20.0:
        return x
    return float(np.log(np.expm1(x)))


mat_path = os.environ.get("SILVERBOX_MAT_PATH")
if not mat_path:
    raise RuntimeError("SILVERBOX_MAT_PATH is not set")
if not Path(mat_path).is_file():
    raise FileNotFoundError(mat_path)

mat = scipy.io.loadmat(mat_path, simplify_cells=True)
channels = find_channels(mat)
if channels is None:
    raise KeyError("Could not locate V1 and V2 channels in the MAT file")

u_all = as_float_vector(channels[0])
y_all = as_float_vector(channels[1])
u = u_all[42650:46746].copy()
y = y_all[42650:46746].copy()
if len(u) != N_TOTAL or len(y) != N_TOTAL:
    raise ValueError(
        f"Expected 4096 excerpt samples; obtained len(u)={len(u)}, len(y)={len(y)}"
    )

u -= np.mean(u)
y -= np.mean(y)
u_train = u[:N_TRAIN]
y_train = y[:N_TRAIN]

# SG derivatives and Duffing least-squares warm start use training data only.
sg_window = 31
y_sg = savgol_filter(y_train, sg_window, 3, deriv=0, delta=DT, mode="interp")
v_sg = savgol_filter(y_train, sg_window, 3, deriv=1, delta=DT, mode="interp")
a_sg = savgol_filter(y_train, sg_window, 3, deriv=2, delta=DT, mode="interp")
design = np.column_stack((-v_sg, -y_sg, -(y_sg ** 3), u_train))
duffing, *_ = np.linalg.lstsq(design, a_sg, rcond=None)
a0, b0, c0, g0 = [float(x) for x in duffing]

sy = max(float(np.sqrt(np.mean(y_train ** 2))), 1e-8)
su = max(float(np.sqrt(np.mean(u_train ** 2))), 1e-8)
sv = max(float(np.sqrt(np.mean(v_sg ** 2))), 1e-8)
sa = sy * (sv / sy) ** 2
d0 = a0 * sv / sa
k0 = b0 * sy / sa
q0 = c0 * sy ** 3 / sa
r0 = g0 * su / sa

print(
    "Data loaded and centered. "
    f"Excerpt={N_TOTAL}, train={N_TRAIN}, test={N_TOTAL-N_TRAIN}, "
    f"dt={DT:.9g} s; scales sy={sy:.6g}, sv={sv:.6g}, "
    f"su={su:.6g}, sa={sa:.6g}",
    flush=True,
)
print(
    "Duffing warm start (physical a,b,c,g)="
    f"({a0:.7g}, {b0:.7g}, {c0:.7g}, {g0:.7g})",
    flush=True,
)


class ResidualDuffing(nn.Module):
    """Normalized residual network plus trainable normalized Duffing terms."""

    def __init__(self):
        super().__init__()
        self.raw_d = nn.Parameter(torch.tensor(inverse_softplus(d0), dtype=torch.float32))
        self.raw_k = nn.Parameter(torch.tensor(inverse_softplus(k0), dtype=torch.float32))
        self.q = nn.Parameter(torch.tensor(q0, dtype=torch.float32))
        self.r = nn.Parameter(torch.tensor(r0, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def coefficients(self):
        return torch.nn.functional.softplus(self.raw_d), torch.nn.functional.softplus(self.raw_k)

    def normalized_accel(self, y, v, inp):
        Y = y / sy
        V = v / sv
        U = inp / su
        d, k = self.coefficients()
        features = torch.stack((Y, V, U), dim=-1)
        residual = self.net(features).squeeze(-1)
        return -d * V - k * Y - self.q * (Y ** 3) + self.r * U + residual

    def forward(self, y, v, inp):
        return sa * self.normalized_accel(y, v, inp)


def rk4_batch(model, y_state, v_state, inp, dt):
    # Classical RK4 with zero-order-held input at all four stages.
    k1y = v_state
    k1v = model(y_state, v_state, inp)

    k2y = v_state + 0.5 * dt * k1v
    k2v = model(y_state + 0.5 * dt * k1y, v_state + 0.5 * dt * k1v, inp)

    k3y = v_state + 0.5 * dt * k2v
    k3v = model(y_state + 0.5 * dt * k2y, v_state + 0.5 * dt * k2v, inp)

    k4y = v_state + dt * k3v
    k4v = model(y_state + dt * k3y, v_state + dt * k3v, inp)

    y_next = y_state + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    v_next = v_state + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return y_next, v_next


def free_run(accel, u, y0, v0):
    """One uninterrupted free run; deliberately has no measured-output access."""
    y_state, v_state = (float(y0), float(v0))
    states = np.empty((len(u), 2), dtype=np.float64)
    states[0] = (y_state, v_state)
    for k in range(len(u) - 1):
        uk = u[k]
        k1y = v_state
        k1v = accel(y_state, v_state, uk)

        k2y = v_state + 0.5 * DT * k1v
        k2v = accel(y_state + 0.5 * DT * k1y, v_state + 0.5 * DT * k1v, uk)

        k3y = v_state + 0.5 * DT * k2v
        k3v = accel(y_state + 0.5 * DT * k2y, v_state + 0.5 * DT * k2v, uk)

        k4y = v_state + DT * k3v
        k4v = accel(y_state + DT * k3y, v_state + DT * k3v, uk)

        y_state, v_state = (
            y_state + (DT / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
            v_state + (DT / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
        )
        states[k + 1] = (y_state, v_state)
    return states


model = ResidualDuffing().to(DEVICE)
u_t = torch.tensor(u_train, dtype=torch.float32, device=DEVICE)
y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
v_t = torch.tensor(v_sg, dtype=torch.float32, device=DEVICE)
d0_t = torch.tensor(d0, dtype=torch.float32)
k0_t = torch.tensor(k0, dtype=torch.float32)
q0_t = torch.tensor(q0, dtype=torch.float32)
r0_t = torch.tensor(r0, dtype=torch.float32)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
rng = np.random.default_rng(SEED)
start_time = time.time()

# A compact, progressively longer rollout curriculum to stay within the CPU budget.
curriculum = [
    (64, 6, 12, 1e-3),
    (256, 6, 8, 1e-3),
    (512, 4, 6, 5e-4),
    (1024, 2, 4, 2e-4),
]

for length, batch_size, updates, lr in curriculum:
    for group in optimizer.param_groups:
        group["lr"] = lr
    model.train()
    for update in range(updates):
        starts_np = rng.integers(0, N_TRAIN - length + 1, size=batch_size)
        starts = torch.tensor(starts_np, dtype=torch.long, device=DEVICE)
        ys = y_t[starts]
        vs = v_t[starts]
        predictions = [ys]
        for j in range(length - 1):
            inp = u_t[starts + j]
            ys, vs = rk4_batch(model, ys, vs, inp, DT)
            predictions.append(ys)

        pred = torch.stack(predictions, dim=1)
        target_idx = starts[:, None] + torch.arange(length, device=DEVICE)[None, :]
        target = y_t[target_idx]
        rollout_loss = torch.mean(((pred - target) / sy) ** 2)

        d, k = model.coefficients()
        coefficient_penalty = (
            ((d - d0_t) / (abs(d0) + 1.0)) ** 2
            + ((k - k0_t) / (abs(k0) + 1.0)) ** 2
            + ((model.q - q0_t) / (abs(q0) + 1.0)) ** 2
            + ((model.r - r0_t) / (abs(r0) + 1.0)) ** 2
        )
        network_penalty = sum(torch.sum(p ** 2) for p in model.net.parameters())
        loss = rollout_loss + 1e-3 * coefficient_penalty + 1e-8 * network_penalty

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print(
        f"Completed rollout stage length={length}, updates={updates}, "
        f"last_loss={float(loss.detach()):.6g}, elapsed={time.time()-start_time:.1f}s",
        flush=True,
    )


class PhysicalAccel:
    """Scalar adapter for the physical-unit acceleration module."""

    def __init__(self, net):
        self.net = net

    def __call__(self, y_value, v_value, u_value):
        with torch.no_grad():
            yy = torch.tensor(float(y_value), dtype=torch.float32)
            vv = torch.tensor(float(v_value), dtype=torch.float32)
            uu = torch.tensor(float(u_value), dtype=torch.float32)
            return float(self.net(yy, vv, uu).item())


# Select between the warm start and trained candidate using continuous training
# free-run RMSE only. No test outputs are consulted during fitting or selection.
model.eval()
initial_model = ResidualDuffing().to(DEVICE)
initial_model.eval()
v0_train = float(v_sg[0])
initial_states = free_run(PhysicalAccel(initial_model), u_train, float(y_train[0]), v0_train)
initial_score = float(np.sqrt(np.mean((initial_states[:, 0] - y_train) ** 2)))

trained_states = free_run(PhysicalAccel(model), u_train, float(y_train[0]), v0_train)
trained_score = float(np.sqrt(np.mean((trained_states[:, 0] - y_train) ** 2)))

if np.isfinite(initial_score) and (
    not np.isfinite(trained_score) or initial_score < trained_score
):
    kept = initial_model
    selected_name = "Duffing warm start"
    selected_train_score = initial_score
else:
    kept = copy.deepcopy(model).eval()
    selected_name = "trained residual Duffing network"
    selected_train_score = trained_score

print(
    f"Training-only continuous-run selection: {selected_name}, "
    f"train RMSE={selected_train_score:.8g} V",
    flush=True,
)
print(
    "Model: 3x64 tanh fully connected residual acceleration network; "
    "train-only RMS scaling; trainable Duffing coefficients; AdamW; "
    "rollout lengths 64/256/512/1024; RK4 with ZOH input.",
    flush=True,
)


class DeploymentAccel(nn.Module):
    """Physical-unit acceleration, including all feature/output scaling."""

    def __init__(self, source):
        super().__init__()
        self.core = copy.deepcopy(source).eval()

    def forward(self, y_value, v_value, u_value):
        return self.core(y_value, v_value, u_value)


deployment = DeploymentAccel(kept).eval()
trace_inputs = (
    torch.zeros(3, dtype=torch.float32),
    torch.zeros(3, dtype=torch.float32),
    torch.zeros(3, dtype=torch.float32),
)
traced = torch.jit.trace(deployment, trace_inputs, check_trace=True)
traced.save("final_accel.pt")

# Required single final full-record free run. Initial velocity is a forward
# finite difference at excerpt sample zero; the state is never reset.
y0 = float(y[0])
v0 = float((y[1] - y[0]) / DT)
final_states = free_run(PhysicalAccel(kept), u, y0, v0)
prediction = final_states[:, 0].copy()
train_rmse = float(np.sqrt(np.mean((prediction[:N_TRAIN] - y[:N_TRAIN]) ** 2)))
test_rmse = float(np.sqrt(np.mean((prediction[N_TRAIN:] - y[N_TRAIN:]) ** 2)))

np.save("freerun_states.npy", final_states.astype(np.float64))
np.save("freerun_prediction.npy", prediction.astype(np.float64))
with open("boundary_state.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "k_3071": final_states[3071].tolist(),
            "k_3072": final_states[3072].tolist(),
            "state_initialisations": 1,
            "rk4_steps": 4095,
            "rk4_u_convention": "zoh",
        },
        f,
        indent=2,
    )

test_time = np.arange(N_TRAIN, N_TOTAL, dtype=np.float64) * DT
plt.figure(figsize=(11, 4.5))
plt.plot(test_time, y[N_TRAIN:], label="Measured output", linewidth=1.2)
plt.plot(test_time, prediction[N_TRAIN:], label="Free-run prediction", linewidth=1.2)
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
print("Saved freerun_states.npy, freerun_prediction.npy, final_accel.pt,")
print("boundary_state.json, and test_measured_vs_predicted.png", flush=True)
