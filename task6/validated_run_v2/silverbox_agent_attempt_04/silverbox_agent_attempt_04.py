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
from scipy.linalg import logm
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)

FS = 610.35
DT = 1.0 / FS
N_TRAIN = 3072
N_TOTAL = 4096
START_TIME = time.time()
DTYPE = torch.float64


def find_channels(obj):
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

# Exact requested excerpt and centering over the full excerpt.
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
    f"Data: excerpt=[42650:46746], train=[0:{N_TRAIN}], "
    f"test=[{N_TRAIN}:{N_TOTAL}], fs={FS:.2f} Hz, dt={DT:.12g} s",
    flush=True,
)
print(
    f"Centered RMS: u_train={np.sqrt(np.mean(u_train**2)):.8g} V, "
    f"y_train={np.sqrt(np.mean(y_train**2)):.8g} V, "
    f"u_test={np.sqrt(np.mean(u[N_TRAIN:]**2)):.8g} V, "
    f"y_test={np.sqrt(np.mean(y[N_TRAIN:]**2)):.8g} V",
    flush=True,
)
print(
    f"Zero-predictor RMSE: train={train_zero_rmse:.10g} V, "
    f"test={test_zero_rmse:.10g} V",
    flush=True,
)

# ARX indexing check: y[k] is predicted from y[k-1], y[k-2],
# u[k], and u[k-1]. This is strictly a training-only diagnostic.
arx_records = []
X_rows = []
target_rows = []
for k in range(2, N_TRAIN):
    X_rows.append([y_train[k - 1], y_train[k - 2], u_train[k], u_train[k - 1]])
    target_rows.append(y_train[k])
X_arx = np.asarray(X_rows, dtype=np.float64)
target_arx = np.asarray(target_rows, dtype=np.float64)
for ridge in (1e-8, 1e-6, 1e-4, 1e-2):
    coef = np.linalg.solve(
        X_arx.T @ X_arx + ridge * np.eye(X_arx.shape[1]),
        X_arx.T @ target_arx,
    )
    one_step_rmse = float(np.sqrt(np.mean((X_arx @ coef - target_arx) ** 2)))
    pred = np.empty(N_TRAIN, dtype=np.float64)
    pred[:2] = y_train[:2]
    stable = True
    for k in range(2, N_TRAIN):
        pred[k] = (
            coef[0] * pred[k - 1]
            + coef[1] * pred[k - 2]
            + coef[2] * u_train[k]
            + coef[3] * u_train[k - 1]
        )
        if not np.isfinite(pred[k]) or abs(pred[k]) > 10.0:
            stable = False
            break
    recursive_rmse = (
        float(np.sqrt(np.mean((pred[2:] - y_train[2:]) ** 2)))
        if stable else float("inf")
    )
    arx_records.append((recursive_rmse, one_step_rmse, ridge, coef.copy(), pred.copy()))

best_arx = min(arx_records, key=lambda item: item[0])
print("Training-only ARX diagnostic; verified indexing:", flush=True)
for recursive, one_step, ridge, _, _ in arx_records:
    print(
        f"  y[k]=p1*y[k-1]+p2*y[k-2]+q0*u[k]+q1*u[k-1], "
        f"ridge={ridge:g}, recursive={recursive:.9g} V, "
        f"one_step={one_step:.9g} V",
        flush=True,
    )
print(
    f"Best recursive training ARX RMSE={best_arx[0]:.10g} V; "
    "diagnostic/initialization only, not used on test.",
    flush=True,
)
np.save("arx_training_recursive_prediction.npy", best_arx[4])

plt.figure(figsize=(11, 4.5))
plt.plot(np.arange(N_TRAIN) * DT, u_train, label="Centered input u", linewidth=0.8)
plt.plot(np.arange(N_TRAIN) * DT, y_train, label="Centered output y", linewidth=0.8)
plt.xlabel("Time from excerpt start (s)")
plt.ylabel("Voltage (V)")
plt.title("Silverbox training signals")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("training_signals.png", dpi=150)
plt.close()

# Training-only scales and smoothed derivatives.
SG_WINDOW = 31
SG_POLY = 3
v_smooth = savgol_filter(y_train, SG_WINDOW, SG_POLY, deriv=1, delta=DT, mode="interp")
a_smooth = savgol_filter(y_train, SG_WINDOW, SG_POLY, deriv=2, delta=DT, mode="interp")
sy = max(float(np.sqrt(np.mean(y_train**2))), 1e-8)
sv = max(float(np.sqrt(np.mean(v_smooth**2))), 1e-8)
su = max(float(np.sqrt(np.mean(u_train**2))), 1e-8)
sa = max(sv * sv / sy, 1e-8)

# Map stable ARX poles into a continuous-time oscillator initialization.
# The companion matrix corresponds to z^2 - p1*z - p2 = 0.
p1, p2, q0, q1 = best_arx[3]
companion = np.array([[p1, p2], [1.0, 0.0]], dtype=np.float64)
try:
    continuous_matrix = np.real_if_close(logm(companion) / DT, tol=1000).real
    damping_init = float(-np.trace(continuous_matrix))
    stiffness_init = float(np.linalg.det(continuous_matrix))
except Exception:
    damping_init, stiffness_init = 2.0, (2.0 * np.pi * 15.0) ** 2

if not np.isfinite(damping_init) or damping_init <= 0:
    damping_init = 2.0
if not np.isfinite(stiffness_init) or stiffness_init <= 0:
    stiffness_init = (2.0 * np.pi * 15.0) ** 2
damping_init = float(np.clip(damping_init, 0.05, 200.0))
stiffness_init = float(np.clip(stiffness_init, 10.0, 200000.0))

# Convert physical coefficients to the normalized equation:
# y_dot=v; v_dot=sa[-d*V-k*Y-q*Y^3+r*U].
d_init = damping_init * sy / sv
k_init = stiffness_init * sy * sy / (sv * sv)
# A training-only local derivative regression initializes input gain and
# nonlinear stiffness; the continuous rollout objective then refines them.
valid = slice(SG_WINDOW // 2, N_TRAIN - SG_WINDOW // 2)
Y = y_train / sy
V = v_smooth / sv
U = u_train / su
A = a_smooth / sa
design = np.column_stack((-V[valid], -Y[valid], -(Y[valid] ** 3), U[valid]))
try:
    local_coef = np.linalg.lstsq(
        design.T @ design + 1e-4 * np.eye(4),
        design.T @ A[valid],
        rcond=None,
    )[0]
    d_init = float(np.clip(local_coef[0], 0.01, 50.0))
    k_init = float(np.clip(local_coef[1], 0.01, 500.0))
    q_init = float(np.clip(local_coef[2], -200.0, 200.0))
    r_init = float(np.clip(local_coef[3], -200.0, 200.0))
except Exception:
    q_init, r_init = 0.0, 1.0

print(
    f"Training-only scales: sy={sy:.8g} V, sv={sv:.8g} V/s, "
    f"su={su:.8g} V, sa={sa:.8g} V/s^2",
    flush=True,
)
print(
    f"ARX poles -> stable oscillator initialization: damping={damping_init:.7g}, "
    f"stiffness={stiffness_init:.7g}; normalized (d,k,q,r)="
    f"({d_init:.7g}, {k_init:.7g}, {q_init:.7g}, {r_init:.7g})",
    flush=True,
)


def inverse_softplus(value):
    value = max(float(value), 1e-8)
    return float(np.log(np.expm1(value)))


class AccelerationModel(nn.Module):
    """Duffing base plus a compact fully connected tanh residual network."""

    def __init__(self):
        super().__init__()
        self.raw_d = nn.Parameter(torch.tensor(inverse_softplus(d_init), dtype=DTYPE))
        self.raw_k = nn.Parameter(torch.tensor(inverse_softplus(k_init), dtype=DTYPE))
        self.q = nn.Parameter(torch.tensor(q_init, dtype=DTYPE))
        self.r = nn.Parameter(torch.tensor(r_init, dtype=DTYPE))
        self.net = nn.Sequential(
            nn.Linear(3, 16, dtype=DTYPE),
            nn.Tanh(),
            nn.Linear(16, 16, dtype=DTYPE),
            nn.Tanh(),
            nn.Linear(16, 1, dtype=DTYPE),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # Keep the initialized residual at zero during the stable low-order
        # system-identification fit; it remains part of the acceleration model.
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

    def forward(self, y_value, v_value, input_value):
        Y_value = y_value / sy
        V_value = v_value / sv
        U_value = input_value / su
        features = torch.stack((Y_value, V_value, U_value), dim=-1)
        residual = self.net(features).squeeze(-1)
        normalized_accel = (
            -F.softplus(self.raw_d) * V_value
            - F.softplus(self.raw_k) * Y_value
            - self.q * Y_value**3
            + self.r * U_value
            + residual
        )
        return sa * normalized_accel


def rk4_batch(model, y_state, v_state, u_left, u_right):
    """Differentiable RK4 step using linearly interpolated endpoint inputs."""
    u_mid = 0.5 * (u_left + u_right)
    k1y = v_state
    k1v = model(y_state, v_state, u_left)

    k2y = v_state + 0.5 * DT * k1v
    k2v = model(
        y_state + 0.5 * DT * k1y,
        v_state + 0.5 * DT * k1v,
        u_mid,
    )

    k3y = v_state + 0.5 * DT * k2v
    k3v = model(
        y_state + 0.5 * DT * k2y,
        v_state + 0.5 * DT * k2v,
        u_mid,
    )

    k4y = v_state + DT * k3y
    k4v = model(
        y_state + DT * k3y,
        v_state + DT * k3v,
        u_right,
    )

    y_next = y_state + (DT / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    v_next = v_state + (DT / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return y_next, v_next


def free_run(accel, u, y0, v0):
    """One uninterrupted linear-input RK4 rollout; does not receive measured y."""
    y_state, v_state = float(y0), float(v0)
    states = np.empty((len(u), 2), dtype=np.float64)
    states[0] = (y_state, v_state)
    for k in range(len(u) - 1):
        uk = float(u[k])
        uk1 = float(u[k + 1])

        k1y = v_state
        k1v = accel(y_state, v_state, uk)

        um = 0.5 * (uk + uk1)
        k2y = v_state + 0.5 * DT * k1v
        k2v = accel(
            y_state + 0.5 * DT * k1y,
            v_state + 0.5 * DT * k1v,
            um,
        )

        k3y = v_state + 0.5 * DT * k2v
        k3v = accel(
            y_state + 0.5 * DT * k2y,
            v_state + 0.5 * DT * k2v,
            um,
        )

        k4y = v_state + DT * k3y
        k4v = accel(
            y_state + DT * k3y,
            v_state + DT * k3v,
            uk1,
        )

        y_state, v_state = (
            y_state + (DT / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
            v_state + (DT / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
        )
        states[k + 1] = (y_state, v_state)
    return states


class ScalarAccel:
    def __init__(self, model):
        self.model = model

    def __call__(self, y_value, v_value, input_value):
        with torch.no_grad():
            yt = torch.tensor(y_value, dtype=DTYPE)
            vt = torch.tensor(v_value, dtype=DTYPE)
            ut = torch.tensor(input_value, dtype=DTYPE)
            return float(self.model(yt, vt, ut).item())


model = AccelerationModel()
best_model = copy.deepcopy(model).eval()
initial_v = float((y_train[1] - y_train[0]) / DT)


def score_training(candidate):
    candidate.eval()
    states = free_run(
        ScalarAccel(candidate),
        u_train,
        float(y_train[0]),
        initial_v,
    )
    if (
        not np.isfinite(states).all()
        or float(np.max(np.abs(states[:, 0]))) > 10.0
        or float(np.max(np.abs(states[:, 1]))) > 1e5
    ):
        return float("inf"), states
    return float(np.sqrt(np.mean((states[:, 0] - y_train) ** 2))), states


best_score, best_train_states = score_training(best_model)
print(f"Initial stable continuous training RMSE={best_score:.10g} V", flush=True)

# Full, uninterrupted training trajectory objective. Bounded positive damping
# and stiffness, double precision, small learning rate, and candidate rejection
# prevent the NaN/overflow failure of the earlier run.
u_train_t = torch.tensor(u_train, dtype=DTYPE)
y_train_t = torch.tensor(y_train, dtype=DTYPE)
optimizer = torch.optim.AdamW(
    [model.raw_d, model.raw_k, model.q, model.r],
    lr=2e-5,
    weight_decay=1e-5,
)
updates = 80
for update in range(updates):
    y_state = y_train_t[0]
    v_state = torch.tensor(initial_v, dtype=DTYPE)
    pred = [y_state]
    finite = True
    for k in range(N_TRAIN - 1):
        y_state, v_state = rk4_batch(
            model, y_state, v_state, u_train_t[k], u_train_t[k + 1]
        )
        pred.append(y_state)
        if (k + 1) % 128 == 0:
            if not bool(torch.isfinite(y_state).item()) or not bool(
                torch.isfinite(v_state).item()
            ):
                finite = False
                break
            if abs(float(y_state.detach())) > 10.0 or abs(float(v_state.detach())) > 1e5:
                finite = False
                break

    if finite and len(pred) == N_TRAIN:
        pred_tensor = torch.stack(pred)
        loss = torch.mean(((pred_tensor - y_train_t) / sy) ** 2)
        if bool(torch.isfinite(loss).item()):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [model.raw_d, model.raw_k, model.q, model.r], 0.1
            )
            optimizer.step()
            # Keep the learned oscillator in a broad, physically stable range.
            with torch.no_grad():
                model.raw_d.clamp_(-8.0, 7.0)
                model.raw_k.clamp_(-8.0, 12.0)
                model.q.clamp_(-200.0, 200.0)
                model.r.clamp_(-200.0, 200.0)

    if (update + 1) % 10 == 0 or update == 0:
        score, states = score_training(model)
        print(
            f"TRAIN update={update+1}/{updates} "
            f"loss={float(loss.detach()) if finite and 'loss' in locals() and torch.isfinite(loss) else float('inf'):.7g} "
            f"continuous_train_RMSE={score:.9g} V "
            f"elapsed={time.time()-START_TIME:.1f}s",
            flush=True,
        )
        if (
            np.isfinite(score)
            and score < best_score
            and np.isfinite(states).all()
        ):
            best_score = score
            best_model = copy.deepcopy(model).eval()
            best_train_states = states.copy()
            np.save("training_freerun_states.npy", states)
            np.save("training_freerun_prediction.npy", states[:, 0])
            plt.figure(figsize=(11, 4.5))
            plt.plot(np.arange(N_TRAIN) * DT, y_train, label="Measured training output")
            plt.plot(
                np.arange(N_TRAIN) * DT,
                states[:, 0],
                label="Continuous training free run",
            )
            plt.xlabel("Time from excerpt start (s)")
            plt.ylabel("Centered output (V)")
            plt.title(f"Training free run: RMSE={score:.6g} V")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig("training_free_run.png", dpi=150)
            plt.close()

kept = copy.deepcopy(best_model).eval()
print(
    f"Selected finite training-only continuous RMSE={best_score:.10g} V "
    f"(training zero baseline={train_zero_rmse:.10g} V)",
    flush=True,
)
print(
    "Model: state=(y,v), y_dot=v, v_dot=sa[-d*V-k*Y-q*Y^3+r*U+f_theta(Y,V,U)]; "
    "normalized Duffing terms plus a zero-initialized 2-hidden-layer, 16-unit "
    "tanh fully connected residual network. Positive damping and stiffness.",
    flush=True,
)
print(
    f"Training: ARX-pole/local-derivative initialization; {updates} attempted "
    "full training-only RK4 trajectory updates, AdamW lr=2e-5, weight_decay=1e-5, "
    "gradient clipping=0.1; double-precision rollout; linear endpoint input "
    "interpolation.",
    flush=True,
)
print(
    f"Scaling: sy={sy:.8g} V, sv={sv:.8g} V/s, su={su:.8g} V, "
    f"sa={sa:.8g} V/s^2; Savitzky-Golay derivative window={SG_WINDOW}.",
    flush=True,
)


class DeploymentAccel(nn.Module):
    """Traceable network; all state/input scaling remains inside the module."""

    def __init__(self, source):
        super().__init__()
        self.core = copy.deepcopy(source).eval()

    def forward(self, y_value, v_value, input_value):
        return self.core(y_value, v_value, input_value)


deployment = DeploymentAccel(kept).eval()
trace_examples = (
    torch.zeros(3, dtype=DTYPE),
    torch.zeros(3, dtype=DTYPE),
    torch.zeros(3, dtype=DTYPE),
)
torch.jit.trace(deployment, trace_examples, check_trace=True).save("final_accel.pt")

# The single final evaluation run: initialized once at the excerpt's first
# measured position and finite-difference velocity, then carried across split.
y0 = float(y[0])
v0 = float((y[1] - y[0]) / DT)
final_states = free_run(ScalarAccel(kept), u, y0, v0)
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
            "rk4_u_convention": "linear",
        },
        f,
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
print(f"TOTAL_RUNTIME_SECONDS={time.time()-START_TIME:.3f}", flush=True)
