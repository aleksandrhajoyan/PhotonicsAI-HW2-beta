import os
import random
import time
import json
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Reproducibility and CPU-only execution.
SEED = 20250308
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
DEVICE = torch.device("cpu")

FS = 610.35
DT = 1.0 / FS
TAU = 0.05
N_TRAIN = 3072
N_TOTAL = 4096
EXCERPT_START = 42650


def _numeric_vector(value):
    """Convert a MATLAB channel value to a flat numeric vector when possible."""
    try:
        arr = np.asarray(value)
        if arr.dtype == object and arr.size == 1:
            return _numeric_vector(arr.item())
        if np.issubdtype(arr.dtype, np.number):
            arr = np.squeeze(arr)
            if arr.ndim == 1:
                return np.asarray(arr, dtype=np.float64)
            if arr.ndim == 2 and 1 in arr.shape:
                return np.asarray(arr.reshape(-1), dtype=np.float64)
    except Exception:
        pass
    return None


def _find_named_channels(obj, found):
    """Recursively search MATLAB structs/containers for V1 and V2 fields."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            low = str(key).lower()
            if low in ("v1", "v2"):
                vec = _numeric_vector(value)
                if vec is not None:
                    found[low] = vec
            _find_named_channels(value, found)
        return

    if hasattr(obj, "_fieldnames"):
        for field in obj._fieldnames:
            value = getattr(obj, field)
            low = str(field).lower()
            if low in ("v1", "v2"):
                vec = _numeric_vector(value)
                if vec is not None:
                    found[low] = vec
            _find_named_channels(value, found)
        return

    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            for field in obj.dtype.names:
                value = obj[field]
                low = str(field).lower()
                if low in ("v1", "v2"):
                    vec = _numeric_vector(value)
                    if vec is not None:
                        found[low] = vec
                _find_named_channels(value, found)
        elif obj.dtype == object:
            for value in obj.flat:
                _find_named_channels(value, found)


def _load_channels(path):
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    found = {}
    for key, value in mat.items():
        if not key.startswith("__"):
            low = key.lower()
            if low in ("v1", "v2"):
                vec = _numeric_vector(value)
                if vec is not None:
                    found[low] = vec
            _find_named_channels(value, found)

    if "v1" in found and "v2" in found:
        return found["v1"], found["v2"]

    # Some releases store the two channels together in a numeric data matrix.
    candidates = []
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        try:
            arr = np.asarray(value)
            if np.issubdtype(arr.dtype, np.number) and arr.ndim == 2:
                if arr.shape[0] >= 46746 and arr.shape[1] >= 2:
                    candidates.append((arr[:, 0], arr[:, 1]))
                elif arr.shape[1] >= 46746 and arr.shape[0] >= 2:
                    candidates.append((arr[0, :], arr[1, :]))
        except Exception:
            pass

    if candidates:
        return (np.asarray(candidates[0][0], dtype=np.float64).reshape(-1),
                np.asarray(candidates[0][1], dtype=np.float64).reshape(-1))
    raise ValueError("Could not find Silverbox channels V1 and V2 in the MAT-file.")


class AccelerationMLP(nn.Module):
    """Dimensionless acceleration q(Y,W,U)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


class PhysicalAcceleration(nn.Module):
    """Physical-unit acceleration interface, with all scaling inside."""

    def __init__(self, network, sy, su, sv, sa):
        super().__init__()
        self.network = network
        self.register_buffer("sy", torch.tensor(float(sy), dtype=torch.float32))
        self.register_buffer("su", torch.tensor(float(su), dtype=torch.float32))
        self.register_buffer("sv", torch.tensor(float(sv), dtype=torch.float32))
        self.register_buffer("sa", torch.tensor(float(sa), dtype=torch.float32))

    def forward(self, y, v, u):
        normalized = torch.cat(
            (y.reshape(-1, 1) / self.sy,
             v.reshape(-1, 1) / self.sv,
             u.reshape(-1, 1) / self.su),
            dim=1,
        )
        return (self.network(normalized).reshape(-1) * self.sa)


def free_run(accel, u, y0, v0):
    """One full free run; input is zero-order held over each RK4 interval."""
    y_state, v_state = float(y0), float(v0)
    states = np.empty((len(u), 2), dtype=np.float64)
    states[0, 0] = y_state
    states[0, 1] = v_state

    def acceleration(y_value, v_value, u_value):
        yy = torch.tensor([y_value], dtype=torch.float32, device=DEVICE)
        vv = torch.tensor([v_value], dtype=torch.float32, device=DEVICE)
        uu = torch.tensor([u_value], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return float(accel(yy, vv, uu).reshape(-1)[0].item())

    for k in range(len(u) - 1):
        uk = float(u[k])

        k1_y = v_state
        k1_v = acceleration(y_state, v_state, uk)

        k2_y = v_state + 0.5 * DT * k1_v
        k2_v = acceleration(
            y_state + 0.5 * DT * k1_y,
            v_state + 0.5 * DT * k1_v,
            uk,
        )

        k3_y = v_state + 0.5 * DT * k2_v
        k3_v = acceleration(
            y_state + 0.5 * DT * k2_y,
            v_state + 0.5 * DT * k2_v,
            uk,
        )

        k4_y = v_state + DT * k3_v
        k4_v = acceleration(
            y_state + DT * k3_y,
            v_state + DT * k3_v,
            uk,
        )

        y_state, v_state = (
            y_state + (DT / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y),
            v_state + (DT / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v),
        )
        states[k + 1, 0] = y_state
        states[k + 1, 1] = v_state

    return states


def main():
    start_time = time.time()
    path = os.environ.get("SILVERBOX_MAT_PATH")
    if not path:
        raise RuntimeError("SILVERBOX_MAT_PATH is not set.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Silverbox MAT-file does not exist: {path}")

    v1, v2 = _load_channels(path)
    stop = EXCERPT_START + N_TOTAL
    if len(v1) < stop or len(v2) < stop:
        raise ValueError(
            f"Channels are too short for [{EXCERPT_START}:{stop}]: "
            f"V1={len(v1)}, V2={len(v2)}"
        )

    u = np.asarray(v1[EXCERPT_START:stop], dtype=np.float64).copy()
    y = np.asarray(v2[EXCERPT_START:stop], dtype=np.float64).copy()
    if u.shape != (N_TOTAL,) or y.shape != (N_TOTAL,):
        raise ValueError(f"Unexpected excerpt shapes: u={u.shape}, y={y.shape}")

    u -= np.mean(u)
    y -= np.mean(y)
    u_train = u[:N_TRAIN]
    y_train = y[:N_TRAIN]

    # All scales are determined from training samples only.
    sy = float(np.sqrt(np.mean(y_train ** 2)))
    su = float(np.sqrt(np.mean(u_train ** 2)))
    if sy <= 0.0 or su <= 0.0:
        raise ValueError(f"Invalid training scales: sy={sy}, su={su}")
    sv = sy / TAU
    sa = sy / (TAU * TAU)

    model = AccelerationMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-3, weight_decay=1.0e-5
    )

    # Finite-difference training initial velocities, computed exclusively
    # from the training record.
    v_train = np.empty(N_TRAIN, dtype=np.float64)
    v_train[0] = (y_train[1] - y_train[0]) / DT
    v_train[-1] = (y_train[-1] - y_train[-2]) / DT
    v_train[1:-1] = (y_train[2:] - y_train[:-2]) / (2.0 * DT)

    y_train_n = torch.tensor(y_train / sy, dtype=torch.float32, device=DEVICE)
    u_train_n = torch.tensor(u_train / su, dtype=torch.float32, device=DEVICE)
    v_train_n = torch.tensor(v_train / sv, dtype=torch.float32, device=DEVICE)

    def derivative(y_norm, w_norm, u_norm):
        x = torch.stack((y_norm, w_norm, u_norm), dim=-1)
        q = model(x).squeeze(-1)
        return w_norm / TAU, q / TAU

    def rk4_batch(y_state, w_state, input_value):
        # Zero-order-hold convention: all stages use the same input sample.
        k1_y, k1_w = derivative(y_state, w_state, input_value)
        k2_y, k2_w = derivative(
            y_state + 0.5 * DT * k1_y,
            w_state + 0.5 * DT * k1_w,
            input_value,
        )
        k3_y, k3_w = derivative(
            y_state + 0.5 * DT * k2_y,
            w_state + 0.5 * DT * k2_w,
            input_value,
        )
        k4_y, k4_w = derivative(
            y_state + DT * k3_y,
            w_state + DT * k3_w,
            input_value,
        )
        y_next = y_state + (DT / 6.0) * (
            k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y
        )
        w_next = w_state + (DT / 6.0) * (
            k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w
        )
        return y_next, w_next

    batch_size = 12
    training_updates = 720
    progress_every = 120
    print(
        "Training 3-64-64-64-1 tanh acceleration MLP; "
        "AdamW lr=1e-3, batch=12, rollout curriculum=32/64/96.",
        flush=True,
    )

    last_loss = float("nan")
    for update in range(training_updates):
        if update < 180:
            rollout_length = 32
        elif update < 420:
            rollout_length = 64
        else:
            rollout_length = 96

        max_start = N_TRAIN - rollout_length
        starts_np = np.random.randint(0, max_start + 1, size=batch_size)
        starts = torch.tensor(starts_np, dtype=torch.long, device=DEVICE)

        y_state = y_train_n[starts]
        w_state = v_train_n[starts]
        losses = []

        for offset in range(rollout_length - 1):
            indices = starts + offset
            input_value = u_train_n[indices]
            y_state, w_state = rk4_batch(y_state, w_state, input_value)
            target = y_train_n[indices + 1]
            losses.append(torch.mean((y_state - target) ** 2))

        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = float(loss.detach().item())

        if (update + 1) % progress_every == 0 or update == 0:
            print(
                f"update={update + 1}/{training_updates} "
                f"rollout={rollout_length} normalized_mse={last_loss:.6g} "
                f"elapsed_s={time.time() - start_time:.1f}",
                flush=True,
            )

    model.eval()
    accel_module = PhysicalAcceleration(model, sy, su, sv, sa).to(DEVICE).eval()

    # Trace the required physical-unit interface in the rollout dtype.
    trace_inputs = (
        torch.zeros(1, dtype=torch.float32, device=DEVICE),
        torch.zeros(1, dtype=torch.float32, device=DEVICE),
        torch.zeros(1, dtype=torch.float32, device=DEVICE),
    )
    traced_accel = torch.jit.trace(accel_module, trace_inputs)
    traced_accel.save("final_accel.pt")

    # The sole final full-record free run. Its only output initialization is
    # at sample zero; measured y is not passed into or read by free_run.
    y0 = float(y[0])
    v0 = float((y[1] - y[0]) / DT)
    states = free_run(traced_accel, u, y0, v0)
    prediction = states[:, 0].copy()

    train_rmse = float(np.sqrt(np.mean((prediction[:N_TRAIN] - y[:N_TRAIN]) ** 2)))
    test_rmse = float(
        np.sqrt(np.mean((prediction[N_TRAIN:] - y[N_TRAIN:]) ** 2))
    )

    np.save("freerun_states.npy", states.astype(np.float64, copy=False))
    np.save("freerun_prediction.npy", prediction.astype(np.float64, copy=False))
    with open("boundary_state.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "k_3071": [float(states[3071, 0]), float(states[3071, 1])],
                "k_3072": [float(states[3072, 0]), float(states[3072, 1])],
                "state_initialisations": 1,
                "rk4_steps": 4095,
                "rk4_u_convention": "zoh",
            },
            f,
            indent=2,
        )

    test_indices = np.arange(N_TRAIN, N_TOTAL)
    relative_time = (test_indices - N_TRAIN) * DT
    plt.figure(figsize=(10, 4.5))
    plt.plot(relative_time, y[N_TRAIN:], label="Measured output", linewidth=1.2)
    plt.plot(relative_time, prediction[N_TRAIN:], label="Free-run prediction", linewidth=1.2)
    plt.xlabel("Time from test-segment start (s)")
    plt.ylabel("Centered output voltage (V)")
    plt.title("Silverbox test segment: continuous free run")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("silverbox_test_free_run.png", dpi=160)
    plt.close()

    print(
        "State equation: y_dot=v, v_dot=a_theta(y,v,u); "
        "RK4 with zero-order-held u.",
        flush=True,
    )
    print(
        "Architecture: fully connected 3-64-64-64-1 MLP, tanh hidden "
        "activations; training-only RMS scaling; AdamW lr=1e-3, "
        f"weight_decay=1e-5, {training_updates} updates, rollout curriculum "
        "32/64/96, batch=12.",
        flush=True,
    )
    print(f"TRAIN_RMSE={train_rmse:.10g}", flush=True)
    print(f"TEST_RMSE={test_rmse:.10g}", flush=True)
    print("Saved plot: silverbox_test_free_run.png", flush=True)
    print(f"Total elapsed seconds: {time.time() - start_time:.1f}", flush=True)


if __name__ == "__main__":
    main()
