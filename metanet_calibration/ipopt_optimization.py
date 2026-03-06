import numpy as np
from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Param,
    Var,
    Objective,
    SolverFactory,
    minimize,
    value,
    Constraint,
    ConstraintList,
    Expression,
    NonNegativeReals
)
import pyomo.environ as pyo

from .metanet_dynamics import density_dynamics, velocity_dynamics_MN
from matplotlib import pyplot as plt

def smooth_inflow(inflow, window_size=2):
    # Create averaging kernel
    kernel = np.ones(window_size) / window_size

    # Compute asymmetric padding for even window sizes
    pad_left = window_size // 2
    pad_right = window_size - pad_left - 1

    # Pad using boundary values (edge padding)
    if inflow.ndim == 1:
        padded = np.pad(inflow, (pad_left, pad_right), mode="edge")
    else:
        padded = np.pad(inflow, ((pad_left, pad_right), (0, 0)), mode="edge")

    # Convolve along time dimension
    smoothed = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=padded
    )
    return smoothed

def fit_fd1(
    flattened_rho_hat,
    flattened_q_hat,
    C_i=None,
    V_free_init=60,
    a_init=1.0,
    solver_name="ipopt",
    plot=True,
    top_k_for_C=5,
):
    """
    Fit a smooth FD1 fundamental diagram to (rho_hat, q_hat) data using Pyomo.
    Uses an asymmetric (envelope) objective so the curve fits the *upper boundary*
    of the scatter, and constrains the curve peak to exactly equal C_i.

    Parameters
    ----------
    flattened_rho_hat : array-like
        Density measurements (veh/km).
    flattened_q_hat : array-like
        Flow measurements (veh/h).
    C_i : float or None
        Fixed capacity value. If None, capacity is estimated from the top K flows.
    V_free_init : float
        Initial guess for free-flow speed.
    a_init : float
        Initial guess for shape parameter a.
    solver_name : str
        Solver to use (default: 'ipopt').
    plot : bool
        Whether to plot the fitted curve and data.
    top_k_for_C : int
        Number of top flow values to average for capacity estimate if C_i is None.

    Returns
    -------
    dict
        Optimized parameters {'rho_crit', 'V_free', 'a', 'C'}.
    """

    flattened_rho_hat = np.array(flattened_rho_hat)
    flattened_q_hat = np.array(flattened_q_hat)
    K = len(flattened_rho_hat)

    # Estimate capacity if not provided
    if C_i is None:
        C_i = np.mean(sorted(flattened_q_hat)[-top_k_for_C:])

    # Build Pyomo model
    model = ConcreteModel()
    model.k = RangeSet(0, K - 1)

    # Parameters
    model.rho_hat = Param(model.k, initialize=dict(enumerate(flattened_rho_hat)))
    model.q_hat   = Param(model.k, initialize=dict(enumerate(flattened_q_hat)))
    model.C       = Param(initialize=C_i)

    # Decision variables
    model.rho_crit = Var(
        bounds=(1e-2, max(flattened_rho_hat)),
        initialize=np.median(flattened_rho_hat),
    )
    model.V_free = Var(bounds=(10, 150), initialize=V_free_init)
    model.a      = Var(bounds=(0.01, 10), initialize=a_init)

    # ── FD1 expression ────────────────────────────────────────────────────────
    # q(ρ) = ρ · V_free · exp(−(ρ/ρ_crit)^a / a)
    # Peak is at ρ = ρ_crit  →  q_peak = ρ_crit · V_free · exp(−1/a)
    def q_pred_expr(model, k):
        rho = model.rho_hat[k]
        return rho * model.V_free * pyo.exp(-1 / model.a * (rho / model.rho_crit) ** model.a)

    model.q_pred = Expression(model.k, rule=q_pred_expr)

    # ── Peak constraint: force q(ρ_crit) = C_i ────────────────────────────────
    # Substituting ρ = ρ_crit into the formula gives:
    #   q_peak = ρ_crit · V_free · exp(−1/a)  =  C_i
    def peak_at_capacity(model):
        return model.rho_crit * model.V_free * pyo.exp(-1 / model.a) == model.C

    model.peak_con = Constraint(rule=peak_at_capacity)

    # ── Asymmetric (envelope) objective ───────────────────────────────────────
    # Introduce non-negative slack variables s_k ≥ q_hat[k] − q_pred[k].
    # Minimising Σ s_k² pushes the curve *up* to envelope the data from above
    # instead of splitting it down the middle.
    #
    # Points already below the curve contribute 0 (slack stays at 0).
    # Points above the curve are penalised quadratically.
    model.slack = Var(model.k, within=NonNegativeReals, initialize=0.0)

    def slack_upper_bound(model, k):
        # slack[k]  ≥  q_hat[k] − q_pred[k]   (active only when data > curve)
        return model.slack[k] >= model.q_hat[k] - model.q_pred[k]

    model.slack_con = Constraint(model.k, rule=slack_upper_bound)

    def obj_rule(model):
        return sum(model.slack[k] ** 2 for k in model.k)

    model.obj = Objective(rule=obj_rule, sense=minimize)

    # Solve
    solver = SolverFactory(solver_name)
    solver.solve(model, tee=False)

    # Extract results
    rho_crit_opt = value(model.rho_crit)
    V_free_opt   = value(model.V_free)
    a_opt        = value(model.a)
    C_opt        = value(model.C)

    def Q_fd1(rho):
        rho = np.array(rho)
        return V_free_opt * rho * np.exp(-1 / a_opt * (rho / rho_crit_opt) ** a_opt)

    # Plot
    if plot:
        rho_range = np.linspace(0, max(flattened_rho_hat) * 1.1, 500)
        q_fit = Q_fd1(rho_range)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            flattened_rho_hat, flattened_q_hat,
            color="gray", alpha=0.5, label="Data", s=1,
        )
        plt.plot(rho_range, q_fit, linewidth=2.5, label="Fitted FD1 (envelope)", zorder=10)
        plt.axvline(rho_crit_opt, color="red",  linestyle="--", label=f"ρ_crit = {rho_crit_opt:.1f}")
        plt.axhline(C_opt,        color="blue", linestyle=":",  label=f"C = {C_opt:.1f}")
        plt.xlabel("Density ρ (veh/km)")
        plt.ylabel("Flow q (veh/h)")
        plt.title("Fundamental Diagram Fit (FD1) — Envelope")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "rho_crit": rho_crit_opt,
        "V_free":   V_free_opt,
        "a":        a_opt,
        "C":        C_opt,
        "Q_fd1":    Q_fd1,
    }
def metanet_param_fit(
    v_hat,
    rho_hat,
    q_hat,
    T,
    l,
    initial_traffic_state,
    downstream_density,
    num_calibrated_segments,
    include_ramping=True,
    varylanes=False,
    lane_mapping=None,
    on_ramp_mapping=None,
    off_ramp_mapping=None,
    on_ramp_flows_gt=None,
    off_ramp_flows_gt=None,
    time_varying_ramping=False,
    bounds=None,
    initialization=None,
):
    initial_flow_or = initial_traffic_state

    if initial_flow_or.ndim == 1:
        initial_flow_or = initial_flow_or.reshape(-1, 1)

    if downstream_density.ndim == 1:
        downstream_density = downstream_density.reshape(-1, 1)

    # Ensure no divide-by-zero

    num_timesteps, num_segments = v_hat.shape

    model = ConcreteModel()
    model.t = RangeSet(0, num_timesteps - 1)
    model.i = RangeSet(0, num_segments - 1)
    # model.i = RangeSet(0, num_calibrated_segments - 1)
    model.num_segments = num_segments
    model.num_calibrated_segments = num_calibrated_segments

    # Fixed params
    model.T = Param(initialize=T)
    model.l = Param(initialize=l)

    # Number of lanes (per calibrated segment)
    if lane_mapping is not None:
        model.n_lanes = Param(model.i, initialize={i: lane_mapping[i] for i in model.i})
    elif varylanes:
        model.n_lanes = Var(model.i, bounds=(3, 5), initialize=3)
    else:
        model.n_lanes = Param(model.i, initialize=4)

    combined_bounds = {
        "eta_high": (15.0, 60.0),
        "tau": (15.0 / 3600, 60.0 / 3600),
        "K": (5.0, 60.0),
        "rho_crit": (15, 100),
        "v_free": (110, 150),
        "a": (0.5, 5),
        "beta": (1e-3, 0.9),
        "r_inflow": (1e-3, 2000),
    }

    if bounds is not None:
        print("Using custom bounds from input")
        combined_bounds.update(bounds)

    # Default initialization will be the midpoint of the bounds except for beta and r
    combined_initialization = {
        "eta_high": np.mean(combined_bounds["eta_high"]),
        "tau": np.mean(combined_bounds["tau"]),
        "K": np.mean(combined_bounds["K"]),
        "rho_crit": np.mean(combined_bounds["rho_crit"]),
        "v_free": np.mean(combined_bounds["v_free"]),
        "a": np.mean(combined_bounds["a"]),
        "beta": 1e-3,
        "r_inflow": 1e-3,
    }
    if time_varying_ramping:
        combined_initialization["beta"] = np.full((num_timesteps, num_segments), combined_initialization["beta"])
        combined_initialization["r_inflow"] = np.full(
            (num_timesteps, num_segments), combined_initialization["r_inflow"]
        )
        if on_ramp_flows_gt is not None:
            print("Using on-ramp flow ground truth for r_inflow initialization")
            combined_initialization["r_inflow"] = on_ramp_flows_gt
        if off_ramp_flows_gt is not None:
            print("Using off-ramp flow ground truth for beta initialization")
            combined_initialization["beta"] = off_ramp_flows_gt
    if initialization is not None:
        print("Using custom initialization from input")
        combined_initialization.update(initialization)
    if not time_varying_ramping:
        for key in ("beta", "r_inflow"):
            val = combined_initialization[key]
            if hasattr(val, "ndim") and val.ndim > 0:
                # collapse to a scalar (take the mean, or first element)
                combined_initialization[key] = float(np.mean(val))
    model.eta_high = Var(
        model.i,
        bounds=combined_bounds["eta_high"],
        initialize=combined_initialization["eta_high"],
    )
    model.tau = Var(
        model.i,
        bounds=combined_bounds["tau"],
        initialize=combined_initialization["tau"],
    )
    model.K = Var(
        model.i, bounds=combined_bounds["K"], initialize=combined_initialization["K"]
    )
    model.rho_crit = Var(
        model.i,
        bounds=combined_bounds["rho_crit"],
        initialize=combined_initialization["rho_crit"],
    )
    model.v_free = Var(
        model.i,
        bounds=combined_bounds["v_free"],
        initialize=combined_initialization["v_free"],
    )
    model.a = Var(
        model.i, bounds=combined_bounds["a"], initialize=combined_initialization["a"]
    )

    if include_ramping:
        # model.gamma = Var(model.i, bounds=(0.5, 1.5), initialize=1)
        if not time_varying_ramping:
            model.beta = Var(
                model.i,
                bounds=combined_bounds["beta"],
                initialize=combined_initialization["beta"],
            )
            model.r_inflow = Var(
                model.i,
                bounds=combined_bounds["r_inflow"],
                initialize=combined_initialization["r_inflow"],
            )
        if on_ramp_mapping is not None and off_ramp_mapping is not None:

            if time_varying_ramping:
                if on_ramp_flows_gt is not None:
                    model.r_inflow = Var(
                        model.t,
                        model.i,
                        bounds={
                            (t, i): (
                                combined_initialization["r_inflow"][t, i],
                                combined_initialization["r_inflow"][t, i],
                            )
                            for t in model.t
                            for i in model.i
                        },
                        initialize={
                            (t, i): combined_initialization["r_inflow"][t, i]
                            for t in model.t
                            for i in model.i
                        },
                    )
                else:
                    model.r_inflow = Var(
                        model.t,
                        model.i,
                        bounds=combined_bounds["r_inflow"],
                        initialize={
                            (t, i): combined_initialization["r_inflow"][t, i]
                            for t in model.t
                            for i in model.i
                        },
                    )
                if off_ramp_flows_gt is not None:
                    model.beta = Var(
                        model.t,
                        model.i,
                        bounds={
                            (t, i): (
                                combined_initialization["beta"][t, i],
                                combined_initialization["beta"][t, i],
                            )
                            for t in model.t
                            for i in model.i
                        },
                        initialize={
                            (t, i): combined_initialization["beta"][t, i]
                            for t in model.t
                            for i in model.i
                        },
                    )
                else:
                    model.beta = Var(
                        model.t,
                        model.i,
                        bounds=combined_bounds["beta"],
                        initialize={
                            (t, i): combined_initialization["beta"][t, i]
                            for t in model.t
                            for i in model.i
                        },
                    )
                
                for t in model.t:
                    for i in model.i:
                        if on_ramp_mapping[i] == 0:
                            model.r_inflow[t, i].setlb(0.0)
                            model.r_inflow[t, i].setub(0.0)
                        if off_ramp_mapping[i] == 0:
                            model.beta[t, i].setlb(0.0)
                            model.beta[t, i].setub(0.0)

            else:  # set bounds based on mappings
                for i in model.i:
                    if on_ramp_mapping[i] == 0:
                        model.r_inflow[i].setlb(0.0)
                        model.r_inflow[i].setub(0.0)
                    if off_ramp_mapping[i] == 0:
                        model.beta[i].setlb(0.0)
                        model.beta[i].setub(0.0)

    else:
        model.beta = Var(model.i, bounds=(0.0, 0.0), initialize=0.0)
        model.r_inflow = Var(model.i, bounds=(0.0, 0.0), initialize=0.0)

    # Variables to predict (per-lane values)
    model.v_pred = Var(
        model.t,
        model.i,
        bounds=(1e-4, 150),
        initialize={(t, i): float(v_hat[t, i]) for t in model.t for i in model.i},
    )
    model.rho_pred = Var(
        model.t,
        model.i,
        bounds=(1e-4, 400),
        initialize={(t, i): float(rho_hat[t, i]) for t in model.t for i in model.i},
    )
    model.q_pred = Var(
        model.t,
        model.i,
        bounds=(1e-4, 10000),
        initialize={(t, i): float(q_hat[t, i]) for t in model.t for i in model.i},
    )

    # Initial conditions
    model.constraints = ConstraintList()
    for i in range(num_segments):
        model.constraints.add(model.v_pred[0, i] == v_hat[0, i].item())
        model.constraints.add(model.rho_pred[0, i] == rho_hat[0, i].item())

    # Observed data
    model.v_hat = Param(
        model.t,
        model.i,
        initialize={(t, i): float(v_hat[t, i]) for t in model.t for i in model.i},
    )
    model.rho_hat = Param(
        model.t,
        model.i,
        initialize={(t, i): float(rho_hat[t, i]) for t in model.t for i in model.i},
    )
    model.q_hat = Param(
        model.t,
        model.i,
        initialize={(t, i): float(q_hat[t, i]) for t in model.t for i in model.i},
    )

    def rho_update(m, t, i):
        if t == 0:
            return Constraint.Skip
        seg = i
        if i == 0:
            current = m.rho_pred[t - 1, 0]
            inflow = initial_flow_or[t - 1, 0]
            outflow = m.rho_pred[t - 1, i] * m.v_pred[t - 1, i]
        else:
            current = m.rho_pred[t - 1, i]
            inflow = m.rho_pred[t - 1, i - 1] * m.v_pred[t - 1, i - 1]
            outflow = m.rho_pred[t - 1, i] * m.v_pred[t - 1, i]
        if include_ramping:
            if time_varying_ramping:
                return m.rho_pred[t, i] == density_dynamics(
                    current,
                    inflow,
                    outflow,
                    model.T,
                    model.l,
                    model.beta[t - 1, i],
                    model.r_inflow[t - 1, i],
                )
            else:
                return m.rho_pred[t, i] == density_dynamics(
                    current,
                    inflow,
                    outflow,
                    model.T,
                    model.l,
                    model.beta[i],
                    model.r_inflow[i],
                )
        else:
            return m.rho_pred[t, i] == density_dynamics(
                current, inflow, outflow, model.T, model.l, 0.0, 0.0
            )

    model.rho_dyn = Constraint(model.t, model.i, rule=rho_update)
    # Velocity dynamics
    VSL = 150

    def v_update(m, t, i):
        seg = i
        if t == 0:
            return Constraint.Skip

        current = m.v_pred[t - 1, i]
        prev_state = m.v_pred[t - 1, i]
        density = m.rho_pred[t - 1, i]
        current_lanes = m.n_lanes[i]

        if num_segments == 1:
            # single-segment case
            next_density = downstream_density[t - 1]
            next_lanes = current_lanes
        elif i == 0:
            # first segment in a multi-segment block
            next_density = m.rho_pred[t - 1, i + 1]
            next_lanes = m.n_lanes[i + 1]
        elif i == num_segments - 1:
            # last segment in block
            prev_state = m.v_pred[t - 1, i - 1]
            next_density = downstream_density[t - 1]
            next_lanes = current_lanes
        else:
            # interior segment
            prev_state = m.v_pred[t - 1, i - 1]
            next_density = m.rho_pred[t - 1, i + 1]
            next_lanes = m.n_lanes[i + 1]

        args = {
            "current": current,
            "prev_state": prev_state,
            "total_density": density,
            "next_total_density": next_density,
            "lanes_current": current_lanes,
            "lanes_next": next_lanes,
            "v_ctrl": VSL,
            "T": m.T,
            "l": m.l,
            "eta_high": m.eta_high[i],
            "K": m.K[i],
            "tau": m.tau[i],
            "a": m.a[i],
            "rho_crit": m.rho_crit[i],
            "v_free": m.v_free[i],
        }
        return m.v_pred[t, i] == velocity_dynamics_MN(**args)

    model.v_dyn = Constraint(model.t, model.i, rule=v_update)

    # Objective: per-lane error
    def loss_fn(m):
        v_max = max(m.v_hat[t, i] for t in m.t for i in m.i)
        rho_max = max(m.rho_hat[t, i] for t in m.t for i in m.i)
        q_max = max(m.q_hat[t, i] for t in m.t for i in m.i)

        return sum(
            (20 * ((m.v_pred[t, i] - m.v_hat[t, i]) / v_max) ** 2)
            + ((m.rho_pred[t, i] - m.rho_hat[t, i]) / rho_max) ** 2
            + ((m.q_pred[t, i] - m.q_hat[t, i]) / q_max) ** 2
            for t in m.t
            for i in m.i
        )

    model.loss = Objective(rule=loss_fn, sense=minimize)

    # Solve
    solver = SolverFactory("ipopt")
    # solver.options["tol"] = 1e-15
    # solver.options["constr_viol_tol"] = 1e-10    # constraint violation tolerance
    # solver.options["acceptable_tol"] = 1e-9      # early stopping criterion
    # solver.options["acceptable_constr_viol_tol"] = 1e-9
    # solver.options["dual_inf_tol"] = 1e-10       # dual infeasibility tolerance
    # solver.options["compl_inf_tol"] = 1e-10
    solver.options["max_iter"] = 20000
    solver.options["acceptable_constr_viol_tol"] = 1e-30
    solver.options["constr_viol_tol"] = 1e-13
    # solver.options["dual_inf_tol"] = 1e-11
    # solver.options["acceptable_dual_inf_tol"] = 1e-11
    solver.solve(model, tee=True)

    return model


def run_calibration(
    rho_hat,
    q_hat,
    T,
    l,
    num_calibrated_segments=1,
    sep_boundary_conditions=None,
    include_ramping=True,
    varylanes=True,
    lane_mapping=None,
    on_ramp_mapping=None,
    off_ramp_mapping=None,
    on_ramp_flows_gt=None,
    off_ramp_flows_gt=None,
    smoothing=True,
    time_varying_ramping=False,
    bounds=None,
    initialization=None,
):
    """
    Run METANET parameter calibration with configurable segment grouping.

    Parameters
    ----------
    rho_hat : np.ndarray
        Density measurements (time, segment).
    q_hat : np.ndarray
        Flow measurements (time, segment).
    T : float
        Time step (hours).
    l : float
        Segment length (km).
    num_calibrated_segments : int
        Number of consecutive segments to calibrate at a time.

    Returns
    -------
    results : dict
        Dictionary with concatenated predictions and parameter arrays.
    """

    v_hat = q_hat / rho_hat

    results = {
        "v_pred": [],
        "rho_pred": [],
        "tau": [],
        "K": [],
        "eta_high": [],
        "rho_crit": [],
        "v_free": [],
        "a": [],
        "num_lanes": [],
    }
    # results["gamma"] = []
    results["beta"] = []
    results["r_inflow"] = []

    n_segments = rho_hat.shape[1]

    # If sep_boundary_conditions is None we exclude the first (upstream) and last (downstream) columns
    if sep_boundary_conditions is None:
        # calibrate only "interior" segments, keep upstream inflow and downstream density as boundaries
        start_segment = 1
        end_segment_exclusive = n_segments - 1
    else:
        # sep boundary conds already supply boundaries; calibrate all segments
        start_segment = 0
        end_segment_exclusive = n_segments

    if start_segment >= end_segment_exclusive:
        raise ValueError(
            f"No segments to calibrate: start_segment={start_segment}, "
            f"end_segment_exclusive={end_segment_exclusive}, n_segments={n_segments}"
        )

    for start_idx in range(
        start_segment, end_segment_exclusive, num_calibrated_segments
    ):
        end_idx = min(start_idx + num_calibrated_segments, end_segment_exclusive)
        # defensive check: ensure the slice is non-empty
        if end_idx <= start_idx:
            # shouldn't happen due to the loop setup, but guard anyway
            continue

        # logging for debugging
        print(f"Calibrating segments [{start_idx}:{end_idx}) out of {n_segments}")

        # Slice for this group
        segment_rho_hat = rho_hat[:, start_idx:end_idx]
        segment_v_hat = v_hat[:, start_idx:end_idx]
        segment_q_hat = q_hat[:, start_idx:end_idx]

        # safety: if any slice is empty, raise a helpful error
        if (
            segment_rho_hat.shape[1] == 0
            or segment_v_hat.shape[1] == 0
            or segment_q_hat.shape[1] == 0
        ):
            raise ValueError(
                f"Empty slice for calibration block: start_idx={start_idx}, end_idx={end_idx}, "
                f"shapes rho_hat={segment_rho_hat.shape}, v_hat={segment_v_hat.shape}, q_hat={segment_q_hat.shape}. "
                "Check num_calibrated_segments and boundary-condition indexing."
            )

        # Boundary conditions depend on group position
        if sep_boundary_conditions is not None:
            initial_flow = sep_boundary_conditions["initial_flow"]
            downstream_density = sep_boundary_conditions["downstream_density"]
        else:
            initial_flow = q_hat[
                :, start_idx - 1 : start_idx
            ]  # upstream inflow (one column)
            downstream_density = rho_hat[
                :, end_idx : end_idx + 1
            ]  # downstream density (one column)

        if smoothing:
            initial_flow = smooth_inflow(initial_flow)  # upstream inflow
            downstream_density = smooth_inflow(downstream_density)  # downstream density
        # Run calibration for this block
        res_model = metanet_param_fit(
            segment_v_hat,
            segment_rho_hat,
            segment_q_hat,
            T,
            l,
            initial_flow,
            downstream_density,
            num_calibrated_segments,
            include_ramping=include_ramping,
            varylanes=varylanes,
            lane_mapping=(
                lane_mapping[start_idx:end_idx] if lane_mapping is not None else None
            ),
            on_ramp_mapping=(
                on_ramp_mapping[start_idx:end_idx]
                if on_ramp_mapping is not None
                else None
            ),
            off_ramp_mapping=(
                off_ramp_mapping[start_idx:end_idx]
                if off_ramp_mapping is not None
                else None
            ),
            on_ramp_flows_gt=(
                on_ramp_flows_gt[:, start_idx:end_idx]
                if on_ramp_flows_gt is not None
                else None
            ),
            off_ramp_flows_gt=(
                off_ramp_flows_gt[:, start_idx:end_idx]
                if off_ramp_flows_gt is not None
                else None
            ),
            time_varying_ramping=time_varying_ramping,
            bounds=bounds,
            initialization=initialization,
        )

        num_timesteps, num_segments = segment_v_hat.shape

        v_pred_array = np.zeros((num_timesteps, num_segments))
        rho_pred_array = np.zeros((num_timesteps, num_segments))

        for t in range(num_timesteps):
            for i in range(num_segments):
                v_pred_array[t, i] = value(res_model.v_pred[t, i])
                rho_pred_array[t, i] = value(res_model.rho_pred[t, i])

        # Append predictions
        if len(results["v_pred"]) == 0:
            results["v_pred"] = v_pred_array
            results["rho_pred"] = rho_pred_array
        else:
            results["v_pred"] = np.concatenate(
                [results["v_pred"], v_pred_array], axis=1
            )
            results["rho_pred"] = np.concatenate(
                [results["rho_pred"], rho_pred_array], axis=1
            )

        # Append parameter arrays
        results["tau"].extend([value(res_model.tau[i]) for i in range(num_segments)])
        results["K"].extend([value(res_model.K[i]) for i in range(num_segments)])
        results["eta_high"].extend(
            [value(res_model.eta_high[i]) for i in range(num_segments)]
        )
        results["rho_crit"].extend(
            [value(res_model.rho_crit[i]) for i in range(num_segments)]
        )
        results["v_free"].extend(
            [value(res_model.v_free[i]) for i in range(num_segments)]
        )
        results["a"].extend([value(res_model.a[i]) for i in range(num_segments)])
        results["num_lanes"].extend(
            [value(res_model.n_lanes[i]) for i in range(num_segments)]
        )

        if time_varying_ramping:
            results["beta"] = np.ndarray((num_timesteps, num_segments))
            results["r_inflow"] = np.ndarray((num_timesteps, num_segments))
            for t in range(num_timesteps):
                for i in range(num_segments):
                    results["beta"][t, i] = value(res_model.beta[t, i])
                    results["r_inflow"][t, i] = value(res_model.r_inflow[t, i])
        else:
            results["beta"].extend(
                [value(res_model.beta[i]) for i in range(num_segments)]
            )
            results["r_inflow"].extend(
                [value(res_model.r_inflow[i]) for i in range(num_segments)]
            )

    # Convert parameter lists to numpy arrays
    for key in ["tau", "K", "eta_high", "rho_crit", "v_free", "a", "num_lanes"]:
        results[key] = np.array(results[key])
    results["beta"] = np.array(results["beta"])
    results["r_inflow"] = np.array(results["r_inflow"])
    return results


def mape(flow_hat, flow_pred):
    """
    Compute the Mean Absolute Percentage Error (MAPE) between ground truth and prediction.

    Parameters:
        flow_hat (np.ndarray): Ground truth array of shape [t, i]
        flow_pred (np.ndarray): Predicted array of shape [t, i]

    Returns:
        float: The mean absolute percentage error (in percent)
    """
    # Avoid division by zero by masking out zero ground truth values
    mask = flow_hat != 0

    error = np.abs((flow_pred[mask] - flow_hat[mask]) / flow_hat[mask])
    return np.mean(error) * 100


def rmse(flow_hat, flow_pred):
    """
    Compute the Root Mean Squared Error (RMSE) between ground truth and prediction.

    Parameters:
        flow_hat (np.ndarray): Ground truth array of shape [t, i]
        flow_pred (np.ndarray): Predicted array of shape [t, i]

    Returns:
        float: The root mean squared error
    """
    error = flow_pred - flow_hat
    mse = np.mean(np.square(error))
    return np.sqrt(mse)
