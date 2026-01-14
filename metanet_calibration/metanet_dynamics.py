import numpy as np
from typing import Dict, Tuple, Optional

# ----------------------------
# Core METANET helper functions (Total Density Version)
# ----------------------------

def density_dynamics(current_total: float, inflow: float, outflow: float, T: float, l: float,
                     beta: float = 0.0, r: float = 0.0) -> float:
    """Update total density with conservation equation (per segment)."""
    return current_total + T / l * (inflow - outflow / (1 - beta) + r)


def flow_dynamics(total_density: float, velocity: float) -> float:
    """Fundamental relation: q = rho_total * v."""
    return total_density * velocity


def queue_dynamics(current: float, demand: float, flow_origin: float, T: float) -> float:
    """Update queue length."""
    return current + T * (demand - flow_origin)


def calculate_V(rho_per_lane, v_ctrl: float, a: float, rho_crit: float, v_free: float = 150.0):
    """Desired speed function V(rho), capped by control speed v_ctrl.
    
    Handles both Pyomo expressions and numpy/scalar types.
    """
    try:
        from pyomo.environ import exp as pyo_exp
        from pyomo.core.expr.numeric_expr import NumericExpression
        
        if isinstance(rho_per_lane, NumericExpression):
            # For Pyomo expressions - no capping, return uncapped desired speed
            return v_free * pyo_exp(- (rho_per_lane / rho_crit) ** a / a)
    except ImportError:
        pass
    
    # For numpy/scalar
    if isinstance(rho_per_lane, np.ndarray):
        return np.minimum(v_free * np.exp(- (rho_per_lane / rho_crit) ** a / a), v_ctrl)
    
    return min(v_free * np.exp(- (rho_per_lane / rho_crit) ** a / a), v_ctrl)


def calculate_V_arr(rho_arr: np.ndarray, v_ctrl_arr: np.ndarray, a: float, rho_crit: float, v_free: float) -> np.ndarray:
    """Vectorized desired speed function. Not used in sim, but useful for plotting."""
    return np.minimum(v_free * np.exp(- (rho_arr / rho_crit) ** a / a), v_ctrl_arr)


def velocity_dynamics_MN(current,
                         prev_state,
                         total_density,
                         next_total_density,
                         lanes_current: int,
                         lanes_next: int,
                         v_ctrl: float,
                         T: float,
                         l: float,
                         eta_high: float = 30.0,
                         K: float = 40.0,
                         tau: float = 18 / 3600,
                         a: float = 1.4,
                         rho_crit: float = 37.45,
                         v_free: float = 120.0):
    """One-step METANET velocity update with standard terms.
    
    Returns:
        Updated velocity (for Pyomo: symbolic expression; for numpy/float: with positive floor)
    """
    # Convert to per-lane density for calculations
    rho_per_lane = total_density / lanes_current
    next_rho_per_lane = next_total_density / lanes_next
    
    nxt = (
        current
        + T / tau * (calculate_V(rho_per_lane, v_ctrl, a, rho_crit, v_free) - current)
        + T / l * current * (prev_state - current)
        - (eta_high * T) / (tau * l) * (next_rho_per_lane - rho_per_lane) / (rho_per_lane + K)
    )
    
    # Apply positive floor only for non-Pyomo expressions
    try:
        from pyomo.core.expr.numeric_expr import NumericExpression
        if isinstance(nxt, NumericExpression):
            return nxt  # Return symbolic expression as-is
    except ImportError:
        pass
    
    # For numpy/scalar, apply floor
    if isinstance(nxt, np.ndarray):
        return np.maximum(1e-4, nxt)
    return max(1e-4, nxt)


def origin_flow_dynamics_MN(demand: float,
                            total_density_first: float,
                            lanes_first: int,
                            queue: float,
                            T: float,
                            p_max: float = 180.0,
                            rho_crit: float = 37.45,
                            q_capacity: float = 2200.0) -> float:
    """Origin (on-ramp) sending/merging flow constraint."""
    rho_per_lane_first = total_density_first / lanes_first
    
    return min(
        demand + queue / T,
        lanes_first * q_capacity * (p_max - rho_per_lane_first) / (p_max - rho_crit),
        lanes_first * q_capacity,
    )

def _get_time_space_param(param, t: int, i: int):
    """Helper to index params that may be scalar, 1D (over i), or 2D (over t,i)."""
    if np.ndim(param) == 0:
        return float(param)
    if np.ndim(param) == 1:
        return float(param[i])
    # assume 2D
    return float(param[t, i])


def metanet_step(t: int,
                 density_t: np.ndarray,
                 velocity_t: np.ndarray,
                 queue_t: float,
                 flow_origin_t: float,
                 *,
                 T: float,
                 l: float,
                 vsl_speeds: np.ndarray,
                 demand: np.ndarray,
                 downstream_density: np.ndarray,
                 params: Dict[str, np.ndarray],
                 lanes: Dict[int, int],
                 real_data: bool = False,
                 ) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Compute a single simulation step (t -> t+1) for METANET using total densities.

    Args:
        t: current time index - used to compute state at t+1
        density_t: array of shape (num_segments,) - TOTAL densities at time t
        velocity_t: arrays of shape (num_segments,) at time t
        queue_t, flow_origin_t: scalars at time t
        T, l: discretization time and segment length
        vsl_speeds: (time_steps, num_segments) control speeds
        demand: (time_steps,) exogenous demand at origin
        downstream_density: (time_steps,) boundary TOTAL density at downstream end
        params: dict with keys 'beta','r','eta_high','K','tau','a','rho_crit','v_free','q_capacity'
        lanes: dict mapping segment index -> number of lanes
        real_data: if True, use demand directly for first cell inflow

    Returns:
        density_tp1, velocity_tp1, queue_tp1, flow_origin_tp1, flow_tp1
        All densities are TOTAL densities, flows are total flows
    """
    num_segments = density_t.shape[0]
    density_tp1 = np.empty_like(density_t, dtype=float)

    # --- density update ---
    for i in range(num_segments):
        beta = _get_time_space_param(params["beta"], t if np.ndim(params["beta"]) == 2 else 0, i)
        r = _get_time_space_param(params["r"], t if np.ndim(params["r"]) == 2 else 0, i)

        if i == 0:
            inflow = demand[t] if real_data else flow_origin_t
            outflow = density_t[i] * velocity_t[i]
            density_tp1[i] = density_dynamics(density_t[i], inflow, outflow, T, l, beta=beta, r=r)
        else:
            inflow = density_t[i - 1] * velocity_t[i - 1]
            outflow = density_t[i] * velocity_t[i]
            density_tp1[i] = density_dynamics(density_t[i], inflow, outflow, T, l, beta=beta, r=r)

    # --- velocity update ---
    velocity_tp1 = np.empty_like(velocity_t, dtype=float)
    for i in range(num_segments):
        kwargs = dict(
            eta_high=_get_time_space_param(params["eta_high"], t, i),
            K=_get_time_space_param(params["K"], t, i),
            tau=_get_time_space_param(params["tau"], t, i),
            a=_get_time_space_param(params["a"], t, i),
            rho_crit=_get_time_space_param(params["rho_crit"], t, i),
            v_free=_get_time_space_param(params["v_free"], t, i),
        )
        if i == 0:
            prev_vel = velocity_t[i]
            next_dens = density_t[i + 1] if num_segments > 1 else downstream_density[t]
            lanes_next = lanes[i + 1] if num_segments > 1 else lanes[i]
            velocity_tp1[i] = velocity_dynamics_MN(
                velocity_t[i], prev_vel, density_t[i], next_dens, lanes[i], lanes_next, vsl_speeds[t, i], T, l, **kwargs
            )
        elif i == num_segments - 1:
            velocity_tp1[i] = velocity_dynamics_MN(
                velocity_t[i], velocity_t[i - 1], density_t[i], downstream_density[t], lanes[i], lanes[i], vsl_speeds[t, i], T, l, **kwargs
            )
        else:
            velocity_tp1[i] = velocity_dynamics_MN(
                velocity_t[i], velocity_t[i - 1], density_t[i], density_t[i + 1], lanes[i], lanes[i + 1], vsl_speeds[t, i], T, l, **kwargs
            )

    # --- queue update (only when not using real data) ---
    if not real_data:
        queue_tp1 = queue_dynamics(queue_t, demand[t], flow_origin_t, T)
    else:
        queue_tp1 = queue_t

    # --- per-segment flow at t+1 (total flow) ---
    flow_tp1 = np.empty_like(density_tp1, dtype=float)
    for i in range(num_segments):
        flow_tp1[i] = flow_dynamics(density_tp1[i], velocity_tp1[i])

    # --- origin flow remains unchanged ---
    flow_origin_tp1 = flow_origin_t

    return density_tp1, velocity_tp1, queue_tp1, flow_origin_tp1, flow_tp1


def run_metanet_sim(T: float,
                    l: float,
                    init_traffic_state: Tuple[np.ndarray, np.ndarray, float, float],
                    demand: np.ndarray,
                    downstream_density: np.ndarray,
                    params: Dict[str, np.ndarray],
                    vsl_speeds: Optional[np.ndarray] = None,
                    lanes: Optional[Dict[int, int]] = None,
                    plotting: bool = False,
                    real_data: bool = False,
                    opt: bool = False):
    """Run a METANET simulation using TOTAL densities."""
    time_steps = downstream_density.shape[0]
    num_segments = init_traffic_state[0].shape[0]

    if vsl_speeds is None:
        vsl_speeds = np.full((time_steps, num_segments), 1000)
    if lanes is None or len(lanes) == 0:
        lanes = {i: 1 for i in range(num_segments)}

    initial_density, initial_velocity, initial_flow_or, initial_queue = init_traffic_state
    
    # Allocate histories
    density = np.zeros((time_steps + 1, num_segments), dtype=float)
    velocity = np.zeros((time_steps + 1, num_segments), dtype=float)
    flow = np.zeros((time_steps + 1, num_segments), dtype=float)
    queue = np.zeros((time_steps + 1, 1), dtype=float)
    flow_origin = np.zeros((time_steps + 1, 1), dtype=float)

    # Initial conditions
    density[0] = initial_density
    velocity[0] = initial_velocity
    flow[0] = np.array([initial_density[i] * initial_velocity[i] for i in range(num_segments)], dtype=float)
    flow_origin[0, 0] = initial_flow_or
    queue[0, 0] = initial_queue

    # Main loop: metanet_step(t) computes state[t+1] from state[t] using params[t]
    for t in range(time_steps):
        d_tp1, v_tp1, q_tp1, fo_tp1, f_tp1 = metanet_step(
            t,
            density[t],
            velocity[t],
            queue[t, 0],
            flow_origin[t, 0],
            T=T,
            l=l,
            vsl_speeds=vsl_speeds,
            demand=demand,
            downstream_density=downstream_density,
            params=params,
            lanes=lanes,
            real_data=real_data
        )
        density[t + 1] = d_tp1
        velocity[t + 1] = v_tp1
        queue[t + 1, 0] = q_tp1
        flow[t + 1] = f_tp1
        flow_origin[t + 1, 0] = fo_tp1

    # Compute total travel time using total density
    total_travel_time = T * (
        sum([np.sum(density[:, i]) * l for i in range(num_segments)])
        + np.sum(queue)
    )

    if plotting:
        return density, velocity, queue, total_travel_time
    elif opt:
        # For V_fd calculation, need per-lane density
        density_per_lane = np.array([density[:-1, i] / lanes[i] for i in range(num_segments)]).T
        V_fd = calculate_V_arr(
            density_per_lane,
            vsl_speeds,
            params["a"][0],
            params["rho_crit"][0],
            params["v_free"][0],
        )
        return density, velocity, queue, flow_origin, V_fd, total_travel_time
    else:
        final_tuple = (density[-1], velocity[-1], float(flow_origin[-1, 0]), float(queue[-1, 0]))
        return final_tuple, total_travel_time