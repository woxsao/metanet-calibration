import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ijson
import seaborn as sns


def compute_speed(timestamps, x_positions, y_positions):
    """
    Computes speed in km/h from position and time data.

    Args:
        timestamps (np.ndarray): Array of timestamps in seconds.
        x_positions (np.ndarray): Array of x positions in meters.
        y_positions (np.ndarray): Array of y positions in meters.

    Returns:
        np.ndarray: Speed in km/h.
    """
    time_diffs = np.diff(timestamps)
    time_diffs[time_diffs == 0] = np.nan  # Avoid division by zero

    dx = np.diff(x_positions)
    dy = np.diff(y_positions)
    displacements = np.sqrt(dx**2 + dy**2)  # Euclidean distance

    speeds = displacements / time_diffs  # m/s
    speeds *= 3.6  # Convert to km/h

    return np.insert(speeds, 0, np.nan)  # Align size by inserting NaN at the start


def load_trajectories(
    file_path,
    trajectory_timeframe=pd.Timedelta(minutes=10),
    min_time=None,
    direction_str="west",
):
    """
    Loads trajectories from a given file path, filters by direction and time range.

    Args:
        file_path (str): Path to the file containing the trajectories.
        trajectory_timeframe (pd.Timedelta): Time range for which to load trajectories. Default is 10 minutes.
        min_time (pd.Timestamp): Minimum time for which to load trajectories. Default is None.
        direction_str (str): Direction for which to load trajectories. Default is "west".

    Returns:
        pd.DataFrame: DataFrame containing the loaded trajectories, with columns "trajectory_id", "timestamp", "x_position", "y_position", and "speed".
    """
    if direction_str == "west":
        direction_num = -1
    if direction_str == "east":
        direction_num = 1
    westbound_trajectories = []
    t_min = None
    t_max = None
    MIN_MILE_MARKER = 58.8 * 5280 * 0.3048  # meters
    MAX_MILE_MARKER = 62.8 * 5280 * 0.3048  # 2800 meters
    # Open file and stream data
    with open(file_path, "r") as f:
        trajectory_iterator = ijson.items(f, "item")

        for traj in trajectory_iterator:
            # Mile marker 61 is 322080 feet or 98170 m
            # Mile marker 62 is 327360 feet or 99779.3 m
            x_positions = (
                np.array(traj.get("x_position", []), dtype=np.float32) * 0.3048
            )  # Convert feet to meters
            y_positions = (
                np.array(traj.get("y_position", []), dtype=np.float32) * 0.3048
            )  # Convert feet to meters
            direction = traj.get("direction")

            if len(x_positions) > 1 and direction == direction_num:
                timestamps = np.array(traj.get("timestamp", []), dtype=np.float64)
                timestamps = (
                    pd.to_datetime(timestamps, unit="s").astype(np.int64) / 1e9
                )  # Convert to seconds

                if min_time and (timestamps[0] < min_time.timestamp()):
                    continue

                westbound_trajectories.append(
                    {
                        "trajectory": traj,
                        "timestamps": timestamps,
                        "x_positions": x_positions,
                        "y_positions": y_positions,
                    }
                )

                # Efficient min/max tracking
                t_min = timestamps[0] if t_min is None else min(t_min, timestamps[0])
                t_max = timestamps[0] if t_max is None else max(t_max, timestamps[0])

                if (
                    t_max is not None
                    and t_min is not None
                    and (t_max - t_min) > trajectory_timeframe.total_seconds()
                ):
                    break

    print(f"Loaded {len(westbound_trajectories)} westbound trajectories.")

    if not westbound_trajectories:
        return pd.DataFrame(
            columns=["trajectory_id", "timestamp", "x_position", "speed"]
        )

    # Vectorized DataFrame creation
    all_trajectory_ids = []
    all_timestamps = []
    all_x_positions = []
    all_y_positions = []

    for idx, traj in enumerate(westbound_trajectories):
        mask = (traj["x_positions"] >= MIN_MILE_MARKER) & (
            traj["x_positions"] <= MAX_MILE_MARKER
        )

        filtered_timestamps = traj["timestamps"][mask]
        filtered_x_positions = traj["x_positions"][mask]
        filtered_y_positions = traj["y_positions"][mask]

        num_points = len(filtered_timestamps)
        all_trajectory_ids.extend([idx] * num_points)
        all_timestamps.extend(filtered_timestamps)
        all_x_positions.extend(filtered_x_positions)
        all_y_positions.extend(filtered_y_positions)
    df = pd.DataFrame(
        {
            "trajectory_id": np.array(all_trajectory_ids, dtype=np.int32),
            "timestamp": pd.to_datetime(all_timestamps, unit="s"),
            "x_position": np.array(all_x_positions, dtype=np.float32),
            "y_position": np.array(all_y_positions, dtype=np.float32),
        }
    )

    print(df.columns.tolist())
    print(df)

    return df


def get_flow_density_matrix(
    df,
    time_interval=pd.Timedelta(minutes=1),
    space_interval=100,
    output_filename="output.csv",
):
    """
    Computes flow and density matrices from a given DataFrame, given as follows:

    Args:
        df (pd.DataFrame): DataFrame containing the trajectories, with columns "trajectory_id", "timestamp", "x_position", "speed".
        time_interval (pd.Timedelta): Time interval for which to compute the flow and density matrices. Default is 1 minute.
        space_interval (int): Space interval for which to compute the flow and density matrices. Default is 100 meters.
        output_filename (str): Output filename for the flow and density matrices. Default is "output.csv".

    Returns:
        np.ndarray: Flow matrix.
        np.ndarray: Density matrix.
    """
    t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
    x_min, x_max = df["x_position"].min(), df["x_position"].max()

    print("xmax", x_max)
    print("x_min", x_min)
    # Ensure valid ranges
    if x_min == x_max:
        raise ValueError(
            "x_min and x_max are identical, meaning no variation in x_position."
        )

    # Create time and space bins
    time_bins = pd.date_range(start=t_min, end=t_max, freq=time_interval)
    space_bins = np.arange(x_min, x_max + space_interval, space_interval)

    if len(space_bins) < 2:
        raise ValueError(
            "space_bins array is empty or too small, adjust space_interval."
        )

    # Assign bin indices using `pd.cut()`
    df["time_bin"] = pd.cut(
        df["timestamp"], bins=time_bins, labels=False, include_lowest=True
    )
    df["space_bin"] = pd.cut(
        df["x_position"], bins=space_bins, labels=False, include_lowest=True
    )

    # Remove NaNs (out-of-range values)
    df = df.dropna(subset=["time_bin", "space_bin"]).astype(
        {"time_bin": int, "space_bin": int}
    )

    # Compute flow and density using `groupby()`
    flow_matrix = np.zeros((len(time_bins) - 1, len(space_bins) - 1))
    density_matrix = np.zeros_like(flow_matrix)
    lane_matrix = np.zeros_like(flow_matrix)

    grouped = df.groupby(["time_bin", "space_bin"])
    area_bin = (
        (space_interval / 1000.0) * time_interval.total_seconds() / 3600.0
    )  # convert space interval to kilometers, time_interval to hours
    for (time_bin, space_bin), group in grouped:
        # print(time_bin, space_bin)
        traj_group = group.groupby("trajectory_id")
        traj_dict = {traj_id: traj_data for traj_id, traj_data in traj_group}

        total_distance = sum(
            traj_group["x_position"].apply(lambda x: x.max() - x.min())
        )
        total_time = sum(
            traj_group["timestamp"].apply(lambda x: (x.max() - x.min()).total_seconds())
        )

        flow_matrix[time_bin, space_bin] = (total_distance / (1000.0)) / area_bin
        density_matrix[time_bin, space_bin] = (total_time / (3600.0)) / area_bin
    # Plot histogram of y_position for each space_bin
    space_grouped = df.groupby("space_bin")
    for space_bin, group in space_grouped:
        plt.figure(figsize=(8, 5))
        plt.hist(group["y_position"], bins=30, alpha=0.7)
        plt.title(f"Histogram of y_position for space_bin {space_bin}")
        plt.xlabel("y_position")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"data/histogram_ypos_spacebin_{space_bin}.png")  # Save to file
        plt.close()
    print(grouped)

    full_filepath = "data/" + output_filename
    form_csv(
        flow_matrix,
        density_matrix,
        time_interval,
        space_interval,
        t_min,
        t_max,
        x_min,
        x_max,
        lane_matrix,
        full_filepath,
    )
    return flow_matrix, density_matrix


def form_csv(
    flow_matrix,
    density_matrix,
    time_increment,
    space_increment,
    t_min,
    t_max,
    x_min,
    x_max,
    y_position_ranges,
    output_filename="output.csv",
):
    """
    Save the flow and density matrices to a CSV file.

    Parameters:
    flow_matrix (np.ndarray): Flow matrix.
    density_matrix (np.ndarray): Density matrix.
    time_increment (pd.Timedelta): Time interval.
    space_increment (float): Space interval.
    t_min (pd.Timestamp): Minimum timestamp.
    t_max (pd.Timestamp): Maximum timestamp.
    x_min (float): Minimum x_position.
    x_max (float): Maximum x_position.
    y_position_ranges (np.ndarray): y_position ranges for each space bin.
    output_filename (str): Output filename (default: "output.csv").

    Returns:
    None
    """
    num_time_bins, num_space_bins = flow_matrix.shape

    time_values = np.array([t_min + i * time_increment for i in range(num_time_bins)])
    space_values = np.array(
        [x_min + i * space_increment for i in range(num_space_bins)]
    )

    time_grid, space_grid = np.meshgrid(time_values, space_values, indexing="ij")

    df = pd.DataFrame(
        {
            "Time": time_grid.ravel(),
            "Space": space_grid.ravel(),
            "Flow": flow_matrix.ravel(),
            "Density": density_matrix.ravel(),
            "y_position_range": y_position_ranges.ravel(),
        }
    )

    df.to_csv(output_filename, index=False)

    print(f"CSV file saved as {output_filename}")



def get_ramps_per_segment(ramps_path, space_interval=400, direction="west",
                          min_pos=None, max_pos=None):
    """
    Computes binary on/off ramp indicators per 400 m segment along the highway.
    Allows specifying a fixed spatial range via min_pos and max_pos (in meters).
    """
    MILE_TO_METER = 1609.34

    # Load and convert ramp coordinates
    ramps_df = pd.read_csv(ramps_path).copy()
    ramps_df["x_m"] = ramps_df["x_rcs_miles"] * MILE_TO_METER

    # Use provided min/max range if given
    if min_pos is None:
        min_pos = ramps_df["x_m"].min()
    if max_pos is None:
        max_pos = ramps_df["x_m"].max()

    # Uniform bins
    space_bins = np.arange(min_pos, max_pos + space_interval, space_interval)
    bin_midpoints_m = 0.5 * (space_bins[:-1] + space_bins[1:])

    on_ramp = np.zeros(len(space_bins) - 1, dtype=bool)
    off_ramp = np.zeros(len(space_bins) - 1, dtype=bool)

    for i in range(len(space_bins) - 1):
        seg_start, seg_end = space_bins[i], space_bins[i + 1]
        in_bin = (ramps_df["x_m"] >= seg_start) & (ramps_df["x_m"] < seg_end)

        if in_bin.any():
            if (ramps_df.loc[in_bin, "entry_node"].astype(str).str.upper() == "TRUE").any():
                on_ramp[i] = True
            if (ramps_df.loc[in_bin, "exit_node"].astype(str).str.upper() == "TRUE").any():
                off_ramp[i] = True

    if direction == "west":
        on_ramp = on_ramp[::-1]
        off_ramp = off_ramp[::-1]
        space_bins = space_bins[::-1]
        bin_midpoints_m = bin_midpoints_m[::-1]

    return on_ramp, off_ramp, space_bins, bin_midpoints_m

def get_lanes_per_segment(lanes_path, space_interval=400, direction="west",
                          min_pos=None, max_pos=None):
    """
    Computes weighted average lane count for each 400 m segment along the highway.
    Allows specifying a fixed spatial range via min_pos and max_pos (in meters).
    """
    MILE_TO_METER = 1609.34

    # Load lane data
    lanes_df = pd.read_csv(lanes_path).copy()
    lanes_df["x_start_m"] = lanes_df["x_start_mile"] * MILE_TO_METER
    lanes_df["x_end_m"] = lanes_df["x_end_mile"] * MILE_TO_METER
    lanes_df["x_min_m"] = lanes_df[["x_start_m", "x_end_m"]].min(axis=1)
    lanes_df["x_max_m"] = lanes_df[["x_start_m", "x_end_m"]].max(axis=1)
    lanes_df = lanes_df.sort_values("x_min_m").reset_index(drop=True)

    # Use provided min/max range if given
    if min_pos is None:
        min_pos = lanes_df["x_min_m"].min()
    if max_pos is None:
        max_pos = lanes_df["x_max_m"].max()

    # Uniform bins
    space_bins = np.arange(min_pos, max_pos + space_interval, space_interval)
    bin_midpoints_m = 0.5 * (space_bins[:-1] + space_bins[1:])

    lanes_per_bin = np.zeros(len(space_bins) - 1)

    for i in range(len(space_bins) - 1):
        seg_start, seg_end = space_bins[i], space_bins[i + 1]
        overlapping = lanes_df[
            (lanes_df["x_max_m"] > seg_start) & (lanes_df["x_min_m"] < seg_end)
        ]

        if overlapping.empty:
            lanes_per_bin[i] = lanes_per_bin[i - 1] if i > 0 else np.nan
            continue

        overlap_lengths = np.minimum(overlapping["x_max_m"], seg_end) \
                        - np.maximum(overlapping["x_min_m"], seg_start)
        overlap_lengths = np.clip(overlap_lengths, 0, None)

        if np.any(overlap_lengths > 0):
            lanes_per_bin[i] = np.average(overlapping["lanes"], weights=overlap_lengths)
        else:
            lanes_per_bin[i] = overlapping["lanes"].mean()

    if direction == "west":
        lanes_per_bin = lanes_per_bin[::-1]
        space_bins = space_bins[::-1]
        bin_midpoints_m = bin_midpoints_m[::-1]

    return lanes_per_bin, space_bins, bin_midpoints_m


def plot_simulation_comparison(rho_sim, v_sim, rho_true, v_true, q_true=None, include_fd=True, save_path=None, lanes=None):
    """
    Plot side-by-side comparison of simulated vs ground truth traffic states.
    Uses PREDICTED values for color scale (vmin/vmax) to make differences more visible.
    
    Args:
        rho_sim: Simulated density (time_steps, num_segments) - TOTAL density
        v_sim: Simulated velocity (time_steps, num_segments)
        rho_true: Ground truth density (time_steps, num_segments) - TOTAL density
        v_true: Ground truth velocity (time_steps, num_segments)
        q_true: Ground truth flow (time_steps, num_segments), optional
        include_fd: If True, add fundamental diagram plot at bottom
        save_path: Path to save figure
        lanes: Array of lanes per segment (needed for FD conversion)
    """
    # Compute flows
    q_sim = rho_sim * v_sim
    if q_true is None:
        q_true = rho_true * v_true
    
    # Create figure
    if include_fd:
        fig = plt.figure(figsize=(14, 18))
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.8])
        axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(3)]
        ax_fd = fig.add_subplot(gs[3, :])
    else:
        fig, axes = plt.subplots(3, 2, figsize=(14, 16), sharey='row')
    
    # --- Row 1: Velocity ---
    # USE PREDICTED VALUES FOR COLOR SCALE
    v_min = v_sim.min()
    v_max = v_sim.max()
    
    im0 = axes[0][0].imshow(
        v_sim.T, aspect="auto", origin="lower", cmap="RdYlGn", 
        interpolation="none", vmin=v_min, vmax=v_max
    )
    axes[0][0].set_xlabel("Time Step")
    axes[0][0].set_ylabel("Segment Index")
    axes[0][0].set_title("Predicted Velocity")
    fig.colorbar(im0, ax=axes[0][0], label="Velocity (km/h)")
    
    im1 = axes[0][1].imshow(
        v_true.T, aspect="auto", origin="lower", cmap="RdYlGn", 
        interpolation="none", vmin=v_min, vmax=v_max
    )
    axes[0][1].set_xlabel("Time Step")
    axes[0][1].set_title("Ground Truth Velocity")
    fig.colorbar(im1, ax=axes[0][1], label="Velocity (km/h)")
    
    # --- Row 2: Flow ---
    # USE PREDICTED VALUES FOR COLOR SCALE
    q_min = q_sim.min()
    q_max = q_sim.max()
    
    im2 = axes[1][0].imshow(
        q_sim.T, aspect="auto", origin="lower", cmap="RdYlGn", 
        interpolation="none", vmin=q_min, vmax=q_max
    )
    axes[1][0].set_xlabel("Time Step")
    axes[1][0].set_ylabel("Segment Index")
    axes[1][0].set_title("Predicted Flow")
    fig.colorbar(im2, ax=axes[1][0], label="Flow (veh/h)")
    
    im3 = axes[1][1].imshow(
        q_true.T, aspect="auto", origin="lower", cmap="RdYlGn", 
        interpolation="none", vmin=q_min, vmax=q_max
    )
    axes[1][1].set_xlabel("Time Step")
    axes[1][1].set_title("Ground Truth Flow")
    fig.colorbar(im3, ax=axes[1][1], label="Flow (veh/h)")
    
    # --- Row 3: Density ---
    # USE PREDICTED VALUES FOR COLOR SCALE
    rho_min = rho_sim.min()
    rho_max = rho_sim.max()
    
    im4 = axes[2][0].imshow(
        rho_sim.T, aspect="auto", origin="lower", cmap="RdYlGn", 
        interpolation="none", vmin=rho_min, vmax=rho_max
    )
    axes[2][0].set_xlabel("Time Step")
    axes[2][0].set_ylabel("Segment Index")
    axes[2][0].set_title("Predicted Density")
    fig.colorbar(im4, ax=axes[2][0], label="Density (veh)")
    
    im5 = axes[2][1].imshow(
        rho_true.T, aspect="auto", origin="lower", cmap="RdYlGn", 
        interpolation="none", vmin=rho_min, vmax=rho_max
    )
    axes[2][1].set_xlabel("Time Step")
    axes[2][1].set_title("Ground Truth Density")
    fig.colorbar(im5, ax=axes[2][1], label="Density (veh)")
    
    # --- Row 4: Fundamental Diagram ---
    if include_fd:
        # CRITICAL FIX: Convert TOTAL density to PER-LANE density for FD plot
        if lanes is not None:
            # Expand lanes to match shape
            num_timesteps, num_segments = rho_sim.shape
            lanes_expanded_sim = np.tile(lanes, (num_timesteps, 1))
            lanes_expanded_true = np.tile(lanes, (rho_true.shape[0], 1))
            
            # Convert to per-lane density
            rho_per_lane_sim = rho_sim / lanes_expanded_sim
            rho_per_lane_true = rho_true / lanes_expanded_true
            
            all_rho_pred = rho_per_lane_sim.flatten()
            all_rho_true = rho_per_lane_true.flatten()
        else:
            # Fallback: assume data is already per-lane (but warn)
            print("WARNING: lanes not provided, assuming density is per-lane")
            all_rho_pred = rho_sim.flatten()
            all_rho_true = rho_true.flatten()
        
        all_q_pred = q_sim.flatten()
        all_q_true = q_true.flatten()
        
        ax_fd.scatter(all_rho_true, all_q_true, color="gray", alpha=0.7, s=1, label="Data (measured)")
        ax_fd.scatter(all_rho_pred, all_q_pred, alpha=0.6, s=1, label="Predicted")
        ax_fd.set_xlabel("Density (veh/lane/km)")
        ax_fd.set_ylabel("Flow (veh/h)")
        ax_fd.set_title("Fundamental Diagram: Flow vs. Density (per lane)")
        ax_fd.legend()
        ax_fd.grid(True)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()