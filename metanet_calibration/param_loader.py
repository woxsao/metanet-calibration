import numpy as np
import pandas as pd


class METANET_Params:
    def __init__(self, path=None, num_segments=30):
        if path is not None:
            self.params = {
                "tau": np.load(f"{path}/tau.npy").reshape(-1).tolist(),
                "K": np.load(f"{path}/K.npy").reshape(-1).tolist(),
                "eta_high": np.load(f"{path}/eta_high.npy").reshape(-1).tolist(),
                "rho_crit": np.load(f"{path}/rho_crit.npy").reshape(-1).tolist(),
                "v_free": np.load(f"{path}/v_free.npy").reshape(-1).tolist(),
                "a": np.load(f"{path}/a.npy").reshape(-1).tolist(),
                "q_capacity": [2200 for i in range(num_segments)],
            }
            try:
                self.params["r"] = np.load(f"{path}/r_inflow_array.npy")
                print("Loaded r_inflow_array.npy")
                print(self.params["r"].shape)
            except:
                self.params["r"] = np.array([0 for i in range(num_segments)])
            try:
                self.params["beta"] = np.load(f"{path}/beta_array.npy")
            except:
                self.params["beta"] = np.array([0 for i in range(num_segments)])
        else:
            # Use default
            self.params = {
                "tau": [18 / 3600 for i in range(num_segments)],
                "K": [40 for i in range(num_segments)],
                "eta_high": [30 for i in range(num_segments)],
                "rho_crit": [37.45 for i in range(num_segments)],
                "v_free": [120 for i in range(num_segments)],
                "a": [1.4 for i in range(num_segments)],
                "q_capacity": [2200 for i in range(num_segments)],
                "r": [0 for i in range(num_segments)],
                "beta": [0 for i in range(num_segments)],
            }

    def get_params(self):
        return self.params

    def get_param(self, key):
        return self.params.get(key, None)
