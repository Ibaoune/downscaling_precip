"""
==========================================================
 Script: config.py
 Author: M. El Aabaribaoune
 Description:
     Loads configuration from YAML and exposes attributes
     as a simple namespace with robust type casting
     (int / float / bool / string safe).
==========================================================
"""

import yaml
import torch
import os


# ==========================================================
# Helper: safe casting
# ==========================================================
def _to_int(x):
    return int(x) if x is not None else None


def _to_float(x):
    return float(x) if x is not None else None


def _to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in ["true", "1", "yes", "y"]
    return bool(x)


# ==========================================================
# Config class
# ==========================================================
class Config:
    def __init__(self, cfg_dict, train_mode=True):
        self.train_mode = train_mode

        # ----------------------
        # Device
        # ----------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # ----------------------
        # General
        # ----------------------
        self.verbose = _to_bool(cfg_dict["general"].get("verbose", True))
        self.target = str(cfg_dict["general"].get("target"))
        self.src = str(cfg_dict["general"].get("src"))
        self.model_type = str(cfg_dict["general"].get("model_type", "vit"))

        # ----------------------
        # Training
        # ----------------------
        tr = cfg_dict["training"]
        self.learning_rate = _to_float(tr.get("learning_rate"))
        self.epochs = _to_int(tr.get("epochs"))
        self.batch_size = _to_int(tr.get("batch_size"))
        self.loss_type = str(tr.get("loss_type"))
        self.norm_mode = str(tr.get("norm_mode"))
        self.early_stopping_max = _to_int(tr.get("early_stopping_max"))

        # ----------------------
        # ViT parameters
        # ----------------------
        vit = cfg_dict["vit"]
        self.emb_size = _to_int(vit.get("emb_size"))
        self.patch_size = _to_int(vit.get("patch_size"))
        self.num_layers = _to_int(vit.get("num_layers"))
        self.num_heads = _to_int(vit.get("num_heads"))
        self.dropout = _to_float(vit.get("dropout"))

        # ----------------------
        # Region
        # ----------------------
        reg = cfg_dict["region"]
        self.lon_min = _to_float(reg.get("lon_min"))
        self.lon_max = _to_float(reg.get("lon_max"))
        self.lat_min = _to_float(reg.get("lat_min"))
        self.lat_max = _to_float(reg.get("lat_max"))

        # ----------------------
        # Dates
        # ----------------------
        self.start_date_train = str(cfg_dict["dates"]["train"]["start"])
        self.end_date_train = str(cfg_dict["dates"]["train"]["end"])
        self.start_date_test = str(cfg_dict["dates"]["test"]["start"])
        self.end_date_test = str(cfg_dict["dates"]["test"]["end"])

        # ----------------------
        # Paths
        # ----------------------
        paths = cfg_dict["paths"]
        self.data_path = str(paths["data_path"])
        self.model_save_dir = str(paths["model_save_dir"])
        self.path_out = str(paths["eval_out_dir"])

        os.makedirs("results", exist_ok=True)


# ==========================================================
# YAML loader
# ==========================================================
def load_config(train_mode=True, path="config.yaml"):
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return Config(cfg_dict, train_mode=train_mode)

