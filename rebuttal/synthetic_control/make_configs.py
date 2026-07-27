"""
Generate the rank-sweep training configs for the synthetic-control experiment.

For each intrinsic bias rank K in {1,2,4} (dataset data/processed/mogi_synth/K{K})
and each model residual rank k in {0,1,2,4,6,8}, clone the established real-data
rank-ablation config (dim_z_aux == residual_rank, edge_penalty=10, ortho=0.1) and
swap only: name, save_dir, data paths, dim_z_aux, residual_rank.

k=0 => dim_z_aux=0 => pure physics, no residual (baseline: bias fully uncorrected).

Configs are written to rebuttal/configs/. Run:  python rebuttal/make_configs.py
"""
import os
import json
import copy

HERE = os.path.dirname(os.path.abspath(__file__))     # rebuttal/synthetic_control
REPO = os.path.dirname(os.path.dirname(HERE))         # repo root
TEMPLATE = os.path.join(
    REPO, "saved/mogi/PHYS_VAE_MOGI_C_SMPL/1016_191930_rank4/models/config.json")
OUT_DIR = os.path.join(HERE, "configs")

K_LIST = [1, 2, 4]              # intrinsic bias rank of the dataset
RANK_LIST = [0, 1, 2, 4, 6, 8]  # model residual rank
EPOCHS = 150                    # match the real ablation


def make_one(template, K, k, epochs):
    c = copy.deepcopy(template)
    dd = f"data/processed/mogi_synth/K{K}"
    c["name"] = f"SYN_K{K}_r{k}"
    c["trainer"]["save_dir"] = "saved/mogi_synth"
    c["trainer"]["epochs"] = epochs
    c["arch"]["args"]["standardization"]["x_mean"] = f"{dd}/train_x_mean.npy"
    c["arch"]["args"]["standardization"]["x_scale"] = f"{dd}/train_x_scale.npy"
    c["data_loader"]["args"]["data_dir"] = f"{dd}/train.csv"
    c["data_loader"]["data_dir_valid"] = f"{dd}/valid.csv"
    c["data_loader"]["data_dir_test"] = f"{dd}/test.csv"
    # rank knob: dim_z_aux == residual_rank (their convention). k=0 => no residual.
    c["arch"]["phys_vae"]["dim_z_aux"] = k
    c["arch"]["phys_vae"]["residual_rank"] = max(k, 1)  # unused when dim_z_aux=0
    return c


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    template = json.load(open(TEMPLATE))
    n = 0
    for K in K_LIST:
        for k in RANK_LIST:
            cfg = make_one(template, K, k, EPOCHS)
            path = os.path.join(OUT_DIR, f"SYN_K{K}_r{k}.json")
            json.dump(cfg, open(path, "w"), indent=2)
            n += 1
    print(f"wrote {n} configs to {OUT_DIR} (K in {K_LIST} x rank in {RANK_LIST})")


if __name__ == "__main__":
    main()
