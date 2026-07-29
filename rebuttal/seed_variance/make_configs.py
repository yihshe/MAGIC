"""
Generate the multi-seed Austria configs (reviewer Q: run-to-run variance in Fig. 8).

Reviewer: "The difference observed in Figure 8 seems relatively small and could be
anecdotal without multiple runs. If this plot is included, it might be useful to
include variance between runs."

Each config is a verbatim copy of the published Austria config
  configs/phys_smpl/PILA_RTM_C_austria.json   (PILA)
  configs/phys/HVAE_RTM_C_austria.json        (HVAE)
with exactly three fields changed: `name`, `seed`, and `trainer.save_dir`.
Nothing else -- no architecture, weight or schedule differs from the paper runs.

`seed` is read by train_pila.py / train_hvae.py (both re-seed torch and numpy when
it differs from their built-in SEED=123), which covers parameter initialisation,
batch shuffling and the reparameterisation noise. Seed 123 reproduces the paper
run exactly, so it is included as the first member of the sweep rather than
re-used from saved/ -- that way all three runs come from one code state.

Usage:  python rebuttal/seed_variance/make_configs.py
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, 'configs')

# 123 is the seed hardcoded in the train scripts, i.e. the published run.
SEEDS = [123, 240, 318]

SOURCES = {
    'PILA': 'configs/phys_smpl/PILA_RTM_C_austria.json',
    'HVAE': 'configs/phys/HVAE_RTM_C_austria.json',
}

SAVE_DIR = 'saved/rebuttal/seed_variance'


def main():
    os.makedirs(OUT, exist_ok=True)
    for fam, src in SOURCES.items():
        template = json.load(open(os.path.join(REPO, src)))
        for seed in SEEDS:
            cfg = copy.deepcopy(template)
            cfg['name'] = f"{template['name']}_s{seed}"
            cfg['seed'] = seed
            cfg['trainer']['save_dir'] = SAVE_DIR
            path = os.path.join(OUT, f"{cfg['name']}.json")
            with open(path, 'w') as f:
                json.dump(cfg, f, indent=4)
            print(f'wrote {os.path.relpath(path, REPO)}  (from {src}, seed={seed})')


if __name__ == '__main__':
    main()
