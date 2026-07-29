"""
Generate the PILA + physical-consistency configs (reviewer Q: would it help?).

Reviewer: "Could PILA potentially be improved through addition of a physical
consistency loss as in HVAE?"

HVAE's physical-consistency term is the synthetic-pair inversion loss: sample
z ~ U(0,1), push it through the physics model, and require the encoder to invert
it back, comparing the inferred z against the sampled z. PILA gets exactly that
comparison via PhysVAETrainerSMPL._synthetic_consistency_loss_z. The only
difference from HVAE is where the squashing sits: HVAE's encoder head ends in
Softplus and emits z directly, PILA's emits u, so the sigmoid from draw() is
applied before comparing. Both compare drawn z against sampled z.

(PILA's pre-existing _synthetic_data_loss, which regresses u against logit(z),
is the Stage-A bootstrap. It is left untouched and is inactive at
epochs_pretrain = 0.)

Run on GreenHyperSpectra so the answer is a per-trait R2 against measured traits.
Data and evaluation are shared with rebuttal/greenhyperspectra/, which is
read-only here. No baseline is trained: the weight-0 reference is the existing
PILA_PROSAIL_C_GHS run under saved/greenhyperspectra/.

  S1_w0.1    weight 0.1
  S2_w1.0    weight 1.0
  S3_w10     weight 10.0

On the weights -- measured, not guessed. In z-space the term is a sum of squared
errors over dim_z_phy = 4 latents in [0,1], so it is small: 0.32 at the end of
epoch 1, falling to 0.19 by epoch 2, against rec_loss 6.4 -> 1.9. (The u-space
Stage-A version is ~30x larger, around 10-40, because logit targets have
var = pi^2/3; weights calibrated for that term would be ~30x too small here.)
The three weights therefore span roughly 1% of rec_loss (negligible), 5-10%
(the order of magnitude at which HVAE keeps its own regularisers), and 50-100%
(dominant).

Note the band count needs no correction either way: this loss never touches the
spectral axis, and rec_loss is already normalised per band via
trainer.rec_loss_scale.

The spread is itself part of the finding: the added term brings a weight that
has to be tuned per dataset, which is the balancing burden PILA was designed to
avoid.

Caveat worth keeping in the write-up: matching synthetic pairs is not the same
as supervision. Where labels exist, semi-supervised training on the measured
traits is the better use of them; this experiment only answers the question as
the reviewer posed it.

Usage:  python rebuttal/consistency_loss/make_configs.py
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, 'configs')

# Read-only source: the PILA variant-C config of the GHS experiment.
SOURCE = 'rebuttal/greenhyperspectra/configs/PILA_PROSAIL_C_ghs.json'
SAVE_DIR = 'saved/rebuttal/consistency_loss'

# No baseline run here: the weight-0 reference is the existing PILA_PROSAIL_C_GHS
# run under saved/greenhyperspectra/ -- same config, same data.
VARIANTS = [
    ('S1_w0.1', 0.1),
    ('S2_w1.0', 1.0),
    ('S3_w10', 10.0),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    template = json.load(open(os.path.join(REPO, SOURCE)))

    for suffix, weight in VARIANTS:
        cfg = copy.deepcopy(template)
        cfg['name'] = f"{template['name']}_{suffix}"
        cfg['trainer']['save_dir'] = SAVE_DIR
        cfg['trainer']['phys_vae']['balance_data_aug_cotrain'] = weight
        path = os.path.join(OUT, f"{cfg['name']}.json")
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=4)
        print(f"wrote {os.path.relpath(path, REPO)}  (consistency weight {weight})")

        # 2-epoch smoke variant of one weighted run, under its own experiment
        # name so it can never land next to a real one.
        if suffix == 'S2_w1.0':
            dbg = copy.deepcopy(cfg)
            dbg['name'] = f"DEBUG_{cfg['name']}"
            dbg['trainer']['epochs'] = 2
            dbg['trainer']['save_period'] = 1
            dbg['trainer']['log_step'] = 5
            dbg['trainer']['wandb'] = False
            dbg['lr_scheduler']['args']['T_max'] = 2
            dpath = os.path.join(OUT, f"_debug_{cfg['name']}.json")
            with open(dpath, 'w') as f:
                json.dump(dbg, f, indent=4)
            print(f'wrote {os.path.relpath(dpath, REPO)}  (2-epoch smoke test)')


if __name__ == '__main__':
    main()
