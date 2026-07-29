"""
Generate the HVAE + end-stop configs (reviewer Q: architecture or prior?).

Reviewer: "The authors show that using a different prior in PILA reduces the
quality of the output. What about using the PILA end-stop loss in the HVAE method
- would that improve it? Basically, the question here is whether the benefits of
PILA come from the architecture or the prior."

The study runs on GreenHyperSpectra, not Austria, because GHS carries measured
traits: the answer is then a per-trait R2 rather than a plausibility argument.
Data, latent ranges and evaluation are shared with rebuttal/greenhyperspectra/
(read-only here -- nothing in that folder is modified).

HVAE's architecture is left exactly as published: Softplus mean, Gaussian draw,
clamp to [0,1]. Only the loss changes, along two axes:

  KL on z_phy      the N(0.5, 0.866) prior HVAE places on the physical latents
  end-stop         PILA's barrier -log(z) - log(1-z), R_end

  E0_base                    KL on,  no end-stop     the published HVAE
  E1_endstop                 KL on,  end-stop 1.0    does R_end help HVAE as-is?
  E2_noklphy                 KL off, no end-stop     control: is the prior load-bearing?
  E3_noklphy_endstop         KL off, end-stop 1.0    PILA's loss on HVAE's architecture
  E4_noklphy_endstop_w0.1    KL off, end-stop 0.1    weight sensitivity of E3

Read the result as: if E3 reaches PILA-C, the benefit is the loss; if it does
not, the benefit is the architecture. E4 exists so that a negative E3 cannot be
dismissed as a badly chosen weight.

Note on the end-stop weight. PILA uses 1.0 on GHS, where its rec_loss settles
around 0.9 and the barrier around 1.8, i.e. R_end is ~2x reconstruction. HVAE's
full-rank correction drives rec_loss to ~0.05, so the same absolute 1.0 makes
R_end ~30x reconstruction. E3 keeps the literal transplant (1.0) and E4 adds
0.1, which is the weight that reproduces PILA's *relative* balance.

Note on the KL switch. HVAE weights its KL by `x_var = exp(x_lnvar) = e^-9`, so
the effective weight is ~1.2e-4 even at balance_kld = 1.0; the prior is already
close to inert, and it is the clamp that keeps z_phy in range. The trainer logs
`z_phy_clamp_frac` so this can be quoted rather than assumed.

Usage:  python rebuttal/prior_vs_endstop/make_configs.py
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, 'configs')

# Read-only source: the HVAE variant-C config of the GHS experiment.
SOURCE = 'rebuttal/greenhyperspectra/configs/HVAE_PROSAIL_C_ghs.json'
SAVE_DIR = 'saved/rebuttal/prior_vs_endstop'

# One run: PILA's loss (no prior on z_phy, end-stop instead) on HVAE's
# architecture. The baseline it is compared against is the existing
# HVAE_PROSAIL_C_GHS run under saved/greenhyperspectra/ -- same config, same data,
# so it is not retrained here.
#
# (suffix, kl_z_phy_off, end-stop enabled)
VARIANTS = [
    ('endstop', True, True),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    template = json.load(open(os.path.join(REPO, SOURCE)))

    for suffix, kl_off, edge_on in VARIANTS:
        cfg = copy.deepcopy(template)
        cfg['name'] = f"{template['name']}_{suffix}"
        cfg['trainer']['save_dir'] = SAVE_DIR
        pv = cfg['trainer']['phys_vae']

        if kl_off:
            # Per-latent KL weights. z_phy's prior is switched off; z_aux2 keeps
            # the weight it had (balance_kld + balance_lact_enc), so the only
            # thing that changes is the prior on the *physical* latents.
            pv['balance_kld_z_phy'] = 0.0
            pv['balance_kld_z_aux'] = pv['balance_kld'] + pv['balance_lact_enc']

        if edge_on:
            # Dynamically weighted, like HVAE's other regularisers: the trainer
            # rescales it to sit an order of magnitude below rec_loss each step,
            # so nothing has to be hand-tuned and the term cannot run away as
            # rec_loss falls. The weight below is then just the on switch.
            pv['edge_penalty_weight'] = 1.0
            pv['edge_penalty_dynamic'] = True
            pv['edge_penalty_power'] = 1.0
            # Penalise the pre-clamp draw: on the clamped value the barrier is
            # constant exactly at the end-stops, so it would have no gradient
            # where it is meant to act. See model.loss.end_stop_penalty.
            pv['edge_penalty_target'] = 'raw'
            pv['edge_penalty_eps'] = 1e-2

        path = os.path.join(OUT, f"{cfg['name']}.json")
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=4)
        print(f"wrote {os.path.relpath(path, REPO)}"
              f"  (KL_z_phy {'off' if kl_off else 'on'}, end-stop {'dynamic' if edge_on else 'off'})")

        # 2-epoch smoke variant, under its own experiment name so it can never
        # land next to a real run.
        if True:
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
