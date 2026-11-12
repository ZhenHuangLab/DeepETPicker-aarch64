# DeepETPicker — Linux aarch64 + CUDA 13 + GB10 (sm_121)

Upstream: https://github.com/cbmi-group/DeepETPicker

## Result: 🟢 PASS — 7 patches, **none of them architecture-related**

The table's earlier verdict ("torch==1.7.1 too old, forcing 1.13.1 errors out,
conflicts with requirement.txt") was right about the symptom and wrong about the
cause. Nothing here is aarch64-specific: every failure is a 2021-era dependency
pin or an API that modern PyTorch/Lightning removed.

## Environment — do not use `requirement.txt` verbatim

```bash
mamba create -y -p ./env -c conda-forge python=3.11
./env/bin/pip install torch torchvision                      # 2.13.0+cu130 aarch64
mamba install -y -p ./env -c conda-forge pyqt pyqtgraph matplotlib pandas \
              scikit-image pillow opencv
./env/bin/pip install mrcfile pycm scikit-plot batchgenerators tinyaes \
              "pytorch-lightning>=2.2" tensorboard einops
```

**PyQt5 must come from conda-forge, not pip.** PyPI has no aarch64 wheel for
PyQt5 5.15.x, so pip falls back to the sdist and its SIP metadata step fails
(`Preparing metadata (pyproject.toml) did not run successfully`). conda-forge's
`pyqt` 5.15.11 installs cleanly. Same story for `opencv` (use conda's `opencv`,
not `opencv-python`).

## Patch 1 — `utils/misc.py`: pycm >= 4 module rename

```python
try:  # pycm >= 4 renamed pycm_output/pycm_param to output/params
    from pycm.output import table_print, stat_print
    from pycm.params import SUMMARY_CLASS, SUMMARY_OVERALL
except ImportError:  # pycm 3.x
    from pycm.pycm_output import table_print, stat_print
    from pycm.pycm_param import SUMMARY_CLASS, SUMMARY_OVERALL
```

## Patch 2 — `utils/misc.py`: make scikit-plot optional

`scikit-plot` 0.3.7 is unmaintained and does `from scipy import interp`, removed
in SciPy >= 1.12, so importing it kills the whole package. It is used for exactly
one optional multi-class confusion-matrix figure, so it is now guarded:

```python
try:
    import scikitplot as skplt
except ImportError:
    skplt = None
...
        if skplt is not None:
            skplt.metrics.plot_confusion_matrix(...)
```

## Patch 3 — `model/residual_unet_att.py`: in-place residual add (3 sites)

This is the real reason "forcing torch 1.13.1 errors out":

```text
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation: [torch.cuda.FloatTensor [2, 24, 72, 72, 72]],
which is output 0 of Relu, is at version 1; expected version 0 instead.
```

`SingleConv` ends in a ReLU, and ReLU's backward needs its own output. `out +=
residual` mutates that output in place, which modern autograd rejects.

```python
        out = out + residual        # was: out += residual
        out = self.non_linearity(out)
```

Applied in `ExtResNetBlock`, `ExtResNetBlock_att` and the light-weight variant.

## Patch 4 — `train.py`: PyTorch Lightning 2.x hooks

`validation_step_end` is gone and `validation_epoch_end` no longer receives the
step outputs, so they are buffered manually:

```python
    def on_validation_epoch_start(self):
        self._val_outputs = []

    def validation_step(self, val_batch, batch_idx):
        out = self._validation_step_impl(val_batch, batch_idx)
        if out is not None:
            self._val_outputs.append(out)
        return out

    def _validation_step_impl(self, val_batch, batch_idx):   # the original body
        ...

    def on_validation_epoch_end(self):                       # was validation_epoch_end
        epoch_output = self._val_outputs
        if not epoch_output:
            return
        ...
```

`test.py` gets the identical treatment (`on_test_epoch_start` /
`_test_step_impl` / `on_test_epoch_end`).

## Patch 5 — `train.py` / `test.py`: Trainer keyword migration

```python
    gpu_ids = args.gpu_id if isinstance(args.gpu_id, (list, tuple)) else [args.gpu_id]
    runner = Trainer(min_epochs=min(50, args.max_epoch),
                     max_epochs=args.max_epoch,
                     logger=tb_logger,
                     accelerator='gpu',                       # was gpus=args.gpu_id
                     devices=list(gpu_ids),
                     callbacks=[checkpoint_callback, lr_monitor],   # was checkpoint_callback=
                     strategy='ddp' if len(gpu_ids) > 1 else 'auto',  # was accelerator='dp'
                     precision=32,
                     profiler='simple',                       # was profiler=True
                     sync_batchnorm=len(gpu_ids) > 1)
    ...
    runner.fit(model, ckpt_path=args.checkpoints if args.checkpoints else None)
```

(`resume_from_checkpoint` moved from `Trainer.__init__` to `Trainer.fit`.)

## Patch 6 — `test.py`: `torch.load` weights-only default

torch >= 2.6 defaults `weights_only=True`, which refuses the `argparse.Namespace`
stored in the checkpoint hyper-parameters
(`Unsupported global: GLOBAL argparse.Namespace`):

```python
    model = UNetTest.load_from_checkpoint(args.checkpoints, weights_only=False)
```

## Patch 7 — `train.py`: stop swallowing the traceback

`runner.fit()` sat inside a bare `except:` that printed nothing and returned
normally, so **every** failure looked like a silent success — exit code 0, no
message, no model. That cost real debugging time here; it now re-raises through
`traceback.print_exc()`.

## Smoke test — synthetic tomograms, full pipeline

Two 160^3 tomograms with 60 dark Gaussian particles each.

```bash
python bin/preprocess.py            --pre_configs   /tmp/detp-smoke/pre_config.py
python bin/generate_train_config.py --pre_configs   /tmp/detp-smoke/pre_config.py \
       --dset_name smoke_train --cfg_save_path /tmp/detp-smoke \
       --train_set_ids 0 --val_set_ids 1 --batch_size 2 --block_size 72 \
       --pad_size 12 --learning_rate 0.001 --max_epoch 2 --threshold 0.5 --gpu_id 0
python bin/train_bash.py            --train_configs /tmp/detp-smoke/smoke_train.py
python bin/test_bash.py             --train_configs /tmp/detp-smoke/smoke_train.py \
                                    --checkpoints <ckpt>
```

| stage | result |
|---|---|
| preprocess | coords / standardised tomos / gaussian11 labels / occupancy maps all generated |
| train | 2 epochs, 30 steps/epoch on GPU, checkpoint written |
| test | `coords_out: (120, 5)` and `(110, 5)`, de-duplicated to 60 coordinates each |
| **accuracy** | **recall 60/60 within 10 voxels** on tomo1 after only 2 epochs |
