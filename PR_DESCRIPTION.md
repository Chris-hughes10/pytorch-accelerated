# Modernize checkpoint system, fix bugs, and add training resume support

## Summary

- Fix eval loop gradient leak, `NotImplementedError`, `CallbackHandler.__iter__`, and unnecessary numpy dependency
- Add `torch.compile`-safe checkpointing and secure `torch.load` (`weights_only=True`)
- Add full training state save/resume via Accelerate's `save_state`/`load_state` (handles FSDP, RNG, scaler)
- Add `resume_from` parameter to `Trainer.train()` for seamless training continuation
- Add `SaveTrainingStateCallback` with configurable save intervals and max checkpoint cleanup
- Update `WSDCheckpointCallback`: pre-decay saves full training state (for continuation), post-decay saves portable model export
- Bump minimum Python to 3.10, accelerate to >=1.12.0
- Fix 5 pre-existing test failures, add 24 new tests (43 total passing)

## Changes by area

### Bug fixes
- Wrap eval batch loop in `torch.no_grad()` so users who override `calculate_eval_batch_loss` don't silently leak gradients
- `NotImplemented` → `NotImplementedError` in `TrainerPlaceholderValues.__sub__`
- `CallbackHandler.__iter__` now returns `iter(self.callbacks)` instead of the list itself
- Replace `numpy` import with `operator` module (`np.greater`/`np.less` → `op.gt`/`op.lt`)

### Checkpoint improvements
- `save_checkpoint` strips `torch.compile` wrapper via `get_model(keep_torch_compile=False)` — no more `_orig_mod.` key prefixes
- `load_checkpoint` uses `weights_only=True` for security
- `get_model()` accepts `keep_torch_compile` parameter

### Training state save/resume
- New `save_training_state(output_dir)` — wraps `accelerator.save_state()` plus trainer metadata (epoch, custom kwargs)
- New `load_training_state(input_dir)` — wraps `accelerator.load_state()`, returns metadata dict
- New `resume_from` parameter on `train()` — loads full state and sets epoch before training begins
- New `_set_epoch()` on `RunHistory` to support resume

### New callbacks
- **`SaveTrainingStateCallback`** — periodic full state saves with `save_every_n_epochs`, `save_every_n_steps`, `save_at_end`, and `max_checkpoints` cleanup
- **`WSDCheckpointCallback` updated** — pre-decay checkpoints now use `save_training_state()` (directory, full state for continuation); post-decay checkpoints remain as `save_checkpoint()` (.pt file, portable export); loading auto-detects directory vs file

### Dependency and packaging
- Minimum Python bumped from 3.8 → 3.10 (3.8/3.9 are EOL)
- `accelerate` bumped from >=1.3.0 → >=1.12.0 (required for `keep_torch_compile` in `unwrap_model`)
- Added Python 3.13 classifier

### Documentation
- Updated `trainer.rst` with checkpoint strategy guide (model export vs training state) and `resume_from` usage
- Added `save_training_state`, `load_training_state` to Sphinx autodoc
- Added `SaveTrainingStateCallback` and `WSDCheckpointCallback.__init__` to `callbacks.rst`

## Test plan

- [x] 5 pre-existing test failures fixed (DummyTrainer method name, Mock models, TrainerRunConfig fields, requires_grad tensor, pytest.warns)
- [x] 9 round-trip integration tests: weights, optimizer momentum, scheduler state, custom kwargs, skip flags, WSD metadata, full end-to-end corrupt-and-restore
- [x] 5 SaveTrainingStateCallback tests: save at end, every N epochs, every N steps, max checkpoint cleanup, disable save at end
- [x] 5 WSDCheckpointCallback tests: pre-decay uses save_training_state, post-decay uses save_checkpoint, directory loads training state, file loads checkpoint, training_run_end saves post-decay
- [x] 5 unit tests: compile prefix stripping, weights_only, get_model param, _set_epoch, resume_from
- [x] All 43 tests passing

### Breaking changes

- **Minimum Python 3.10** (was 3.8)
- **Minimum accelerate 1.12.0** (was 1.3.0)
- `load_checkpoint` now uses `weights_only=True` — checkpoints containing non-tensor objects in custom kwargs will need migration
- WSD pre-decay checkpoints are now directories (not .pt files) — existing .pt pre-decay checkpoints can still be loaded via the file detection path
