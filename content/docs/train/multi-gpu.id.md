---
title: Multi-GPU training
seo_title: Multi-GPU training in LibreYOLO
description: >-
  Train on several GPUs with device="0,1". How the library spawns DDP workers,
  why batch is the global batch, when to set sync_bn, and the torchrun path.
lead: >-
  Multi-GPU training in LibreYOLO is PyTorch DistributedDataParallel: one
  process per GPU, each holding a full model replica and a shard of every batch,
  with gradients averaged across ranks at each step.
keywords:
  - pytorch ddp training
  - multi gpu training
  - torchrun nproc_per_node
  - distributed data parallel
  - syncbatchnorm
  - global batch size
  - nccl gloo backend
  - multi gpu windows
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # The __main__ guard is required: each spawned worker re-imports this
        # module, and without the guard it would relaunch training recursively.
        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="my-dataset.yaml",
                epochs=100,
                batch=32,     # global batch: 16 images per GPU on two GPUs
                device="0,1",
            )
  torchrun:
    - label: train.py
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(data="my-dataset.yaml", epochs=100, batch=32)
    - label: Launch
      language: bash
      code: |
        torchrun --nproc_per_node=2 train.py
  syncbn:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreRTDETRr18.pt")
            model.train(
                data="my-dataset.yaml",
                batch=32,
                device="0,1",
                sync_bn=True,
            )
  autobatch:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            # Probed once on GPU 0, scaled to a world-size multiple.
            model.train(data="my-dataset.yaml", batch=-1, device="0,1")
source_hash: 83c1563d68068cd0
---

## Run on two GPUs

Pass a device list. Nothing else changes.

<code-tabs name="train" />

Given more than one device and no torchrun environment, the model's `train()`
saves the weights to a temporary file, resolves autobatch if requested, and
spawns one worker process per GPU with `torch.multiprocessing.spawn`. Each worker
re-imports the model class, rebuilds it from the saved weights, and runs the
ordinary single-device path, because from inside a spawned worker the torchrun
environment variables are set. Rank 0's best checkpoint is loaded back into the
caller's model instance when the run finishes.

`device` accepts `"0,1"`, `[0, 1]`, `0`, `"cuda:0"`, `"cpu"`, `"mps"` and
`"auto"`. Only a list of more than one CUDA index triggers the spawn.

## The `__main__` guard is mandatory

Spawned workers re-import the module they came from. Without a
`if __name__ == "__main__":` guard, that import re-executes the training call and
each worker spawns its own workers. The library detects the case and raises
rather than letting it recurse:

```text
spawn_ddp_train() was called from inside a spawned subprocess. This usually
means your script calls model.train(device=...) at the top level without a
'if __name__ == "__main__":' guard.
```

Everything crossing into a worker is pickled, so `callbacks=` has to be
picklable. A module-level class works; a closure or a lambda does not, and the
error says so and points at the built-in loggers as the alternative.

## batch is the global batch

`batch` is the number of images per optimizer step across all GPUs. Each rank's
dataloader is built at `batch // world_size` with a `DistributedSampler`, so
`batch=32` on two GPUs means 16 images per GPU, not 32.

A batch that does not divide evenly by the world size raises rather than
quietly training at a different size:

```text
batch=6 is the global batch and must be divisible by world_size=4: each rank
trains at batch // world_size, so this value would silently train at a
different global batch than requested. Use batch=4 or batch=8.
```

Gradients are averaged by DDP itself, so the loss is passed through unscaled.
Multiplying it by the world size on top of that would inflate the effective
learning rate by roughly the number of GPUs.

## Autobatch under DDP

`batch=-1` works, and returns a world-size-divisible global batch.

<code-tabs name="autobatch" />

On the spawn path the probe runs in the parent process on the first device
before any worker exists, so every worker receives a concrete integer and no
inter-process coordination is needed. Under torchrun, rank 0 probes and
broadcasts the result as a single long tensor.

The probe measures one GPU's capacity and multiplies by the world size. When
`nbs` is set, the global batch is capped at `nbs` and rounded down to a multiple
of the world size, so adding GPUs reduces the number of accumulation steps rather
than shrinking the per-GPU batch. The mechanics of the probe itself are on
[Hyperparameters](/docs/train/hyperparameters).

## SyncBatchNorm

Under DDP each rank's BatchNorm layers see only its own shard. At
`batch // world_size` that shard can be small enough for the running statistics
to degrade the converged model against a single-GPU run.

`sync_bn=True` converts every BatchNorm to SyncBatchNorm so the statistics are
computed across the global batch. The conversion only happens when distributed is
active, so a single-GPU run is unaffected by the flag either way.

It is already on by default for the BatchNorm-heavy convolutional families:
YOLOX, YOLOv7, YOLOv9 and its variants, YOLO-NAS, PicoDet, RTMDet and FOMO.
Every other family defaults it off. When a model contains BatchNorm, `sync_bn` is
off and the per-rank batch is below 16, the trainer warns.

<code-tabs name="syncbn" />

There is no CLI flag for `sync_bn`. It is a Python argument.

## Launching with torchrun

torchrun works too, and is the right choice when a cluster scheduler already owns
process launch. Write the script for a single device and let torchrun set the
rank environment.

<code-tabs name="torchrun" />

Do not combine the two. With the torchrun environment present, `device="0,1"`
does not spawn; the trainer takes `cuda:LOCAL_RANK` and torchrun owns the
process count.

## Rank behavior

Rank 0 owns every side effect. It resolves the run directory and broadcasts the
resolved name so all ranks agree, writes checkpoints and artifacts, and fires the
user callbacks and loggers. Other ranks train and contribute gradients.

Each rank seeds its dataloader and augmentation RNG differently, derived from the
configured `seed`, so the ranks do not draw identical augmentations.

## Platform and backend

The backend is chosen automatically: NCCL when CUDA and NCCL are both available,
Gloo otherwise. NCCL is not built on Windows, so Windows runs get Gloo without
any configuration. The process group is initialized with a three hour timeout.

## What does not run under DDP

- CUDA graph capture. `cuda_graph=True` logs one line and trains eager. See
  [Training performance](/docs/train/performance).
- The training profiler. `profile=True` is ignored with a warning.

Not every family supports the automatic spawn. Twenty-four do, covering the
detection, classification, semantic and restoration families that train. A family
without it, handed a multi-GPU device, raises an error naming the model API and
the torchrun command rather than quietly training on one GPU.

## Related

- [Hyperparameters](/docs/train/hyperparameters) for `batch`, `nbs` and resume.
- [Experiment loggers](/docs/train/loggers) for the picklability constraint on
  callbacks.
- [Cloud GPUs](/docs/train/cloud-gpus) for renting a multi-GPU box.


