---
title: Pelatihan pada GPU sewaan
seo_title: Latih LibreYOLO pada GPU cloud sewaan
description: >-
  Run a LibreYOLO training job on a rented or serverless GPU: stage the data,
  install, launch, watch it live, retrieve the weights and stop paying.
lead: >-
  A rented GPU turns a training run into a job with a start, an end, and a bill.
  The work is the same as training locally; what changes is getting the data in,
  watching from outside, getting the weights out, and shutting the machine down.
keywords:
  - cloud gpu training
  - rent a gpu
  - vast.ai training
  - modal serverless gpu
  - beam gpu
  - remote training
  - hugging face dataset staging
  - gpu cost per epoch
last_verified: 1.5.0
snippets:
  install:
    - label: On the box
      language: bash
      code: |
        pip install libreyolo

        # Add only the extras the run needs. rfdetr for RF-DETR training,
        # lora for parameter-efficient fine-tuning, onnx to export afterwards.
        pip install "libreyolo[rfdetr,lora]"
    - label: Check the GPU before anything else
      language: python
      code: |
        import torch

        print(torch.__version__, torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))

        # A wheel built for another architecture reports True and then fails
        # on the first real kernel, so run one.
        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  stage:
    - label: 'Pack and upload once, from your machine'
      language: bash
      code: >
        tar cf my-dataset.tar my-dataset/

        huggingface-cli upload my-org/my-dataset my-dataset.tar --repo-type
        dataset
    - label: Stage on the box
      language: python
      code: |
        import tarfile

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "my-org/my-dataset", "my-dataset.tar", repo_type="dataset"
        )
        with tarfile.open(path) as archive:
            archive.extractall("/root/data")
  launch:
    - label: 'Detached, so the job survives a disconnect'
      language: bash
      code: |
        nohup libreyolo train \
          model=LibreYOLO9s.pt \
          data=/root/data/my-dataset/data.yaml \
          epochs=100 batch=-1 imgsz=640 \
          project=/root/runs name=run1 \
          > /root/train.log 2>&1 &
    - label: 'Multi-GPU, from a Python file'
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="/root/data/my-dataset/data.yaml",
                epochs=100,
                batch=64,          # global batch across all GPUs
                device="0,1,2,3",
                project="/root/runs",
                name="run1",
            )
  watch:
    - label: One cheap read
      language: bash
      code: |
        cat /root/runs/run1/status.json
    - label: From a script
      language: python
      code: |
        import json

        with open("/root/runs/run1/status.json") as handle:
            status = json.load(handle)

        print(status["state"], status["current_epoch"], status["eta_seconds"])
        print(status.get("metrics"))
    - label: 'In a browser, over an SSH tunnel'
      language: bash
      code: |
        # On the box (binds 127.0.0.1:8420 by default):
        libreyolo monitor /root/runs/run1 --no-browser

        # From your machine, then open http://localhost:8420 locally:
        #   ssh -L 8420:localhost:8420 <user>@<host>
  push:
    - label: Push the weights somewhere permanent
      language: bash
      code: |
        huggingface-cli upload my-org/my-run \
          /root/runs/run1/weights/best.pt best.pt
source_hash: 75d314de06aca3b6
---

## Sebelum menyewa apa pun

Two decisions cost more later than they do now.

Get the dataset onto a CDN first. Packing it as a single tar in a Hugging Face
dataset repository works the same on every provider, serves fast to all of them,
and needs nothing but an `HF_TOKEN` in the job environment when the repo is
private. Copying a dataset up from a home connection, or pulling it from a slow
origin on the box, is billed GPU time spent waiting.

<code-tabs name="stage" />

Then size the disk. Providers that bill storage bill on allocated capacity, not
used capacity, and a disk cannot be shrunk after creation. Add up the staged
data, the checkpoints, and roughly 30 percent of headroom, and stop there.

## Instalasi pada mesin

<code-tabs name="install" />

Install PyTorch first if the image does not already carry a CUDA build matching
the card, then LibreYOLO, so pip does not resolve its own CPU-only torch. The
second snippet is not optional ceremony: a wheel built for the wrong GPU
architecture reports `torch.cuda.is_available() == True` and then fails on the
first real operation with `CUDA error: no kernel image is available for execution
on the device`. One matrix multiply catches it before an hour of setup does not.

Point `HF_HOME` at persistent storage if the provider offers a volume, so
checkpoint and dataset downloads survive between runs.

## Jalankan

Run the job detached. An interactive session that dies with your network
connection takes the training with it.

<code-tabs name="launch" />

`batch=-1` is worth using here specifically, because you are usually on a card
you have not trained on before. It probes the model in training mode with a real
backward pass and picks the largest power of two that fits, which is faster than
discovering the ceiling with an out-of-memory error twenty minutes in. See
[Hyperparameters](/docs/train/hyperparameters).

On a multi-GPU box, `device="0,1,2,3"` spawns one worker per GPU by itself, and
`batch` stays the global batch across all of them. The `__main__` guard is
mandatory, because each worker re-imports the script. That, and the rest of the
distributed behavior, is on [Multi-GPU training](/docs/train/multi-gpu).

## Pantau dari luar

Every run writes `status.json` into its run directory, rewritten atomically each
epoch. It is the cheap read: a few hundred bytes carrying the state, the current
epoch, the ETA and the latest metrics, without parsing a log.

<code-tabs name="watch" />

`metrics.jsonl` alongside it has the full per-epoch history, and `train.log` has
the console output. `libreyolo monitor` serves a browser dashboard over all
three using only the standard library, so it needs nothing installed on the box
beyond LibreYOLO itself. Reach it over an SSH port forward.

None of these touch the training process, so they attach to a live run, reopen a
finished one, or inspect a crashed one.

## Ambil bobot sebelum berhenti membayar

The box is disposable. Push checkpoints at milestones, not only at the end,
because a crash, a preemption or running out of credit otherwise loses the whole
run.

<code-tabs name="push" />

`weights/best.pt` and `weights/last.pt` are written every epoch and on every
improvement. `save_period=N` adds `weights/epoch_<N>.pt` snapshots on top, which
is what makes a mid-run push cheap. `summary.json` and `results.csv`, where the
family writes them, are small and worth taking too.

A callback on `on_train_epoch_end` is the clean way to automate the push. See
[Experiment loggers](/docs/train/loggers), where the hosted backends also give
you the metrics without touching the box at all.

## Hentikan pembayaran

This is the part that costs real money when it goes wrong, and the rule differs
by provider model.

On a marketplace where you rent a raw machine, billing runs on wall clock until
the instance is destroyed. An idle GPU bills exactly like a busy one, so killing
the training process saves nothing on its own. A stopped instance still bills its
disk.

On a serverless platform where the job is a decorated function, the container
scales to zero when the function returns, so a forgotten box is much less likely.
A hung job with no timeout still bills, so always set one.

Stopping instead of destroying is a real lever, and a real trap. Measured on a
rented 8x RTX 4090 with a 250 GB disk on 2026-07-31: running billed $3.4828 per
hour, stopped billed $0.0694 per hour for the disk alone, and destroyed billed
nothing. That is a 98 percent saving while keeping the environment, the staged
data and the checkpoints in place.

The stopped rate is arithmetic you can do before renting:

```text
stopped $/hr = allocated_GB * storage_cost_per_GB_per_month / 730
             = 250 * 0.20 / 730 = $0.0694/hr
```

Compare it against what a rebuild costs: renting again, pulling the image,
installing, and re-staging the data. On that same box a rebuild was about 15
minutes of setup plus 43 GB of inbound transfer, roughly $1.00 all in. Against
$0.0694 per hour, coming back within about 14 hours favors stopping and a longer
gap favors destroying and rebuilding from the staged copy.

One risk makes stopping unsafe for scarce hardware: stopping releases the GPUs.
Nothing reserves them, so restarting only succeeds if the host still has them
free. Your disk is safe; your GPUs are not.

## Serverless sebagai fungsi

If you would rather not manage a machine, both Modal and Beam run a decorated
Python function on a GPU and scale to zero when it returns. LibreYOLO's own
nightly test suite runs on Modal, and `tools/ci/modal_nightly.py` in the library
repository is the working in-repo example to copy from.

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")   # OpenCV system libraries
    .pip_install("libreyolo[rfdetr]")
)
app = modal.App("libreyolo-train")
cache = modal.Volume.from_name("libreyolo-cache", create_if_missing=True)


@app.function(gpu="A100", timeout=6 * 60 * 60, volumes={"/cache": cache})
def train():
    import os

    os.environ["HF_HOME"] = "/cache/hf"          # cache weights across runs

    from libreyolo import LibreYOLO

    model = LibreYOLO("LibreYOLO9s.pt")
    model.train(data="coco8.yaml", epochs=100, project="/cache/runs")
    cache.commit()                                # persist the volume


@app.local_entrypoint()
def main():
    train.remote()
```

Run it with `modal run modal_train.py`. The container filesystem is ephemeral, so
anything worth keeping goes in the volume or gets pushed out. Set `timeout=`
explicitly; that is the only thing standing between a hung run and an open-ended
bill.

Beam takes the same shape with a `@function` decorator, a `Volume`, and
`train.remote()` called from `__main__`.

## Tentukan ukuran berdasarkan biaya per job

$/hr is the wrong number to optimize. A small model half-idles a large card, so a
cheaper and slower GPU is often cheaper per epoch. Run the profiler for a few
steps on the rented card before committing to a long run: if the verdict is
`dataloader` or `host / launch`, a faster GPU buys nothing and more workers or a
larger batch buys a lot. See
[Training performance](/docs/train/performance).

## Terkait

- [Datasets](/docs/train/datasets) for the layout the staged archive should have,
  and the doctor command that catches problems before a GPU is billing.
- [Multi-GPU training](/docs/train/multi-gpu) for multi-card boxes.
