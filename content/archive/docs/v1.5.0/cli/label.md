---
title: libreyolo label
seo_title: "libreyolo label command reference"
description: "Launch the local bounding-box annotation tool: arguments with defaults, the AI assist switch, and what binding to a network interface exposes."
lead: "Starts a local web tool for drawing and editing bounding boxes. It writes LibreYOLO-native label files, so a dataset annotated here trains with no conversion step."
keywords: [libreyolo label cli, bounding box annotation tool, yolo labeling tool, auto label cli, libreyolo label share]
last_verified: "1.5.0"
meta:
  - label: Command
    value: libreyolo label
    mono: true
  - label: Output
    value: "A server URL on stdout; labels written as labels/*.txt beside the images"
snippets:
  examples:
    - label: Basic
      language: bash
      code: |
        # Opens the project home; pick or create a dataset in the browser.
        libreyolo label
    - label: Manual only, fixed port
      language: bash
      code: |
        libreyolo label no_assist=true port=9200 no_browser=true
    - label: Let teammates join
      language: bash
      code: |
        libreyolo label share=true
---

## Synopsis

```bash
libreyolo label [data=<dataset.yaml|folder>] [key=value ...]
```

Arguments are `key=value` pairs, and POSIX form works too, so `port=9200` and
`--port 9200` are the same argument.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `data` | | Dataset YAML or folder to open directly. Starts on the project home when unset |
| `host` | `127.0.0.1` | Host or interface to bind |
| `port` | `8000` | Port to bind. Bumps to the next free one if taken |
| `device` | `auto` | Device for AI auto-label: `0`, `cpu`, `mps`, `auto` |
| `no_assist` | `false` | Disable AI auto-label, leaving a manual labeler |
| `no_browser` | `false` | Do not auto-open the browser |
| `share` | `false` | Bind on `0.0.0.0` so teammates on your network can join |
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |
| `verbose` | `false` | Verbose stderr output |

## Examples

<code-tabs name="examples" />

## Notes

### What it writes

Boxes are saved as LibreYOLO-native `labels/*.txt` files, which is the format
`libreyolo train` reads, so nothing has to be converted afterwards. This version
handles bounding boxes only. Edits save as you move between images.

### Opening a dataset

With no `data`, the tool starts on the project home and a dataset is chosen or
created from the browser. Passing `data=path/to/data.yaml` opens that dataset
straight away, and the startup line reports the image count, the class count,
and whether the dataset is writable. A read-only dataset still opens and says
why it cannot be written to.

### Sharing, and what `host` does

`share=true` binds the wildcard address, which lets other machines on your
network reach the tool while administrative actions, switching or deleting
projects and starting compute, stay on this machine.

Setting `host` to a specific interface does something different and less safe:
the host becomes indistinguishable from a network client, so every client gets
administrative rights. The command prints a warning on stderr when you do it.
Prefer `share=true`.

### Ports and shutdown

An occupied port moves to the next one, up to twenty past the request. Failing
all twenty exits with `io_error`. The URL printed on stdout is the port that was
actually bound. With `share=true`, the result also carries `lan_url`, the
address teammates should open.

The command serves in the foreground until Ctrl+C.

Related: [`libreyolo doctor`](/docs/cli/doctor) to check the labeled dataset
before training, and [`libreyolo train`](/docs/cli/train) to train on it.
