---
title: libreyolo monitor
seo_title: "libreyolo monitor command reference"
description: "Serve a live dashboard for training runs: arguments with defaults, what the server reads from disk, and how one server covers many runs."
lead: "Serves a web dashboard for training runs, reading the artifacts a run writes to disk. It never attaches to the training process, so live, finished and crashed runs all display."
keywords: [libreyolo monitor cli, training dashboard, watch training run, libreyolo monitor port, training metrics viewer]
last_verified: "1.5.0"
meta:
  - label: Command
    value: libreyolo monitor
    mono: true
  - label: Output
    value: "A server URL on stdout, then the process stays in the foreground"
snippets:
  examples:
    - label: Basic
      language: bash
      code: |
        # Watches runs/ and lists every run under it.
        libreyolo monitor
    - label: A different runs root
      language: bash
      code: |
        libreyolo monitor experiments/
    - label: One run, fixed port, no browser
      language: bash
      code: |
        libreyolo monitor runs/train/exp port=9100 no_browser=true
---

## Synopsis

```bash
libreyolo monitor [<run-dir|runs-root>] [key=value ...]
```

The directory is positional. Everything else is a `key=value` pair, and POSIX
form works too, so `port=9100` and `--port 9100` are the same argument.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `run_dir` | `runs` | Positional. A runs root to watch, or a single run directory to open directly. Either way every run under the root is listed |
| `host` | `127.0.0.1` | Host or interface to bind |
| `port` | `8420` | Port to bind. Bumps to the next free one if taken |
| `no_browser` | `false` | Do not auto-open the browser |
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |
| `verbose` | `false` | Verbose stderr output |

## Examples

<code-tabs name="examples" />

## Notes

### One server, many runs

The server watches a runs root rather than a single run, and addresses each run
by URL, so several runs on one machine share one port. Open the root URL for
the index, or one tab per run; the `?run=` parameter in each URL identifies
which.

Pointing the command at a single run directory roots the server at that
directory's parent, so sibling runs still appear in the index, and deep-links
straight to the one named.

### What it reads

The dashboard is built from the files `libreyolo train` writes: `status.json`,
`metrics.jsonl`, `train.log` and the run's images. Nothing is read from the
training process itself, so a run that has finished, or died, displays exactly
as a live one does.

### Preconditions and ports

At least one run must already exist. With no argument and no `runs/` directory,
the command exits with `source_not_found`; the same happens when the directory
given holds no runs.

An occupied port moves to the next one, up to twenty past the request. Failing
all twenty exits with `io_error`. The URL printed on stdout is the port that
was actually bound.

The command serves in the foreground until Ctrl+C. `json=true` prints the URL,
the root being watched and the number of runs found, as one object with
`schema_version`.

Related: [`libreyolo train`](/docs/cli/train), whose `project` and `name`
arguments decide where these run directories go.
