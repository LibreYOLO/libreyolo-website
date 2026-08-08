---
title: libreyolo ui
seo_title: "libreyolo ui command reference"
description: "Launch the local inference web UI: bind address, port behavior, device selection, and how the command exits."
lead: "Starts a local web server that accepts dropped or pasted images, runs a chosen model on them, and shows the results in the browser."
keywords: [libreyolo ui cli, libreyolo web ui, local inference ui, drag and drop inference, libreyolo ui port]
last_verified: "1.5.0"
meta:
  - label: Command
    value: libreyolo ui
    mono: true
  - label: Output
    value: "A server URL on stdout, then the process stays in the foreground"
snippets:
  examples:
    - label: Basic
      language: bash
      code: |
        libreyolo ui
    - label: Fixed port, no browser
      language: bash
      code: |
        libreyolo ui port=9000 no_browser=true
    - label: On the CPU, machine readable
      language: bash
      code: |
        libreyolo ui device=cpu json=true
---

## Synopsis

```bash
libreyolo ui [key=value ...]
```

Arguments are `key=value` pairs, and POSIX form works too, so `port=9000` and
`--port 9000` are the same argument.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `host` | `127.0.0.1` | Host or interface to bind |
| `port` | `8000` | Port to bind. Bumps to the next free one if taken |
| `device` | `auto` | Device: `0`, `cpu`, `mps`, `auto` |
| `no_browser` | `false` | Do not auto-open the browser |
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |
| `verbose` | `false` | Verbose stderr output |

## Examples

<code-tabs name="examples" />

## Notes

The default bind is loopback, so the UI is reachable from this machine only.

If the requested port is in use, the command tries the next one and keeps going
up to twenty ports past the request. Failing all twenty exits with `io_error`
and the suggestion to pass a different port. The URL printed on stdout is the
port that was actually bound, so read it rather than assuming the one you
asked for.

Unless `no_browser=true`, a browser tab opens at that URL shortly after the
bind.

The command then serves in the foreground until Ctrl+C, which shuts the server
down cleanly. There is no detached mode; background it with your shell if you
want the terminal back.

`json=true` prints the URL and device as one object with `schema_version`
before the server starts, which is how a script picks up the bound port.

Related: [`libreyolo label`](/docs/cli/label) for drawing boxes and saving
labels, [`libreyolo monitor`](/docs/cli/monitor) for watching training runs.
Both are local web servers with the same port and browser behavior.
