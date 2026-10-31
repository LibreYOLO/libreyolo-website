---
title: libreyolo monitor
seo_title: Referenz zum Befehl libreyolo monitor
description: >-
  Ein Live-Dashboard für Trainings-Runs bereitstellen: Argumente mit
  Standardwerten, was der Server von der Festplatte liest und wie ein Server
  viele Runs abdeckt.
lead: >-
  Stellt ein Web-Dashboard für Trainings-Runs bereit und liest dafür die
  Artefakte, die ein Run auf die Festplatte schreibt. Der Befehl hängt sich nie
  an den Trainingsprozess, laufende, fertige und abgestürzte Runs werden deshalb
  alle angezeigt.
keywords:
  - libreyolo monitor cli
  - trainings dashboard
  - training live verfolgen
  - libreyolo monitor port
  - trainingsmetriken anzeigen
last_verified: 1.5.0
meta:
  - label: Befehl
    value: libreyolo monitor
    mono: true
  - label: Ausgabe
    value: 'Eine Server-URL auf stdout, danach bleibt der Prozess im Vordergrund'
snippets:
  examples:
    - label: Basis
      language: bash
      code: |
        # Beobachtet runs/ und listet jeden Run darunter auf.
        libreyolo monitor
    - label: Ein anderes Runs-Wurzelverzeichnis
      language: bash
      code: |
        libreyolo monitor experiments/
    - label: 'Ein Run, fester Port, kein Browser'
      language: bash
      code: |
        libreyolo monitor runs/train/exp port=9100 no_browser=true
source_hash: 4aa178141d451728
---

## Synopsis

```bash
libreyolo monitor [<run-dir|runs-root>] [key=value ...]
```

Das Verzeichnis ist positionell. Alles andere ist ein `key=value`-Paar, und die
POSIX-Form funktioniert ebenfalls, `port=9100` und `--port 9100` sind also
dasselbe Argument.

## Argumente

| Argument | Standard | Bedeutung |
|---|---|---|
| `run_dir` | `runs` | Positionell. Ein Runs-Wurzelverzeichnis, das beobachtet wird, oder ein einzelnes Run-Verzeichnis, das direkt geöffnet wird. In beiden Fällen werden alle Runs unterhalb der Wurzel aufgelistet |
| `host` | `127.0.0.1` | Host oder Interface, an das gebunden wird |
| `port` | `8420` | Port, an den gebunden wird. Rückt auf den nächsten freien, wenn er belegt ist |
| `no_browser` | `false` | Den Browser nicht automatisch öffnen |
| `json` | `false` | JSON-Ausgabe auf stdout |
| `quiet` | `false` | stderr unterdrücken |
| `verbose` | `false` | Ausführliche stderr-Ausgabe |

## Beispiele

<code-tabs name="examples" />

## Hinweise

### Ein Server, viele Runs

Der Server beobachtet ein Runs-Wurzelverzeichnis statt eines einzelnen Runs und
adressiert jeden Run über die URL, sodass sich mehrere Runs auf einer Maschine
einen Port teilen. Öffne die Wurzel-URL für den Index oder einen Tab pro Run;
der Parameter `?run=` in jeder URL gibt an, welcher gemeint ist.

Zeigt der Befehl auf ein einzelnes Run-Verzeichnis, wurzelt der Server im
übergeordneten Verzeichnis, sodass benachbarte Runs weiterhin im Index
auftauchen, und verlinkt direkt auf den genannten Run.

### Was gelesen wird

Das Dashboard wird aus den Dateien gebaut, die `libreyolo train` schreibt:
`status.json`, `metrics.jsonl`, `train.log` und die Bilder des Runs. Aus dem
Trainingsprozess selbst wird nichts gelesen, ein Run, der fertig ist oder
abgestürzt ist, wird also genauso angezeigt wie ein laufender.

### Voraussetzungen und Ports

Mindestens ein Run muss bereits existieren. Ohne Argument und ohne Verzeichnis
`runs/` bricht der Befehl mit `source_not_found` ab; dasselbe passiert, wenn das
angegebene Verzeichnis keine Runs enthält.

Ein belegter Port rückt auf den nächsten weiter, bis zu zwanzig über dem
angeforderten. Scheitern alle zwanzig, endet der Befehl mit `io_error`. Die auf
stdout ausgegebene URL nennt den Port, der tatsächlich gebunden wurde.

Der Befehl läuft im Vordergrund, bis Ctrl+C kommt. `json=true` gibt die URL, die
beobachtete Wurzel und die Anzahl der gefundenen Runs als ein Objekt mit
`schema_version` aus.

Verwandt: [`libreyolo train`](/docs/cli/train), dessen Argumente `project` und
`name` bestimmen, wo diese Run-Verzeichnisse landen.
