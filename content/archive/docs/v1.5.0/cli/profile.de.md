---
title: libreyolo profile
seo_title: Referenz des Befehls libreyolo profile
description: >-
  Trainings- und Inferenzgeschwindigkeit messen und das Ergebnis lesen: jedes
  Unterkommando von profile, seine Argumente und Standardwerte, und was jede
  Sicht berichtet.
lead: >-
  Eine Befehlsgruppe, die misst, wohin die Zeit in einem Trainingsschritt oder
  einem Inferenzaufruf geht, ein eigenständiges Profil schreibt und dieses
  Profil durch mehrere Sichten wieder ausliest.
keywords:
  - libreyolo profile cli
  - yolo training profiling
  - inferenz latenz messen
  - gpu kernel profiling pytorch
  - libreyolo performance vergleichen
last_verified: 1.5.0
meta:
  - label: Befehl
    value: libreyolo profile
    mono: true
  - label: Ausgabe
    value: profile.json und profile_trace.json unter runs/profile
    mono: true
snippets:
  examples:
    - label: Inferenz messen
      language: bash
      code: |
        # Ohne source-Argument wird das mitgelieferte Beispielbild genutzt.
        libreyolo profile infer --device cpu --warmup 5 --runs 20
    - label: Das Urteil lesen
      language: bash
      code: |
        libreyolo profile summary runs/profile/infer/profile.json
    - label: Zwei Messungen vergleichen
      language: bash
      code: >
        libreyolo profile infer --device cpu --warmup 5 --runs 20 --project
        runs/profile/a

        libreyolo profile infer --device cpu --warmup 5 --runs 20 --batch 4
        --project runs/profile/b


        libreyolo profile compare runs/profile/a/infer/profile.json \
          runs/profile/b/infer/profile.json
source_hash: b967e869fd9ba418
---

## Synopsis

```bash
libreyolo profile <subcommand> [<positional>] [--flag value ...]
```

Diese Gruppe nimmt keine `key=value`-Argumente. Ihre Unterkommandos verwenden
positionale Argumente und POSIX-Flags, es heißt also
`--weights LibreYOLO9t.pt`, nicht `weights=LibreYOLO9t.pt`. `libreyolo profile`
ohne Unterkommando gibt die Liste aus.

Zwei Unterkommandos messen und schreiben ein Profil; die übrigen lesen eines.
`run` und `infer` geben beide dasselbe eigenständige `profile.json` aus,
deshalb arbeitet jedes lesende Unterkommando mit beiden.

## profile run

Führt ein kurzes, profiliertes Training aus und schreibt ein Profil.

```bash
libreyolo profile run <data> [--flag value ...]
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `data` | | Positional. Datensatz-YAML oder Name, z. B. `coco128`. Erforderlich |
| `--weights` | `LibreYOLO9t.pt` | Gewichtsdatei oder Name des Modells |
| `--size` | `t` | Größenvariante des Modells |
| `--batch` | `16` | Micro-Batch. `-1` passt automatisch auf etwa 70 % des VRAM |
| `--imgsz` | `640` | Bildgröße beim Training |
| `--workers` | `8` | Dataloader-Worker |
| `--amp` | `true` | Den AMP-Pfad der Familie nutzen. `--no-amp` schaltet ihn ab |
| `--steps` | `20` | Profilierte, also gemessene Schritte |
| `--warmup` | `5` | Warmup-Schritte vor der Messung |
| `--repeat` | `1` | N-mal wiederholen, für Mittelwert und Standardabweichung |
| `--device` | `0` | Gerät |
| `--project` | `runs/profile` | Wurzelverzeichnis der Ausgabe |
| `--json` | `false` | JSON-Ausgabe auf stdout |

Das gemessene Fenster umfasst `--warmup` plus `--steps` Iterationen. Ein
Datensatz, der zu klein ist, um es zu füllen, liefert kein Profil, und der
Befehl endet mit Code `3` und nennt die drei Auswege: ein größerer Datensatz,
weniger Schritte oder ein kleinerer Batch.

`--repeat` über 1 schreibt ein aggregiertes `runs/profile/profile_repeat.json`,
dessen skalare Metriken über die Durchläufe gemittelt sind, während die
Kernel-Listen aus dem letzten Durchlauf stammen. Es ist außerdem die
Voraussetzung für ein Signifikanzurteil in `compare`: Ein einzelner Lauf kann
keines liefern.

## profile infer

Profiliert den Inferenzpfad und schreibt ein Profil.

```bash
libreyolo profile infer [<source>] [--flag value ...]
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `source` | | Positional. Bild oder Verzeichnis. Ohne Angabe das mitgelieferte Beispielbild |
| `--weights` | `LibreYOLO9t.pt` | Gewichtsdatei oder Name des Modells |
| `--size` | `t` | Größenvariante des Modells |
| `--batch` | `1` | Bilder pro Forward-Pass |
| `--imgsz` | `640` | Größe des Eingabebilds |
| `--half` | `false` | Autocast im Forward, nur CUDA. `--no-half` schaltet es ab |
| `--amp-dtype` | `float16` | CUDA-Autocast-dtype: `float16` oder `bfloat16` |
| `--warmup` | `20` | Warmup-Iterationen vor der Messung |
| `--runs` | `100` | Gemessene Iterationen |
| `--repeat` | `1` | N-mal wiederholen, für Mittelwert und Standardabweichung |
| `--conf` | `0.25` | Confidence-Schwellenwert, der ändert, wie viel Arbeit NMS leistet |
| `--iou` | `0.45` | IoU-Schwellenwert für NMS |
| `--max-det` | `300` | Maximale Erkennungen pro Bild, was ändert, wie viel Arbeit NMS leistet |
| `--device` | `0` | Gerät |
| `--trace` | `true` | Einen Chrome-Trace für die Kernel- und Op-Detailanalyse ausgeben. `--no-trace` lässt ihn weg |
| `--project` | `runs/profile` | Wurzelverzeichnis der Ausgabe |
| `--json` | `false` | JSON-Ausgabe auf stdout |

Berichtet die Latenz bei p50, p90 und p99, den Durchsatz in Bildern pro
Sekunde und die Aufteilung der Stufen auf Preprocess, Forward und Postprocess.
Die drei Schwellenwert-Argumente stehen hier, weil sie den Postprocess-Wert
verschieben.

## profile summary

```bash
libreyolo profile summary <trace> [--json]
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `trace` | | Positional. Pfad zu einer `profile.json` oder `profile_trace.json`. Erforderlich |
| `--json` | `false` | JSON-Ausgabe auf stdout |

Die Sicht von oben: Schrittzeit, Durchsatz, GPU-Auslastung, Tensor-Core-Anteil,
VRAM-Spitze, Host-Overhead, Kernel-Starts pro Schritt, das Bottleneck-Urteil
mit seiner Begründung, die Kernel-Verteilung nach Kategorie und die Top-Kernel
pro Schritt. Bei einem Inferenzprofil gibt es zusätzlich die Latenz-Perzentile
und die Aufteilung der Stufen aus.

Ein Profil, das unter VRAM-Thrashing aufgenommen wurde, wird markiert, weil den
dort gemessenen Werten für Auslastung und Durchsatz nicht zu trauen ist.

## profile get

```bash
libreyolo profile get <trace> [<field>] [--json]
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `trace` | | Positional. Pfad zu einem Profil. Erforderlich |
| `field` | | Positional. Name der Metrik. Weglassen, um die verfügbaren Metriken aufzulisten |
| `--json` | `false` | JSON-Ausgabe auf stdout |

Gibt eine Metrik aus und sonst nichts, für Schleifen in Skripten. Ein
unbekanntes Feld endet mit Code `2` und verweist auf die auflistende Form.

## profile phases

```bash
libreyolo profile phases <trace> [--json]
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `trace` | | Positional. Pfad zu einem Profil. Erforderlich |
| `--json` | `false` | JSON-Ausgabe auf stdout |

GPU-Millisekunden, Wall-Millisekunden, Kernel-Anzahl und Op-Anzahl pro Phase:
forward, backward, dataload, to_device, optimizer.

## profile kernels

```bash
libreyolo profile kernels <trace> [--flag value ...]
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `trace` | | Positional. Pfad zu einem Profil. Erforderlich |
| `--top` | `20` | Die Top N nach GPU-Zeit zeigen |
| `--category` | | Nach einem Teilstring der Kategorie filtern: `gemm`, `layout`, `norm`, `elementwise` |
| `--grep` | | Nach einem regulären Ausdruck auf dem Kernel-Namen filtern |
| `--tensorcore` | `false` | Nur Tensor-Core-Kernel |
| `--sort` | `time` | `time`, `count` oder `name` |
| `--phase` | | Auf eine Phase beschränken: `forward`, `backward`, `dataload`, `to_device`, `optimizer` |
| `--json` | `false` | JSON-Ausgabe auf stdout |

Der Boden der Analyse: einzelne GPU-Kernel mit ihrem Anteil an der GPU-Zeit,
Millisekunden pro Schritt, Aufrufen pro Schritt und Kategorie. Ein unbekanntes
`--phase` endet mit Code `2` und listet die Phasen auf, die das Profil hat.

## profile ops

```bash
libreyolo profile ops <trace> [--flag value ...]
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `trace` | | Positional. Pfad zu einem Profil. Erforderlich |
| `--top` | `20` | Die Top N nach CPU-Zeit zeigen |
| `--phase` | | Auf eine Phase beschränken |
| `--json` | `false` | JSON-Ausgabe auf stdout |

Die Sicht des Frameworks statt der Sicht des Geräts: `aten`- und Autograd-Ops
nach CPU-Zeit sortiert, denn dort zeigen sich die Kosten der Host-Starts.

## profile compare

```bash
libreyolo profile compare <before> <after> [--json]
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `before` | | Positional. Basisprofil. Erforderlich |
| `after` | | Positional. Neues Profil. Erforderlich |
| `--json` | `false` | JSON-Ausgabe auf stdout |

Vergleicht Durchsatz, Millisekunden pro Bild, GPU-Auslastung, Host-Overhead,
Kernel-Starts pro Schritt und das Bottleneck-Urteil.

Die Signifikanzaussage braucht beide Seiten mit einem `--repeat` von mindestens
2 gemessen. Dann gilt ein Unterschied als signifikant, wenn er das Doppelte des
kombinierten Standardfehlers übersteigt, und die Ausgabe nennt den Vergleich,
den sie angestellt hat. Ohne das steht in der Zeile, dass ein einzelner Lauf
die Aussage nicht tragen kann.

## profile what-if

```bash
libreyolo profile what-if <trace> [--flag value ...]
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `trace` | | Positional. Pfad zu einem Profil. Erforderlich |
| `--remove-category` | | Das Entfernen einer Kernel-Kategorie hochrechnen: `gemm`, `layout`, `norm`, `elementwise` |
| `--remove-launches` | | Das Entfernen von N Kernel-Starts pro Schritt hochrechnen, zum Beispiel einen Gewinn durch Op-Fusion |
| `--json` | `false` | JSON-Ausgabe auf stdout |

Schätzt ab, was eine Änderung bringt, bevor sie geschrieben ist. Eine der
beiden Optionen ist erforderlich; ohne beide endet der Befehl mit Code `2`.

Die Hochrechnung folgt dem Urteil des Profils selbst. Unter 80 % GPU-Auslastung
modelliert sie die Ersparnis als weniger Starts mal den gemessenen Host-Kosten
pro Start, darüber als weniger GPU-Arbeit. Das Ergebnis führt ein Feld mit
Vorbehalt, weil die Kosten pro Start eine Näherung sind und der einzige Beweis
eine zweite Messung ist.

## Beispiele

<code-tabs name="examples" />

## Hinweise

Der Profiler misst und berichtet. Er ändert nichts: das Urteil lesen, die
Konfiguration oder den Code anpassen, erneut laufen lassen und vergleichen,
dafür ist die Schleife gebaut.

`--device` steht standardmäßig auf `0`, also CUDA-Gerät 0. Mit `--device cpu`
wird auf der CPU gemessen, und es entsteht ein Profil, das die lesenden
Unterkommandos weiterhin annehmen, nur ohne die Details zu den GPU-Kerneln.

Jedes Unterkommando unterstützt `--json`, und die lesenden geben ausschließlich
auf stdout aus, was die Gruppe aus einem Skript heraus nutzbar macht.

Die Exit-Codes sind hier die der Gruppe selbst: `2` für eine Datei, die nicht
existiert, oder ein Argument, das sich nicht auflösen lässt, `3`, wenn `run`
kein Profil erzeugt hat, und `1`, wenn ein Trace nicht analysiert werden kann.

Verwandt: [`libreyolo train`](/docs/cli/train), dessen Argumente man mit einem
Trainingsprofil üblicherweise abstimmt.
