---
title: libreyolo Hilfsbefehle
seo_title: 'libreyolo CLI: Referenz der Hilfsbefehle'
description: >-
  Die kleinen LibreYOLO-Befehle: version, checks, models, formats, cfg, info,
  metadata, enroll und compare, jeweils mit ihren Argumenten und Standardwerten.
lead: >-
  Neun Befehle, die berichten oder inspizieren, statt zu rechnen. Sie geben
  Fakten zur Umgebung aus, das Modell- und Formatinventar, aufgelöste
  Standardwerte und Details zu Checkpoints, und sie bauen und befragen eine
  Gesichts-Galerie.
keywords:
  - libreyolo version
  - libreyolo checks
  - libreyolo cli befehle
  - libreyolo modelle auflisten
  - libreyolo exportformate anzeigen
  - libreyolo standardwerte anzeigen
  - libreyolo checkpoint metadaten prüfen
  - libreyolo gesichter enroll
  - gesichter vergleichen cli
last_verified: 1.5.0
meta:
  - label: Befehle
    value: 'version, checks, models, formats, cfg, info, metadata, enroll, compare'
    mono: true
  - label: Ausgabe
    value: 'stdout, als Text oder mit json=true als ein Objekt mit schema_version'
snippets:
  examples:
    - label: Umgebung
      language: bash
      code: |
        libreyolo version
        libreyolo checks
    - label: Was verfügbar ist
      language: bash
      code: |
        libreyolo models
        libreyolo formats family=yolo9 task=detect
    - label: Checkpoint inspizieren
      language: bash
      code: |
        libreyolo info model=LibreYOLO9s.pt
        libreyolo metadata path=weights/LibreYOLO9s.pt
source_hash: 7b5b53c46df00c06
---

## Synopsis

```bash
libreyolo <command> [key=value ...]
```

Argumente sind `key=value`-Paare, und die POSIX-Form funktioniert ebenfalls, also
sind `model=x` und `--model x` dasselbe Argument. Jeder Befehl hier schreibt
seine Ergebnisse nach stdout und akzeptiert `json=true` und `quiet=true`.

Der Root-Befehl hat ein eigenes Flag, `libreyolo --version`, das den
Versionsstring ausgibt und sich beendet. Das ist eine kleinere Ausgabe als beim
Befehl `version` weiter unten.

## version

Gibt die LibreYOLO-Version aus, dazu die Versionen von Python, torch und CUDA,
gegen die sie läuft.

```bash
libreyolo version
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `json` | `false` | JSON-Ausgabe nach stdout |
| `quiet` | `false` | stderr unterdrücken |

## checks

Gibt die Umgebung ausführlicher aus: Python, torch, CUDA, cuDNN, jede erkannte
GPU mit Namen und Speicher sowie die installierte Version jedes optionalen
Pakets, das die Export-Pfade nutzen.

```bash
libreyolo checks
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `json` | `false` | JSON-Ausgabe nach stdout |
| `quiet` | `false` | stderr unterdrücken |

Die Paketliste umfasst `onnx`, `onnxruntime`, `tensorrt`, `openvino`,
`paddlepaddle`, `x2paddle`, `mnn`, `ncnn`, `onnx2tf`, `ai-edge-litert`,
`transformers` und `scipy`. Ein Paket, das nicht installiert ist, wird als
solches gemeldet, statt weggelassen zu werden, sodass sich ein fehlgeschlagener
Export mit diesem einen Befehl auf eine fehlende Abhängigkeit zurückführen
lässt.

## models

Listet jede Modellfamilie mit ihren Aufgaben, ihren Größen, den CLI-Namen, die
auf ihre Checkpoints auflösen, und der Eingabeauflösung jeder Größe.

```bash
libreyolo models
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `json` | `false` | JSON-Ausgabe nach stdout |
| `quiet` | `false` | stderr unterdrücken |

Eine Familie, deren optionale Abhängigkeit nicht installiert ist, wird als nicht
verfügbar aufgeführt, zusammen mit der `pip install`-Zeile, die sie verfügbar
machen würde. Die CLI-Namen sind das, was `model=` als Kurzform akzeptiert:
`yolox-s` löst auf `LibreYOLOXs.pt` auf, und Aufgaben außerhalb der Erkennung
tragen ihr Aufgaben-Suffix.

## formats

Listet die Exportformate, die die installierte Umgebung erzeugen kann, mit der
Dateiendung jedes Formats und der Angabe, ob es FP16 und INT8 unterstützt.

```bash
libreyolo formats [family=<family>] [task=<task>]
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `family` | | Zeigt die Stufen für eine Modellfamilie. `model=` wird als dieselbe Option akzeptiert |
| `task` | | Kanonische Modellaufgabe. Ohne Angabe die Standardaufgabe der Familie |
| `json` | `false` | JSON-Ausgabe nach stdout |
| `quiet` | `false` | stderr unterdrücken |

Ohne `family` ist die Ausgabe allein das Formatinventar. Mit `family` bekommt
jedes Format zusätzlich die Support-Stufe für diese Familie und Aufgabe, den
Grund für die Stufe und jede daran hängende Einschränkung. Eine unbekannte
Familie oder eine Aufgabe, die die Familie nicht unterstützt, ist ein
Nutzungsfehler.

Format-Aliase stehen neben ihrem kanonischen Namen: `engine` für `tensorrt`,
`litert` für `tflite`.

## cfg

Gibt die aufgelöste Standardkonfiguration aus: die Standardwerte fürs Training,
die für die Validierung, die für die Vorhersage und die Overrides je Familie.

```bash
libreyolo cfg
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `json` | `false` | JSON-Ausgabe nach stdout |
| `quiet` | `false` | stderr unterdrücken |

Die Werte werden aus den Konfigurations-Dataclasses gelesen, nicht aus einer
Kopie, damit ist das die maßgebliche Quelle dafür, was ein Trainingslauf nutzt,
wenn du kein Argument übergibst. `family_overrides` ist der Abschnitt, der
beantwortet, warum eine Familie mit Einstellungen trainiert hat, die du nicht
angefordert hast. Wie diese Overrides angewendet werden, steht unter
[`libreyolo train`](/docs/cli/train).

## info

Lädt ein Modell auf der CPU und meldet dessen Familie, Größe, Parameterzahl,
Klassen und die Export-Stufe für jedes Format.

```bash
libreyolo info model=<name|path>
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `model` | | Modellname oder Pfad zu den Gewichten. Erforderlich |
| `detailed` | `false` | Details je Parameter einbeziehen |
| `json` | `false` | JSON-Ausgabe nach stdout |
| `quiet` | `false` | stderr unterdrücken |

## metadata

Liest die Metadaten eines Checkpoints, ohne ein Modell zu bauen, und validiert
sie gegen das LibreYOLO-Checkpoint-Schema.

```bash
libreyolo metadata path=<checkpoint.pt>
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `path` | | Pfad zu einem `.pt`-Checkpoint. Erforderlich |
| `json` | `false` | JSON-Ausgabe nach stdout |
| `quiet` | `false` | stderr unterdrücken |

Große Einträge, die Tensoren enthalten, werden zusammengefasst statt ausgegeben,
damit die Ausgabe auch bei einem vollständigen Trainings-Checkpoint lesbar
bleibt. Ein Checkpoint, der nicht existiert, beendet sich mit
`checkpoint_not_found`, und einer, dessen Metadaten die Validierung nicht
bestehen, gibt die Fehler aus und beendet sich mit `1`.

## enroll

Baut eine Gesichts-Galerie aus einem Baum mit einem Ordner je Person, damit
spätere Vorhersagen die gefundenen Gesichter benennen können.

```bash
libreyolo enroll model=<embedder> source=<people-dir> gallery=<gallery.npz>
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `model` | | Face-Embedding-Modell, Pfad oder Name. Erforderlich |
| `source` | | Baum mit einem Ordner je Person, `source/<identity>/*.jpg`. Erforderlich |
| `gallery` | | Ausgabedatei der Galerie, `.npz`. Wird erweitert, wenn sie schon existiert. Erforderlich |
| `face_detector` | | Gesichtsdetektor: eine YuNet-`.onnx` oder ein LibreYOLO-Detektor. Ohne Angabe der Standarddetektor der Familie |
| `device` | `auto` | Gerät: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | JSON-Ausgabe nach stdout |
| `quiet` | `false` | stderr unterdrücken |

```bash
# people/ enthält einen Ordner je Identität; der Ordnername wird zur Identität.
libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=people.npz
```

Der Name des Unterordners ist die Identität. Ein Referenzbild ohne erkennbares
Gesicht wird mit einer Zeile auf stderr übersprungen, und der Rest läuft weiter;
eine Quelle ohne Identitäts-Unterordner oder eine, in der überhaupt kein Gesicht
gefunden wurde, ist ein Fehler.

Übergib die entstandene Datei an
[`libreyolo predict`](/docs/cli/predict) als `gallery=people.npz`, damit
Erkennungen eine Identität und einen Match-Score tragen.

## compare

Meldet die Kosinus-Ähnlichkeit zwischen zwei Gesichtsbildern und ob sie den
Schwellenwert für dieselbe Identität überschreitet.

```bash
libreyolo compare model=<embedder> source=<a.jpg> source2=<b.jpg>
```

| Argument | Standard | Bedeutung |
|---|---|---|
| `model` | | Face-Embedding-Modell, Pfad oder Name. Erforderlich |
| `source` | | Erstes Bild. Erforderlich |
| `source2` | | Zweites Bild, mit dem verglichen wird. Erforderlich |
| `face_detector` | | Gesichtsdetektor: eine YuNet-`.onnx` oder ein LibreYOLO-Detektor |
| `threshold` | `0.4` | Schwellenwert der Kosinus-Ähnlichkeit für die Entscheidung über dieselbe Identität |
| `device` | `auto` | Gerät: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | JSON-Ausgabe nach stdout |
| `quiet` | `false` | stderr unterdrücken |

```bash
libreyolo compare model=librefacerec-l.onnx source=a.jpg source2=b.jpg
```

`libreyolo verify` ist als zweiter Name für diesen Befehl registriert und nimmt
dieselben Argumente.

Sowohl `compare` als auch `enroll` brauchen ein Modell, dessen Aufgabe Face
Embedding ist. Alles andere beendet sich mit `config_unsupported`. Als Quellen
werden sowohl lokale Bildpfade als auch `http`- und `https`-URLs akzeptiert.

## Beispiele

<code-tabs name="examples" />

## Hinweise

stdout trägt das Ergebnis; Fortschritt und Warnungen gehen nach stderr.
`json=true` gibt ein Objekt mit `schema_version` aus, das ist die Form, die du
aus einem Skript ausliest. Textausgabe ist der Standard und ist dafür gedacht,
von einem Menschen gelesen zu werden.

Exit-Codes folgen derselben Zuordnung wie im Rest der CLI: `0` bei Erfolg, `2`
bei einem Nutzungs- oder Konfigurationsfehler, `3`, wenn eine Quelle nicht
gefunden wird, `4`, wenn ein Modell oder Checkpoint nicht geladen werden kann,
und `1` bei sonstigen Laufzeitfehlern.

Verwandt: [`libreyolo doctor`](/docs/cli/doctor), der Inspektionsbefehl auf der
Datensatzseite, und [`libreyolo profile`](/docs/cli/profile), der auf der
Performance-Seite.
