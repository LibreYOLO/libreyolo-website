---
title: libreyolo doctor
seo_title: libreyolo doctor Befehlsreferenz
description: >-
  Prüfe einen Detection-Datensatz vor dem Training: Argumente mit Defaults, die
  Prüffamilien, die du überspringen oder auswählen kannst, und die Exit-Codes,
  an denen CI abbrechen kann.
lead: >-
  Führt eine Reihe von Health-Checks über einen Detection-Datensatz aus und
  meldet, was einem Trainingslauf schaden würde: fehlende Dateien, kaputte
  Labels, beschädigte Bilder, Leakage zwischen Splits und
  Klassenungleichgewicht.
keywords:
  - libreyolo doctor cli
  - yolo datensatz prüfen
  - yolo datensatz validieren
  - datensatz leakage prüfen
  - libreyolo doctor strict
last_verified: 1.5.0
meta:
  - label: Befehl
    value: libreyolo doctor
    mono: true
  - label: Erforderlich
    value: data
    mono: true
  - label: Ausgabe
    value: 'Ein Befundbericht auf stdout. Exit 1, wenn Fehler gefunden werden'
snippets:
  examples:
    - label: Basis
      language: bash
      code: >
        # download=true lässt die mitgelieferte coco8.yaml fehlende Bilder
        holen.

        libreyolo doctor coco8.yaml download=true
    - label: 'Schneller Durchlauf, ohne Bilddekodierung'
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true fast=true
    - label: CI-Gate für ausgewählte Prüfungen
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true strict=true json=true \
          only=labels,files,config
source_hash: 79e0ef471d567ea3
---

## Synopsis

```bash
libreyolo doctor <data.yaml> [key=value ...]
```

Der Datensatz steht positional, und `data=<path>` wird als Alternative
akzeptiert. Beides mit unterschiedlichen Werten anzugeben, endet mit
`config_conflict`. Alles andere ist ein `key=value`-Paar, und die POSIX-Form
funktioniert ebenfalls, sodass `imgsz=1024` und `--imgsz 1024` dasselbe
Argument sind.

## Argumente

| Argument | Default | Bedeutung |
|---|---|---|
| `data` | | Positional. Datensatz-YAML im YOLO-Detection-Format, z. B. `coco8.yaml`. Erforderlich |
| `imgsz` | `640` | Trainings-Bildgröße für pixelbasierte Prüfungen wie winzige Objekte |
| `fast` | `false` | Überspringt die Bilddekodierung, wodurch die Prüfungen auf Beschädigung, Duplikate und Leakage entfallen |
| `skip` | | Kommagetrennte Prüf-IDs oder Familien, die übersprungen werden, z. B. `images,labels.tiny_object` |
| `only` | | Kommagetrennte Prüf-IDs oder Familien, die ausschließlich laufen |
| `strict` | `false` | Warnungen lassen den Exit-Code ebenfalls fehlschlagen, für CI-Gates |
| `download` | `false` | Erlaubt den URL-basierten Download des Datensatzes, falls er fehlt. Niemals Skripte |
| `json` | `false` | JSON-Ausgabe auf stdout |
| `quiet` | `false` | Unterdrückt stderr |
| `help_json` | `false` | Gibt das Befehlsschema als JSON aus und beendet sich |

### Prüffamilien

`skip` und `only` akzeptieren entweder eine vollständige Prüf-ID oder ein
Familien-Präfix, sodass `images` jede `images.*`-Prüfung auswählt.

| Familie | Deckt ab |
|---|---|
| `config` | Die Datensatz-YAML selbst: fehlende `names`, `nc` gegen `names`, fehlende Splits, nicht auflösbarer `path`, doppelte Klassennamen |
| `files` | Zuordnung von Bildern und Labels: fehlende Labels, fehlende Bilder, verwaiste Labels, nicht unterstützte Dateiendungen, Kollisionen bei Groß- und Kleinschreibung |
| `labels` | Label-Inhalt: Syntax, Polygonzeilen, Klassen-IDs außerhalb des Bereichs, Koordinaten außerhalb des Bereichs, entartete Boxen, winzige Objekte, riesige Boxen, extreme Seitenverhältnisse, doppelte Boxen, überfüllte Bilder, identische Dateien |
| `images` | Pixeldaten: beschädigte Dateien, EXIF-Orientierung, ungewöhnliche Farbmodi, winzige oder extreme Abmessungen, einfarbige Bilder, exakte und nahe Duplikate |
| `splits` | Leakage zwischen Splits, exakt und nah |
| `balance` | Klassenverteilung: Klassen mit null oder wenigen Instanzen, Ungleichgewicht, Split-Abdeckung, Hintergrundanteil, Split-Schieflage |

## Beispiele

<code-tabs name="examples" />

## Hinweise

### Exit-Codes

`0`, wenn keine Fehler gefunden wurden, `1`, wenn irgendein Befund ein Fehler
ist. Mit `strict=true` heben auch Warnungen den Exit-Code auf `1`, und das ist
die Einstellung, die ein CI-Gate will.

Bedienfehler haben eigene Codes: `2` für eine unbekannte Prüf-ID oder Familie
in `skip` oder `only`, `3`, wenn der Datensatz nicht gefunden wird, und `3`,
wenn der Datensatz nicht die Form eines Detection-Datensatzes hat.

### Die Auswahl wird vor dem Scan aufgelöst

`skip` und `only` werden gegen die Prüf-Registry aufgelöst, bevor irgendetwas
von der Festplatte gelesen wird, sodass ein Tippfehler sofort fehlschlägt und
nicht erst nach einem langen Bilddurchlauf. Ein Selektor, der auf nichts passt,
ist ein Fehler, und die Meldung listet die bekannten Familien auf.

Wenn die Kombination aus `skip`, `only` und `fast` keine Prüfungen übrig lässt,
ist das ebenfalls ein Fehler und kein stiller Durchlauf.

### Downloads

Der Datensatz wird nur mit `download=true` geholt, und es werden ausschließlich
URL-Downloads ausgeführt. Ein eingebettetes Python-Download-Skript in einer
Datensatz-YAML wird von diesem Befehl nie ausgeführt, egal wie das Flag steht.

### Geltungsbereich

Die Prüfungen sind für Detection-Datensätze geschrieben. Ein Datensatz, dessen
Labels die Form von Pose, Segmentierung oder orientierten Boxen haben, wird
erkannt und mit `data_invalid` abgelehnt, statt gegen die falschen Regeln
bewertet zu werden.

### Ausgabe

Der menschenlesbare Bericht geht auf stdout, und `json=true` ersetzt ihn durch
ein strukturiertes Objekt mit den Summenzahlen, den Datensatz-Statistiken,
jedem Befund und der Liste der übersprungenen Prüfungen.

Verwandt: [`libreyolo train`](/docs/cli/train), der Lauf, vor dem dieser Befehl
ausgeführt werden soll.
