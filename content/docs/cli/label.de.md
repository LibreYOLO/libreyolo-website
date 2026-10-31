---
title: libreyolo label
seo_title: Befehlsreferenz libreyolo label
description: >-
  Startet das lokale Annotationswerkzeug für Bounding Boxes: Argumente mit
  Standardwerten, der Schalter für das KI-Auto-Labeling und was die Bindung an
  eine Netzwerkschnittstelle freigibt.
lead: >-
  Startet ein lokales Web-Tool zum Zeichnen und Bearbeiten von Bounding Boxes.
  Es schreibt LibreYOLO-native Label-Dateien, sodass sich ein hier annotierter
  Datensatz ohne Konvertierungsschritt trainieren lässt.
keywords:
  - libreyolo label cli
  - bounding boxes annotieren tool
  - yolo datensatz labeln
  - automatisches labeling tool
  - libreyolo label im netzwerk teilen
last_verified: 1.5.0
meta:
  - label: Befehl
    value: libreyolo label
    mono: true
  - label: Ausgabe
    value: >-
      Eine Server-URL auf stdout; Labels werden als labels/*.txt neben den
      Bildern geschrieben
snippets:
  examples:
    - label: Basis
      language: bash
      code: |
        # Öffnet die Projektübersicht; Datensatz im Browser wählen oder anlegen.
        libreyolo label
    - label: 'Nur manuell, fester Port'
      language: bash
      code: |
        libreyolo label no_assist=true port=9200 no_browser=true
    - label: Teammitglieder dazuholen
      language: bash
      code: |
        libreyolo label share=true
source_hash: bc8acc78af75408f
---

## Synopsis

```bash
libreyolo label [data=<dataset.yaml|folder>] [key=value ...]
```

Argumente sind `key=value`-Paare, und die POSIX-Form funktioniert ebenfalls,
`port=9200` und `--port 9200` sind also dasselbe Argument.

## Argumente

| Argument | Standard | Bedeutung |
|---|---|---|
| `data` | | Dataset-YAML oder Ordner, der direkt geöffnet wird. Startet ohne Angabe auf der Projektübersicht |
| `host` | `127.0.0.1` | Host oder Schnittstelle, an die gebunden wird |
| `port` | `8000` | Port, an den gebunden wird. Rückt auf den nächsten freien vor, wenn er belegt ist |
| `device` | `auto` | Gerät für das KI-Auto-Labeling: `0`, `cpu`, `mps`, `auto` |
| `no_assist` | `false` | KI-Auto-Labeling abschalten, es bleibt ein manueller Labeler |
| `no_browser` | `false` | Den Browser nicht automatisch öffnen |
| `share` | `false` | An `0.0.0.0` binden, damit Teammitglieder in deinem Netzwerk beitreten können |
| `json` | `false` | JSON-Ausgabe auf stdout |
| `quiet` | `false` | stderr unterdrücken |
| `verbose` | `false` | Ausführliche stderr-Ausgabe |

## Beispiele

<code-tabs name="examples" />

## Hinweise

### Was geschrieben wird

Boxen werden als LibreYOLO-native `labels/*.txt`-Dateien gespeichert, also in
dem Format, das `libreyolo train` liest, sodass hinterher nichts konvertiert
werden muss. Diese Version verarbeitet ausschließlich Bounding Boxes.
Änderungen werden gespeichert, sobald du zwischen den Bildern wechselst.

### Öffnen eines Datensatzes

Ohne `data` startet das Tool auf der Projektübersicht, und der Datensatz wird im
Browser gewählt oder angelegt. `data=path/to/data.yaml` öffnet diesen Datensatz
direkt, und die Startzeile meldet die Anzahl der Bilder, die Anzahl der Klassen
und ob der Datensatz beschreibbar ist. Ein schreibgeschützter Datensatz öffnet
sich trotzdem und nennt den Grund, warum er nicht beschrieben werden kann.

### Label-Error-Radar

Mit eingeschaltetem KI-Auto-Labeling lässt der Radar-Button, oder `Y`, das
Assist-Modell über jedes gelabelte Bild laufen und listet die Bilder, bei denen
das Modell den Labels widerspricht, die schlimmsten zuerst: eine Box, in der das
Modell nichts findet, eine Detektion mit hoher Confidence ohne Box oder eine
Box, deren Klasse das Modell anzweifelt. Radar schreibt nichts; jede Korrektur
ist eine manuelle Bearbeitung. `N` springt in dieser Reihenfolge zum nächsten
markierten Bild und beginnt am Ende wieder von vorn. Ein erneutes `Y` öffnet
den letzten Scan wieder, mit dem Zeitpunkt seines Laufs und einem
Rescan-Button, statt neu zu scannen. Radar ist in Projekten mit orientierten
Boxen nicht verfügbar, und beide Tasten lassen sich im Shortcut-Dialog `?` neu
belegen.

### Teilen, und was `host` bewirkt

`share=true` bindet die Wildcard-Adresse, sodass andere Rechner in deinem
Netzwerk das Tool erreichen, während administrative Aktionen, das Wechseln oder
Löschen von Projekten und das Starten von Berechnungen auf diesem Rechner
bleiben.

`host` auf eine bestimmte Schnittstelle zu setzen, bewirkt etwas anderes und
weniger Sicheres: Der Host wird von einem Netzwerk-Client ununterscheidbar, also
bekommt jeder Client administrative Rechte. Der Befehl gibt dabei eine Warnung
auf stderr aus. Nimm lieber `share=true`.

### Ports und Beenden

Ein belegter Port rückt auf den nächsten weiter, bis zu zwanzig über die
Anfrage hinaus. Scheitern alle zwanzig, endet der Befehl mit `io_error`. Die auf
stdout ausgegebene URL nennt den Port, der tatsächlich gebunden wurde. Mit
`share=true` trägt das Ergebnis zusätzlich `lan_url`, die Adresse, die
Teammitglieder öffnen sollten.

Der Befehl läuft im Vordergrund, bis du Ctrl+C drückst.

Verwandt: [`libreyolo doctor`](/docs/cli/doctor), um den gelabelten Datensatz
vor dem Training zu prüfen, und [`libreyolo train`](/docs/cli/train), um darauf
zu trainieren.
