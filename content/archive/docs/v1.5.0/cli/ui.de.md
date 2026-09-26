---
title: libreyolo ui
seo_title: Befehlsreferenz libreyolo ui
description: >-
  Die lokale Web-UI für Inferenz starten: Bind-Adresse, Port-Verhalten,
  Geräteauswahl und wie der Befehl endet.
lead: >-
  Startet einen lokalen Webserver, der abgelegte oder eingefügte Bilder annimmt,
  ein gewähltes Modell darauf ausführt und die Ergebnisse im Browser zeigt.
keywords:
  - libreyolo ui cli
  - libreyolo web ui
  - lokale inferenz im browser
  - objekterkennung per drag and drop
  - libreyolo ui port ändern
last_verified: 1.5.0
meta:
  - label: Befehl
    value: libreyolo ui
    mono: true
  - label: Ausgabe
    value: 'Eine Server-URL auf stdout, danach bleibt der Prozess im Vordergrund'
snippets:
  examples:
    - label: Basis
      language: bash
      code: |
        libreyolo ui
    - label: 'Fester Port, kein Browser'
      language: bash
      code: |
        libreyolo ui port=9000 no_browser=true
    - label: 'Auf der CPU, maschinenlesbar'
      language: bash
      code: |
        libreyolo ui device=cpu json=true
source_hash: b0eebd33fd0f463b
---

## Syntax

```bash
libreyolo ui [key=value ...]
```

Argumente sind `key=value`-Paare, und die POSIX-Form funktioniert ebenfalls,
`port=9000` und `--port 9000` sind also dasselbe Argument.

## Argumente

| Argument | Default | Bedeutung |
|---|---|---|
| `host` | `127.0.0.1` | Host oder Interface, an das gebunden wird |
| `port` | `8000` | Port, an den gebunden wird. Rückt auf den nächsten freien vor, wenn er belegt ist |
| `device` | `auto` | Gerät: `0`, `cpu`, `mps`, `auto` |
| `no_browser` | `false` | Den Browser nicht automatisch öffnen |
| `json` | `false` | JSON-Ausgabe auf stdout |
| `quiet` | `false` | stderr unterdrücken |
| `verbose` | `false` | Ausführliche Ausgabe auf stderr |

## Beispiele

<code-tabs name="examples" />

## Hinweise

Standardmäßig wird an das Loopback-Interface gebunden, die UI ist also nur von
diesem Rechner aus erreichbar.

Ist der angeforderte Port belegt, probiert der Befehl den nächsten und arbeitet
sich bis zu zwanzig Ports über die Anforderung hinaus vor. Schlagen alle zwanzig
fehl, endet er mit `io_error` und dem Hinweis, einen anderen Port zu übergeben.
Die auf stdout ausgegebene URL nennt den Port, an den tatsächlich gebunden
wurde, lies sie also, statt den angefragten vorauszusetzen.

Sofern du nicht `no_browser=true` setzt, öffnet sich kurz nach dem Binden ein
Browser-Tab mit dieser URL.

Der Befehl bedient Anfragen danach im Vordergrund, bis du Ctrl+C drückst, was
den Server sauber herunterfährt. Einen Detached-Modus gibt es nicht; schick ihn
mit deiner Shell in den Hintergrund, wenn du das Terminal zurückhaben willst.

`json=true` gibt URL und Gerät als ein Objekt mit `schema_version` aus, bevor
der Server startet, und genau so holt sich ein Skript den gebundenen Port.

Verwandt: [`libreyolo label`](/docs/cli/label) zum Zeichnen von Boxen und
Speichern von Labels, [`libreyolo monitor`](/docs/cli/monitor) zum Verfolgen von
Trainingsläufen. Beide sind lokale Webserver mit demselben Port- und
Browser-Verhalten.
