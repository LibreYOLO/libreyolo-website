---
title: Checkpoints und Gewichte
seo_title: LibreYOLO-Checkpoints und -Gewichte
description: >-
  So findet, lädt und überprüft LibreYOLO Modellgewichte. Erfahre, wo sie
  gehostet und gespeichert werden, wie der Offline-Betrieb funktioniert und
  wodurch Checkpoints sicher geladen werden können.
lead: >-
  Ein LibreYOLO-Checkpoint ist ein mit torch.save gespeichertes Dictionary, das
  ein State Dict und die zu seiner Identifizierung erforderlichen Metadaten
  enthält. Diese Seite beschreibt Herkunft, Speicherort und Ladevorgang dieser
  Dateien.
keywords:
  - libreyolo gewichte
  - libreyolo checkpoints
  - libreyolo gewichte herunterladen
  - libreyolo offline
  - libreyolo hugging face
  - checkpoint metadaten
last_verified: 1.5.0
meta:
  - label: Gehostet unter
    value: 'Ein Hugging-Face-Repository pro Checkpoint:'
    links:
      - label: huggingface.co/LibreYOLO
        href: 'https://huggingface.co/LibreYOLO'
  - label: Lokaler Cache
    value: weights/ under the working directory
    mono: true
  - label: Metadatenschema
    value: v1.0
snippets:
  load:
    - label: Automatischer Download
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Ein einfacher Dateiname wird als weights/LibreYOLO9t.pt aufgelöst
        # und dort heruntergeladen, falls er noch nicht vorhanden ist.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model(SAMPLE_IMAGE).boxes)
    - label: Expliziter Pfad
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Ein Pfad mit Verzeichniskomponente wird genau wie angegeben
        # verwendet und nie aus dem Netzwerk abgerufen.
        model = LibreYOLO("/opt/models/LibreYOLO9t.pt")
        print(model.family, model.size, model.task)
  inspect:
    - label: CLI
      language: bash
      code: |
        # Liest Metadaten ohne ein Modell zu erstellen und meldet,
        # ob sie dem Schema entsprechen.
        libreyolo metadata path=weights/LibreYOLO9t.pt
    - label: JSON
      language: bash
      code: |
        libreyolo metadata path=weights/LibreYOLO9t.pt --json
    - label: Python
      language: python
      code: >
        from libreyolo.utils.serialization import (
            load_untrusted_torch_file,
            validate_checkpoint_metadata,
        )


        loaded = load_untrusted_torch_file("weights/LibreYOLO9t.pt")


        # Gibt eine Problemliste zurück. Leer bedeutet, die Datei erfüllt v1.0.

        print(validate_checkpoint_metadata(loaded))

        print(loaded["model_family"], loaded["size"], loaded["task"],
        loaded["nc"])
source_hash: 210a12baa1417cfb
---

## Suchpfad eines Checkpoints

Eine Modellreferenz ohne Verzeichniskomponente wie `LibreYOLO9t.pt` wird
relativ zum aktuellen Arbeitsverzeichnis unter `weights/` aufgelöst. Wenn
`weights/LibreYOLO9t.pt` vorhanden ist, wird diese Datei verwendet. Wenn sich
stattdessen eine Datei dieses Namens direkt im Arbeitsverzeichnis befindet,
wird sie verwendet. Andernfalls wird `weights/LibreYOLO9t.pt` zum Downloadziel.

Eine Referenz mit absolutem oder relativem Verzeichnis wird wörtlich
übernommen. Verwende diese Form, wenn Gewichte zentral gespeichert sind und
nichts heruntergeladen werden soll.

<code-tabs name="load" />

## Automatischer Download

Wenn der aufgelöste Pfad nicht existiert, liest LibreYOLO Familie, Größe und
Aufgabe aus dem Dateinamen und fragt die passende Familie nach einer
Download-URL. Die meisten Familien erstellen sie anhand der
LibreYOLO-Organisation auf Hugging Face. Dort besitzt jeder Checkpoint ein nach
der Datei benanntes Repository:

```text
https://huggingface.co/LibreYOLO/<name>/resolve/main/<name>.pt
```

Das Suffix einer Datensatzvariante bleibt Teil des Repository-Namens. Ein auf
einem anderen als dem Standarddatensatz der Familie trainierter Checkpoint wird
daher aus seinem eigenen Repository geladen, statt den Standard zu überschreiben.

Die Übertragung arbeitet defensiv, weil eine abgeschnittene Gewichtsdatei erst
später mit einer wenig hilfreichen Meldung fehlschlägt. Downloads werden in
eine `.part`-Datei gestreamt und erst nach vollständigem Abschluss atomar an
ihren Zielort verschoben. Ein unterbrochener Prozess hinterlässt daher nie
einen halbfertigen Checkpoint am endgültigen Pfad. Eine unterbrochene
Übertragung wird anhand ihres Byte-Offsets mit einem HTTP-Validator fortgesetzt
und beginnt bei null, wenn der Server eine Änderung des Objekts meldet. Fehler
werden mit exponentiellem Backoff dreimal wiederholt. Gleichzeitige Prozesse
mit demselben Ziel verwenden eine Sperrdatei, sodass zwei gemeinsam gestartete
Trainingsläufe nur einmal herunterladen. Wenn eine Familie von einem
Drittanbieterhost statt aus der LibreYOLO-Organisation lädt, kann sie eine
Prüfsumme festlegen und Dateien mit abweichendem Wert ablehnen.

Wenn `HF_TOKEN` gesetzt oder unter `~/.cache/huggingface/token` ein Token
gespeichert ist, wird es als Bearer-Token angefügt. Dies geschieht nur für URLs
auf `huggingface.co`. Eine Familie mit einem anderen Downloadhost erhält es nie.

Nicht jede Familie unterstützt automatische Downloads. Einige geben bewusst
keine URL zurück, weil die veröffentlichten Gewichte nicht weitergegeben werden
dürfen. Die Fehlermeldung erklärt dann, was du stattdessen bereitstellen musst.
Andere geben vor Beginn der Übertragung einen Lizenzhinweis aus. Er ist das
Runtime-Signal dafür, dass die Bedingungen des Checkpoints enger als die des
Codes sind. Lies ihn, statt ihn zu überspringen.

## Die Hugging-Face-Organisation

Veröffentlichte Gewichte liegen unter
[huggingface.co/LibreYOLO](https://huggingface.co/LibreYOLO), ein Repository pro
Checkpoint. Jedes Repository enthält eine Lizenz, die innerhalb einer Familie
nicht einheitlich sein muss. Eine Familie mit MIT-lizenziertem Code kann
Gewichte mit anderen Bedingungen besitzen. Das Repository ist maßgeblich. Jede
Modellseite führt die veröffentlichten Checkpoints der Familie und ihre
Lizenzen in den Abschnitten zu Checkpoints und Lizenzierung auf.

## Offline-Betrieb

Sobald die Dateien lokal vorhanden sind, benötigt die Bibliothek keinen
Netzwerkzugriff. Zwei Verfahren funktionieren:

Fülle vorab ein Verzeichnis `weights/` neben dem Ausführungsort des Jobs. Es
genügt, die Checkpoints einmal auf einem verbundenen Rechner abzurufen und das
Verzeichnis zu übertragen. Die oben beschriebene Auflösung findet sie und
greift nie auf das Netzwerk zu.

Alternativ übergibst du einen absoluten Pfad zu einem gemeinsam genutzten Ort.
Eine Referenz mit Verzeichniskomponente wird unverändert verwendet. Ein
schreibgeschützter Mount mit kuratierten Gewichten ist daher eine gültige
Konfiguration. Wenn der Prozess nicht neben einen zu konvertierenden Checkpoint
schreiben kann, verwendet die Konvertierung ein privates temporäres
Verzeichnis, statt fehlzuschlagen.

Für Datensätze gilt eine getrennte Regel. Sie werden unter `~/datasets` oder in
dem durch `LIBREYOLO_DATASETS_DIR` bezeichneten Verzeichnis aufgelöst.

## Ladesicherheit

Checkpoints sind Pickle-Dateien, und ein Pickle kann beim Öffnen beliebigen
Code ausführen. LibreYOLO behandelt jede Gewichtsdatei als nicht
vertrauenswürdig und lädt sie über den Pfad `weights_only=True` von PyTorch.
Dieser beschränkt den Unpickler auf Tensoren und einen kleinen Satz sicherer
Typen. Die Regel gilt für jede übergebene Datei und nicht nur für von LibreYOLO
heruntergeladene Dateien. Ist ein PyTorch-Build zu alt für dieses Argument,
wird das Laden abgelehnt, statt unsicher ausgeführt zu werden.

Einige Upstream-Trainings-Checkpoints enthalten Objekte, die der eingeschränkte
Unpickler ablehnt, etwa ein Konfigurationsobjekt des Trainings-Frameworks.
LibreYOLO benötigt diese Metadaten nicht. Bei der Konvertierung wird jede
blockierte Klasse durch einen inerten Platzhalter ersetzt, der den Unpickler
zufriedenstellt, ohne Code auszuführen. Nur Tensoren gelangen in die
konvertierte Datei. Sensible Modulnamen werden vollständig abgelehnt und nicht
durch Platzhalter ersetzt. Die Wiederholungsschleife ist begrenzt, sodass eine
Datei, die endlos neue blockierte Klassen einführt, sicher fehlschlägt. Den
restlichen Pfad beschreibt
[Vorhandene Gewichte importieren](/docs/migrate).

## Checkpoint-Metadaten

Ein LibreYOLO-Checkpoint ist ein Dictionary, dessen Schlüssel `model` das
PyTorch-State-Dict enthält. Schema v1.0 verlangt neun Schlüssel. Zusammen
ermöglichen sie der Factory, eine Datei ohne Auswertung ihres Namens oder
Erraten anhand der Tensorformen zu identifizieren.

| Schlüssel | Bedeutung |
|---|---|
| `model` | Das PyTorch-State-Dict |
| `schema_version` | Die Version des Metadatenvertrags. v1.0 verwendet den String `1.0` |
| `libreyolo_version` | Die LibreYOLO-Version, mit der die Datei erzeugt wurde |
| `model_family` | Ein registrierter Familienbezeichner wie `yolo9` |
| `size` | Die Variante innerhalb der Familie, zum Beispiel `t` oder `r18` |
| `task` | Ein kanonischer Aufgabenname |
| `nc` | Eine positive Klassenzahl |
| `names` | Eine Zuordnung von Klassenindex zu Label für `0` bis `nc - 1` |
| `imgsz` | Eine positive Eingabeauflösung |

Aufgaben mit zusätzlicher Struktur speichern sie neben diesen Schlüsseln.
Pose-Checkpoints ergänzen `num_keypoints` und `keypoint_dim` sowie optional
OKS-Sigmas pro Keypoint. OCR-Checkpoints betten den vollständigen CTC-Zeichensatz
ein, damit die Datei eigenständig ist. Restaurierungs-Checkpoints können den
Degradationstyp und einen Vergrößerungsfaktor speichern. Trainer-Checkpoints
ergänzen Zustände zum Fortsetzen, etwa `epoch`, den Optimizer-Zustand und die
EMA-Gewichte. Veröffentlichte Inferenzgewichte sollten diese nicht enthalten.

Eine Datei mit allen neun Schlüsseln wird über den Metadatenpfad geladen. Eine
Datei ohne vollständigen Satz wird konvertiert, wenn eine Familie ihr Layout
erkennt, oder über den Kompatibilitätspfad geladen. Dabei nennt eine Warnung die
fehlenden Angaben.

## Untersuchen eines Checkpoints

<code-tabs name="inspect" />

`libreyolo metadata` erstellt nie ein Modell. Der Befehl funktioniert daher
für eine Datei, deren Familie nicht installiert ist, und für Dateien, bei denen
du dir noch unsicher bist.

