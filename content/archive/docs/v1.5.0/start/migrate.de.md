---
title: Vorhandene Gewichte importieren
seo_title: Upstream-Gewichte in LibreYOLO laden
description: >-
  Übergib LibreYOLO einen Checkpoint aus einem Upstream-Projekt. Die
  automatische Konvertierung verpackt ihn beim Laden neu und behält Klassenzahl
  sowie Namen bei.
lead: >-
  LibreYOLO portiert seine Modellfamilien aus Upstream-Projekten. Deren
  veröffentlichte Checkpoints sind daher bereits fast direkt ladbar. Es fehlen
  lediglich Metadaten, die bei der automatischen Konvertierung ergänzt werden.
keywords:
  - libreyolo gewichte konvertieren
  - upstream checkpoint laden
  - libreyolo migration
  - pth zu libreyolo konvertieren
  - automatische checkpoint-konvertierung
last_verified: 1.5.0
meta:
  - label: Einstiegspunkt
    value: LibreYOLO("path/to/upstream.pth")
    mono: true
  - label: Neben der Quelle geschrieben als
    value: '<source>-<Prefix><size>[-task].pt'
    mono: true
  - label: Konvertierungsskripte
    value: weights/ in the repository
    mono: true
snippets:
  convert:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Ersetze den Pfad durch einen vorhandenen Checkpoint. Ein erkanntes

        # Upstream-Layout wird direkt konvertiert, daneben geschrieben und
        geladen.

        model = LibreYOLO("path/to/upstream-checkpoint.pth")


        # Klassenzahl und Namen stammen aus Tensoren und Dateimetadaten. Ein

        # Fine-Tuning behält daher seinen Labelsatz statt des COCO-Satzes.

        print(model.family, model.size, model.task, model.nb_classes)

        print(model.names)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=path/to/upstream-checkpoint.pth \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Ergebnis prüfen
      language: bash
      code: >
        # Die konvertierte Datei erfüllt dasselbe Schema wie eine
        veröffentlichte.

        libreyolo metadata path=path/to/upstream-checkpoint-LibreYOLO9t.pt
source_hash: bf9d7c7d168fd2c0
---

Diese Seite behandelt Checkpoints aus anderen Projekten. Wenn du eigenen Code
von einer älteren LibreYOLO-Version migrierst, lies
[Upgrade auf 1.5.0](/docs/upgrade).

## Vorgang beim Laden einer fremden Datei

`LibreYOLO()` lädt jede Gewichtsdatei zunächst über den eingeschränkten
Weights-Only-Pfad. Wenn das Ergebnis vollständige LibreYOLO-Metadaten enthält,
wird es direkt verwendet. Andernfalls wird die Datei vor jedem weiteren
Versuch an den automatischen Konverter übergeben. Wenn der eingeschränkte
Ladevorgang vollständig fehlschlägt, was bei einem Checkpoint mit einem
serialisierten Drittanbieterobjekt geschieht, wird der automatische Konverter
mit einem Loader versucht, der diese Objekte neutralisiert.

Die automatische Konvertierung führt vier Schritte aus. Zunächst entpackt sie
das Tensor-Dictionary aus dem vom Upstream-Projekt verwendeten Layout. Dann
fragt sie jede registrierte Familie, ob sie die resultierenden Schlüssel
erkennt, und ordnet Namen neu zu, wenn die Upstream-Benennung von der
LibreYOLO-Portierung abweicht. Anschließend verpackt sie den Treffer in einen
Checkpoint nach Metadatenschema v1.0 und liest Größe, Aufgabe sowie Klassenzahl
aus den Tensoren. Zuletzt schreibt sie das Ergebnis neben die Quelldatei und
lädt es.

<code-tabs name="convert" />

Die Konvertierung geschieht nicht unbemerkt. Für eine konvertierte Datei werden
Familie, Quellname, Ausgabename und resultierende Klassenzahl protokolliert.
Damit hält das Protokoll eines Laufs genau fest, was geladen wurde.

## Entpackte Layouts

Upstream-Checkpoints verschachteln ihre Gewichte an einigen üblichen Stellen.
Der Konverter prüft sie der Reihe nach, bis eine davon Tensoren enthält: einen
EMA-Block unter `ema.module` oder ein flaches `ema`, ein `ema_state_dict` nach
Entfernung seines Präfixes `module.`, anschließend `params_ema`, `params`,
`ema_net`, `net`, `model`, `state_dict` und zuletzt das Objekt selbst. Durch
die Prüfung mehrerer Stellen statt nur der ersten kann ein `ema`-Block, der
lediglich Zähler enthält, die darunterliegenden tatsächlichen Gewichte nicht
verdecken.

Auch Wrapper-Präfixe werden entfernt: `module.` aus verteiltem Training,
`_orig_mod.` aus einem kompilierten Modell und die Verschachtelung
`model.model.`, die einige Weiterverteilungen ergänzen.

## Gelesene Informationen und ihre Quellen

Größe, Aufgabe und Klassenzahl stammen aus den Tensoren und nicht aus dem
Dateinamen. Daher wird ein nachtrainierter Checkpoint mit seiner eigenen
Klassenzahl statt dem Standard der Architektur konvertiert. Klassennamen werden
aus den eigenen Metadaten des Checkpoints übernommen, falls vorhanden, oder
aus einem Block `args` beziehungsweise `hyper_parameters`, wenn sie dort
liegen. Sie werden auf die erkannte Klassenzahl gekürzt. Dadurch behält ein
Fine-Tuning, das den Labelsatz des Basismodells mitführt, keine Indizes, für die
sein Head keine Ausgabe mehr besitzt.

Dichte Aufgaben werden explizit behandelt, statt künstliche Labels zu
erhalten. Ein Tiefen-Checkpoint bekommt eine Klasse namens `depth`, ein
Restaurierungs-Checkpoint eine Klasse namens `image`. Bei einem
Pose-Checkpoint muss eine Keypoint-Anzahl aus den Tensoren oder der Familie
ermittelt werden können. Wenn beides fehlschlägt, wird die Konvertierung
abgelehnt, statt eine unvollständige Datei zu schreiben.

RF-DETR besitzt einen eigenen Erkenner. Die Größenbestimmung benötigt den
gesamten Checkpoint, und der Head besitzt 91 Ausgaben, während LibreYOLO die
COCO-Konvention mit 80 Klassen verwendet. Ein Checkpoint wird auf 80 Klassen
normalisiert, wenn er genau 80 Namen enthält, eine Klassenzahl von 80 angibt,
COCO als Datensatz nennt oder überhaupt keine Klassen- beziehungsweise
Datensatzmetadaten enthält. Ein echtes 90-Klassen-Modell, das durch seine Namen,
eine explizite von 80 abweichende Anzahl oder einen Hinweis auf einen anderen
Datensatz erkennbar ist, bleibt unverändert.

## Speicherort der konvertierten Datei

Die Ausgabe wird neben der Quelle unter einem davon abgeleiteten Namen
geschrieben:

```text
<source stem>-<FilenamePrefix><size>[-<task suffix>].pt
```

Ein kleiner YOLOv9-Detektor in `upstream-checkpoint.pth` wird daher zu
`upstream-checkpoint-LibreYOLO9t.pt`. Die Benennung nach der Quelle statt nach
der Familie verhindert, dass sich zwei Fine-Tunings derselben Familie und
Größe in einem Verzeichnis überschreiben oder mit einem offiziellen Checkpoint
kollidieren. Die Datei wird bei jedem Laden neu geschrieben und kann daher
gegenüber ihrer Quelle nicht veralten. Bei einem schreibgeschützten
Verzeichnis wird die konvertierte Datei stattdessen in ein neues privates
temporäres Verzeichnis geschrieben. Das Protokoll nennt diesen Ort.

Von da an ist die Datei ein gewöhnlicher LibreYOLO-Checkpoint. Sie wird über
den Metadatenpfad geladen und von `libreyolo metadata` als gültig gemeldet.

## Manuell zu behandelnde Fälle

Zwei Familien liegen außerhalb des generischen Erkenners. Die Blickfamilie ist
vollständig ausgeschlossen. Sie unterstützt nur Inferenz, und ihre
veröffentlichten Gewichte unterliegen Einschränkungen für die Weitergabe.
RF-DETR ist ausgeschlossen, weil stattdessen der oben beschriebene eigene
Erkenner zuständig ist.

Rohe Upstream-Checkpoints von PIDNet werden mit einem Fehler und einem Verweis
auf `weights/convert_pidnet_weights.py` abgelehnt. Dieses Skript schreibt die
für den Checkpoint erforderlichen semantischen Cityscapes-Metadaten.

D-FINE und DEIM verwenden dieselben Architekturschlüssel. Die Tensoren allein
können sie daher nicht unterscheiden. Wenn beide eine Datei beanspruchen und
keine verwandte Familie mit einem eindeutigen Merkmal beteiligt ist,
entscheidet der Dateiname. Ein Name der Form `dfine_hgnetv2_n_coco.pth` oder
`deim_hgnetv2_n_coco.pth` löst die Mehrdeutigkeit auf. Ein nichtssagender Name
wird mit einer Erklärung abgelehnt, statt erraten zu werden. Die direkte
Instanziierung von `LibreDFINE` oder `LibreDEIM` löst den Fall ebenfalls auf.

Wenn mehrere Familien berechtigt dieselbe Datei beanspruchen, hat eine
Unterklasse Vorrang vor der von ihr verfeinerten Basisklasse. In allen übrigen
Fällen entscheidet die Registry-Reihenfolge, da sie die Spezifität der
Familienprüfungen abbildet. Nur bei der Mehrdeutigkeit zwischen D-FINE und DEIM
wird der Dateiname berücksichtigt. Ein Dateiname kann daher nie eine breite
Übereinstimmung gegenüber einer präzisen bevorzugen.

## Konvertierungsskripte

Das Repository enthält unter `weights/` Konvertierungsskripte pro Familie sowie
gemeinsame Hilfsfunktionen für wiederkehrende Schritte. Sie sind für Dateien
gedacht, die der Runtime-Pfad ablehnt, für die vorzeitige Erstellung eines
Checkpoints statt der Konvertierung beim Laden und für Familien, deren
Metadaten bereitgestellt werden müssen, weil sie nicht aus Tensoren abgeleitet
werden können.

Diese Skripte sind Teil des Repositorys und nicht des installierten Pakets. Um
eines davon zu verwenden, musst du das Repository klonen:

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
python weights/convert_pidnet_weights.py --help
```

Jedes Skript schreibt einen Checkpoint nach Schema v1.0. Damit erfüllt es
denselben Standard wie die automatische Konvertierung und veröffentlichte
Gewichte. Unter [Checkpoints und Gewichte](/docs/weights) erfährst du, was
dieses Schema enthält.

