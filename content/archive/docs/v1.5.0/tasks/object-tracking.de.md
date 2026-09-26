---
title: Objekt-Tracking
seo_title: Objekt-Tracking in LibreYOLO
description: >-
  Verfolge mit ByteTrack, BoT-SORT, OC-SORT oder Deep OC-SORT Objekte über
  Videoframes hinweg. Verwende dazu jedes LibreYOLO-Modell für Erkennung,
  Segmentierung oder Pose.
lead: >-
  Tracking weist jeder Erkennung über Videoframes hinweg eine stabile Identität
  zu. LibreYOLO modelliert es nicht als Aufgabe mit eigenen Gewichten. Es ist
  ein Vorhersagemodus namens model.track(), der einen ausgewählten Tracker auf
  die frameweise Ausgabe eines Erkennungs-, Segmentierungs- oder Pose-Modells
  anwendet.
keywords:
  - objekt tracking python
  - multi object tracking
  - bytetrack
  - botsort
  - ocsort
  - deep ocsort
  - track id
  - reid tracking
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # track() ist ein Generator: ein Results-Objekt je verarbeitetem Frame.
        for result in model.track("video.mp4"):
            print(result.track_id)        # (N,) Int-Tensor, an Boxen ausgerichtet
            print(result.boxes.xyxy)
    - label: Tracker auswählen
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # "bytetrack" (Standard), "botsort", "ocsort" oder "deepocsort".
        for result in model.track("video.mp4", tracker="botsort"):
            print(result.track_id)
    - label: Annotiertes Video speichern
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Ohne output_path landet die Datei in runs/track/<video_stem>.mp4.
        for result in model.track("video.mp4", save=True, vid_stride=2):
            pass
    - label: Tracker abstimmen
      language: python
      code: >
        from libreyolo import BoTSortConfig, LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Der Konfigurationstyp wählt den Tracker, tracker= ist hier redundant.

        config = BoTSortConfig(track_buffer=60, frame_rate=25, enable_cmc=False)

        for result in model.track("video.mp4", tracker_config=config):
            print(result.track_id)

        # Oder dieselben Felder als Keyword-Argumente übergeben.

        for result in model.track("video.mp4", tracker="botsort",
        track_buffer=60):
            print(result.track_id)
source_hash: f1fa7dcf60597d6b
---

## Definition

Tracking ist keiner der Aufgabenschlüssel von LibreYOLO, und es gibt keinen
Tracking-Checkpoint zum Herunterladen. Es ist eine Modellmethode namens
`model.track(source)`, die in jedem Frame Erkennungen ausführt und die
Ergebnisse zeitlich zuordnet. Die Methode ist ein Generator. Sie liefert pro
verarbeitetem Frame ein `Results`-Objekt, in dem `result.track_id` auf einen an
`result.boxes` ausgerichteten ganzzahligen Tensor der Form `(N,)` gesetzt ist.
Dieselben IDs stehen auch in `result.boxes.id`.

Nur bestätigte und aktuell verfolgte Objekte werden ausgegeben. Ein Track, den
die Zuordnung verliert, bleibt für eine konfigurierte Anzahl von Frames aktiv,
bevor er verworfen wird. Diese Option heißt `track_buffer` bei ByteTrack und
BoT-SORT sowie `max_age` bei den beiden OC-SORT-Varianten. Ein innerhalb dieses
Fensters wiedergefundenes Objekt behält daher seine ursprüngliche ID.

Da die Zuordnung nach der Erkennung erfolgt, bleiben die anderen Payloads des
Frames erhalten. Das verfolgte `Results`-Objekt ist die auf die zugeordneten
Zeilen beschränkte Erkennungsausgabe. Masken und Keypoints werden daher
zusammen mit den Boxen übernommen.

## Modelle

Ein Tracking-Lauf umfasst zwei unabhängige Entscheidungen: das Modell, das in
jedem Frame Boxen erzeugt, und den Tracker, der sie verknüpft.

Jedes native LibreYOLO-Modell für Erkennung, Segmentierung oder Pose stellt
`track()` bereit. Der Detektor wird daher wie üblich ausgewählt. Eine
vollständige Liste findest du im [Modellindex](/docs/models). Du kannst auch
mit [YOLO9](/docs/models/yolov9), [RF-DETR](/docs/models/rf-detr),
[D-FINE](/docs/models/d-fine) oder [RTMDet](/docs/models/rtmdet) beginnen.
Aufgaben ohne zuordenbare Box lehnen den Aufruf ab, statt bedeutungslose IDs
zurückzugeben. Klassifikation, orientierte Boxen, Punkte, Tiefe,
Oberflächennormalen, Kanten, semantische und panoptische Segmentierung,
Restaurierung, OCR und Body Mesh lösen jeweils in `track()` einen Fehler aus.

Zwei Modellstufen von LibreYOLO lehnen den Aufruf ebenfalls ab. Über `LibreSAM`
geladene Modelle sind Bildsegmentierer, und über `LibreOpenVocab` geladene
Modelle sind frameweise Detektoren. Bei beiden löst `track()` einen Fehler aus.
Verwende stattdessen für jeden Frame `predict()`.

Tracking läuft auf nativen PyTorch-Modellen. Ein exportiertes und mit
`LibreYOLO("model.onnx")` geladenes Artefakt gibt ein Runtime-Backend-Objekt
zurück. Dieses stellt `predict()`, aber nicht `track()` bereit.

Die Bibliothek enthält vier Tracker, die du mit dem Argument `tracker` auswählst:

`"bytetrack"` ist der Standard. Es verwendet ausschließlich Bewegung, einen
Kalman-Filter und eine dreistufige Zuordnung: zuerst Erkennungen mit hoher
Confidence, anschließend einen zweiten Durchlauf, in dem Erkennungen mit
niedriger Confidence einem bestehenden Track zugeordnet werden können, bevor
sie verworfen werden, und zuletzt unbestätigte Tracks. Die Konfiguration
erfolgt mit `TrackConfig`.

`"botsort"` übernimmt den dreistufigen Lebenszyklus von ByteTrack, verwendet
aber einen Kalman-Zustand aus Mittelpunkt, Breite und Höhe. Vor der Zuordnung
kompensiert es die vorhergesagten Tracks um die Kamerabewegung. Dies ist die
nur bewegungsbasierte Variante von BoT-SORT und verwendet kein
Erscheinungsmodell. Die Konfiguration erfolgt mit `BoTSortConfig`, das
`enable_cmc`, `cmc_method` und `cmc_downscale` ergänzt.

`"ocsort"` verwendet ebenfalls nur Bewegung. Es ergänzt die Zuordnungskosten
um einen Term für die Geschwindigkeitsrichtung, führt einen zweiten
Zuordnungsdurchlauf anhand der letzten echten Beobachtung jedes Tracks aus und
glättet beim Wiederfinden eines Tracks den Kalman-Zustand entlang einer
virtuellen Trajektorie. Die Konfiguration erfolgt mit `OCSortConfig`.

`"deepocsort"` ergänzt OC-SORT um Erscheinungsmerkmale. Jeder Track hält einen
nach Confidence gewichteten gleitenden Mittelwert aus
Re-Identification-Embeddings. Ein Term für Kosinusähnlichkeit geht in die
Zuordnungskosten ein, damit Identitäten lange Verdeckungen und sich kreuzende
Ziele überstehen. Dafür ist pro Frame ein Forward Pass durch ein kleines
Embedding-Netz erforderlich. Seine OSNet-Gewichte werden bei der ersten
Verwendung heruntergeladen. Die Konfiguration erfolgt mit `DeepOCSortConfig`.

## Vorhersage

<code-tabs name="predict" />

`track_conf` legt den Schwellenwert für die erste Zuordnungsstufe fest:
`track_high_thresh` für ByteTrack und BoT-SORT sowie `det_thresh` für OC-SORT
und Deep OC-SORT. Dies ist nicht `conf` von `predict()`. Bei ByteTrack,
BoT-SORT und OC-SORT läuft der Detektor intern mit einem niedrigeren
Schwellenwert, damit schwache Erkennungen für den Wiederherstellungsdurchlauf
verfügbar bleiben. Deep OC-SORT führt den Detektor mit `det_thresh` selbst aus.
Bei ByteTrack und BoT-SORT muss `track_conf` mindestens `track_low_thresh`
entsprechen, dessen Standardwert 0.1 ist.

Tracker-Einstellungen können auf zwei Arten übergeben werden. Übergib eine
Konfigurationsinstanz als `tracker_config=`. Ihr Typ wählt den Tracker aus und
macht `tracker=` überflüssig. Alternativ übergibst du die Felder als
Keyword-Argumente und lässt `track()` die Konfiguration für den benannten
Tracker erstellen. Bei unbekannten Schlüsseln wird eine Warnung ausgegeben,
statt sie unbemerkt anzuwenden. In beiden Fällen wird `track_conf` ignoriert,
sobald der entsprechende Schlüssel explizit gesetzt ist.

Die übrigen Argumente entsprechen der Vorhersage: `iou`, `imgsz`, `classes`,
`max_det`, `vid_stride`, `show` und `save` zusammen mit `output_path`. Als
Quelle dient der Pfad zu einer Videodatei. Unter [Vorhersage](/docs/predict)
findest du Informationen zur Ergebnisverarbeitung.

## Training

Tracker werden nicht trainiert. Drei der vier sind reine Bewegungsmodelle ohne
gelernte Parameter. Das Erscheinungsnetz von Deep OC-SORT ist ein
veröffentlichter Re-Identification-Checkpoint, der bei der ersten Verwendung
heruntergeladen wird. Verbessere für eine höhere Tracking-Qualität den Detektor
oder stimme die oben genannten Zuordnungsschwellenwerte ab.

