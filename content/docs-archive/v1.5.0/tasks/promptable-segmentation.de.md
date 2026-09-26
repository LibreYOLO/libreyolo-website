---
title: Promptbasierte Segmentierung
seo_title: Promptbasierte Segmentierung in LibreYOLO
description: >-
  Erzeuge in LibreYOLO aus einem Punkt, einer Box oder einem Textkonzept eine
  Objektmaske. Lade SAM, SAM 2, SAM 3, EdgeTAM, MobileSAM oder PicoSAM3 über
  LibreSAM.
lead: >-
  Die promptbasierte Segmentierung verwandelt einen Klick in eine Maske: Du
  zeigst auf ein Objekt oder zeichnest eine Box darum, und das Modell gibt
  seinen Umriss zurück. In LibreYOLO ist dies kein eigener Aufgabenschlüssel,
  sondern eine über die LibreSAM-Factory geladene Modellstufe. Ihre Ergebnisse
  sind normale Segmentierungs-Results.
keywords:
  - promptbasierte segmentierung
  - interaktive segmentierung
  - segment anything python
  - punkt prompt
  - box prompt
  - SAM python
  - maske aus klick
last_verified: 1.5.0
snippets:
  predict:
    - label: Punkt- und Box-Prompts
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # Ein Punkt ist [x, y] in Pixeln; Labels sind 1 positiv, 0 negativ.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # Polygone
        print(result.boxes.xyxy)    # aus den Masken abgeleitete enge Boxen

        # Ein Box-Prompt gibt eine Maske pro Box zurück.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: 'Einmal encodieren, mehrfach prompten'
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # set_image führt den aufwendigen Bild-Encoder einmal aus und speichert
        ihn.

        model.set_image(SAMPLE_IMAGE)

        first = model.predict(points=[640, 420], labels=[1])

        second = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
    - label: Alles segmentieren
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # Ohne Prompt wird ein Punktraster über das ganze Bild gelegt. Das
        # Standardraster mit 32 je Seite benötigt etwa 1024 Decoder-Durchläufe.
        result = model.predict(SAMPLE_IMAGE, points_per_side=8)
        print(len(result.masks))
    - label: Mehrdeutigkeitsmasken
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # Ein Punkt kann Ärmel, Hemd oder Person bedeuten. multimask=True gibt
        # alle drei Ganzes-gegen-Teil-Masken statt nur der besten zurück.
        result = model.predict(
            SAMPLE_IMAGE, points=[640, 420], labels=[1], multimask=True
        )
        print(len(result.masks))
source_hash: bb70ff24e6c0a767
---

## Definition

Die promptbasierte Segmentierung nimmt ein Bild und einen räumlichen Prompt
entgegen und gibt die Maske des Objekts zurück, auf das der Prompt zeigt. Es
findet keine Klassifikation statt: Es gibt keine Klassenliste, und
`result.boxes` enthält enge, aus den Masken abgeleitete Boxen statt
eigenständiger Erkennungen. `result.masks` enthält die Maskendaten,
`result.masks.xy` ihre Polygone.

Der Prompt bildet die Schnittstelle. `points` enthält `[x, y]`-Pixelkoordinaten
mit einem Satz pro Objekt. `labels` markiert jeden Punkt als positiv (1,
einschließen) oder negativ (0, ausschließen). `bboxes` verwendet
`[x1, y1, x2, y2]` mit einer Maske pro Box. Punkte und Boxen können kombiniert
werden. In diesem Fall werden sie objektweise gepaart und müssen dieselbe Länge
besitzen. Wenn jeder Prompt fehlt, wird der Alles-segmentieren-Pfad mit einem
Punktraster über das Bild ausgeführt.

Ein einzelner Punkt ist grundsätzlich mehrdeutig. Ein Klick auf einen Ärmel
kann den Ärmel, das Hemd oder die Person meinen. Daher gibt `multimask=True`
für jeden Prompt alle drei Ganzes-gegen-Teil-Masken zurück, anstatt nur die
beste. `conf` filtert nach der vorhergesagten IoU des Modells. Dies ist ein
Qualitätswert für Masken und keine Erkennungs-Confidence.

LibreYOLO besitzt keinen Aufgabenschlüssel `promptable`. Die Stufe registriert
sich als `segment`, also mit demselben Schlüssel wie die Instanzsegmentierung.
Sie unterscheidet sich durch die Aufrufform und besitzt daher eine eigene
Factory namens `LibreSAM()`, neben `LibreYOLO()`, `LibreOpenVocab()` und
`LibreVLM()`. Eine einzelne Signatur `predict(image)` kann die Schleife nicht
ausdrücken, für die diese Modelle entwickelt wurden: `set_image()` führt den
Bild-Encoder einmal aus und speichert die Embeddings. Jeder spätere Aufruf von
`predict()` mit `source=None` muss nur noch den Prompt decodieren.
`reset_image()` löscht den Cache. Der Bild-Encoder verursacht den größten
Rechenaufwand und läuft einmal pro Bild. Ein zweiter Prompt für dasselbe Bild
überspringt ihn vollständig.

## Modelle

Sechs Familien werden über Aliasse durch `LibreSAM` geladen.

[SAM](/docs/models/sam) ist der Standard und in den Größen `base`, `large` und
`huge` verfügbar, die auch als `b`, `l` und `h` geschrieben werden.

[SAM 2](/docs/models/sam-2) ist als `sam2-tiny`, `sam2-small`,
`sam2-base-plus` und `sam2-large` verfügbar. LibreYOLO unterstützt seinen
Bildpfad.

[SAM 3](/docs/models/sam-3) ist als `sam3` die einzige Familie, die einen
Textkonzept-Prompt akzeptiert: `text="yellow school bus"` gibt jede passende
Instanz zurück. Wird `text=` an eine andere Familie übergeben, löst sie eine
Fehlermeldung mit einem Verweis auf SAM 3 aus. Seine Gewichte stammen von Meta
und stehen unter der benutzerdefinierten SAM License statt der MIT-Lizenz von
LibreYOLO. Das Repository ist zugangsbeschränkt: Akzeptiere vor dem ersten
Download die Bedingungen auf der Modellseite und authentifiziere dich mit
`hf auth login`. Lies vor dem Deployment die Seite zu
[SAM 3](/docs/models/sam-3).

[EdgeTAM](/docs/models/edgetam) wird als `edgetam` geladen und ist eine
On-Device-Variante von SAM 2. LibreYOLO unterstützt seinen Bildpfad.

[MobileSAM](/docs/models/mobilesam) wird als `mobilesam` geladen und ersetzt
den ViT-H-Encoder von SAM durch einen destillierten TinyViT-Encoder.

[PicoSAM3](/docs/models/picosam3) wird als `picosam3` geladen und ist ein
kompaktes CNN für durch Box-Prompts bezeichnete Regionen auf Edge-Sensoren.
Box-Prompts bilden hier den gesamten Vertrag. Punkte, Text, Masken, Multimask
und Alles-segmentieren lösen jeweils eine Fehlermeldung mit einem Verweis auf
SAM 2 oder SAM 3 aus.

Das Extra dieser Stufe deckt die vier über `transformers` geladenen Familien ab:

```bash
pip install "libreyolo[sam]"
```

MobileSAM und PicoSAM3 sind native LibreYOLO-Portierungen und benötigen für die
Ausführung keine Installation von `transformers`.

## Vorhersage

<code-tabs name="predict" />

`source` und `set_image()` sind Alternativen und keine Abfolge. Übergib für
einen einmaligen Aufruf ein Bild an `predict()`, oder rufe zuerst `set_image()`
und anschließend für jeden Prompt `predict(source=None)` auf. Wenn du
`device=` an `predict()` übergibst, wird das Modell für diesen und jeden
späteren Aufruf verschoben. Bereits zwischengespeicherte Embeddings werden
dabei ungültig.

Alles-segmentieren ist der rechenintensive Modus. Der Standardwert von
`points_per_side` ist 32, was ungefähr 1024 Decoder-Durchläufen über das Bild
entspricht. Verringere den Wert für interaktive CPU-Anwendungen. Wenn `conf`
in diesem Modus nicht gesetzt ist, wird der Rasterschwellenwert der Familie
verwendet. Im promptbasierten Pfad bleiben ohne `conf` dagegen alle Masken
erhalten. Mit `conf=0.0` deaktivierst du die Filterung in beiden Modi. `max_det`
begrenzt die Anzahl der zurückgegebenen Masken.

Masken-Prompts werden in dieser Version nicht unterstützt. `masks=` löst einen
Fehler aus, statt ignoriert zu werden. Auch `track()` löst in der gesamten
Stufe einen Fehler aus. Dies sind Bildsegmentierer, führe daher für jeden Frame
`predict()` aus. Unter [Vorhersage](/docs/predict) findest du Informationen zu
Quellen und Ergebnisverarbeitung.

## Training

Keine Familie dieser Stufe wird innerhalb von LibreYOLO trainiert. `train()`
löst einen Fehler aus. Führe das Fine-Tuning im Upstream-Projekt durch und lade
die resultierenden Gewichte.

## Validierung

Für diese Stufe gibt es keinen Validator und `val()` löst einen Fehler aus.
Eine promptbasierte Maske besitzt keinen festen Klassensatz, auf den sich die
üblichen Erkennungs- und Segmentierungsmetriken beziehen könnten. Um eine
promptbasierte Maske zu bewerten, musst du sie selbst anhand der für dich
relevanten Prompts mit einer bereitgestellten Referenzmaske vergleichen.

## Export

Der Export liegt für die Stufe insgesamt außerhalb des Funktionsumfangs und
`export()` löst einen Fehler aus. Es gibt eine Ausnahme:
[PicoSAM3](/docs/models/picosam3) exportiert sein rohes 96x96-Regions-CNN als
`roi_image -> mask_logits` nach ONNX. Das Zuschneiden der Box und die
Rückskalierung der Maske auf die Bildkoordinaten verbleiben in Python. Alle
anderen Familien laufen über `predict()` in PyTorch. Unter
[Export](/docs/export) findest du die an anderer Stelle der Bibliothek
verfügbaren Formate.

