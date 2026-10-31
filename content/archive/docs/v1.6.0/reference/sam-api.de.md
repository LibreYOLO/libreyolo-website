---
title: API für promptbasierte Segmentierung
seo_title: 'LibreSAM-API: Prompts, Aliasse und Signaturen'
description: >-
  Die LibreSAM-Factory, ihre Größenaliasse, Punkt-, Box- und
  Konzepttext-Prompts, der Einmal-encodieren-Lebenszyklus mit set_image und die
  nicht unterstützten Funktionen der Stufe.
lead: >-
  LibreSAM ist die Factory für promptbasierte Segmentierung. Ein Forward Pass
  benötigt einen beim Aufruf bereitgestellten Prompt pro Bild. Die Stufe besitzt
  daher eine eigene Vorhersageschnittstelle, statt durch den promptlosen
  Inferenz-Runner geleitet zu werden.
keywords:
  - LibreSAM
  - promptbasierte segmentierung
  - SAM punkt prompt
  - SAM box prompt
  - set_image
  - alles segmentieren
  - libreyolo sam extra
last_verified: 1.5.0
verification: >-
  Factory-Aliasse, Größen und Repositorys aus libreyolo/models/sam/model.py,
  sam2.py, edgetam.py, sam3.py, libreyolo/models/mobilesam/model.py und
  libreyolo/models/picosam3/model.py. Prompt-Vertrag und Standardwerte aus
  libreyolo/models/sam/base.py. Designabsicht aus
  docs/adr/0007-libresam-contract.md, jeweils für v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[sam]'
  usage:
    - label: Punkt- und Box-Prompts
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        r = model.predict(SAMPLE_IMAGE, points=[900, 370], labels=[1])
        print(r.masks.xy)
        print(r.boxes.xyxy)

        r = model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 200, 200])
        print(len(r))
    - label: 'Einmal encodieren, mehrfach prompten'
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")
        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[500, 375], labels=[1])
        b = model.predict(bboxes=[100, 100, 200, 200])
        print(len(a), len(b))

        model.reset_image()
source_hash: 18e8206c10ce17fd
---

## Installation

Die Stufe benötigt das Extra `sam`.

<code-tabs name="install" />

## Die Factory

```python
LibreSAM(model: str = "base", **kwargs) -> LibreSAMModel
```

`model` ist ein Größenalias und kein Pfad. `**kwargs` erreicht den Konstruktor
der Familie, der `device` und `multimask` entgegennimmt. Ein unbekannter Alias
löst einen `ValueError` aus, dessen Meldung alle bekannten Aliasse aufführt.

<code-tabs name="usage" />

## Aliasse

| Familie | Aliasse | Größen | Gewichte |
|---|---|---|---|
| SAM-1 | `base`, `large`, `huge`, `b`, `l`, `h`, `sam-base`, `sam-large`, `sam-huge`, `sam_b`, `sam_l`, `sam_h` | `base`, `large`, `huge` | `facebook/sam-vit-base`, `-large`, `-huge` |
| SAM-2 | `sam2-tiny`, `sam2-small`, `sam2-base-plus`, `sam2-baseplus`, `sam2-large` sowie die Kurzformen `sam2-t`, `sam2-s`, `sam2-bp`, `sam2-l`, `sam2_t`, `sam2_s`, `sam2_bp`, `sam2_l` | `tiny`, `small`, `base-plus`, `large` | `LibreYOLO/LibreSAM2tiny`, `-small`, `-base-plus`, `-large` |
| EdgeTAM | `edgetam`, `edge-tam`, `edgetam-edge` | `edge` | `LibreYOLO/LibreEdgeTAM` |
| SAM 3 | `sam3`, `sam-3`, `sam3-large` | `large` | `facebook/sam3` |
| MobileSAM | `mobilesam`, `mobilesam-tiny`, `mobilesam_t`, `mobile-sam`, `mobile-sam-tiny` | `tiny` | `LibreYOLO/LibreMobileSAM` |
| PicoSAM3 | `picosam3`, `picosam3-pico`, `picosam3_pico`, `pico-sam3` | `pico` | `LibreYOLO/LibrePicoSAM3` |

Der Standardwert ist `base`. SAM-1, SAM-2, EdgeTAM und MobileSAM verwenden
eine nominelle Canvas mit 1024 Pixeln, SAM 3 mit 1008 und PicoSAM3 mit 96.

Die Gewichte von SAM 3 sind zugangsbeschränkt. Sie werden unter Metas
benutzerdefinierter SAM License von `facebook/sam3` heruntergeladen. Diese ist
weder MIT noch Apache-2.0. LibreYOLO verteilt die Gewichte nicht. Akzeptiere vor
dem Laden die Bedingungen auf der Repository-Seite und authentifiziere dich bei
Hugging Face. Der Loader protokolliert zuerst den Hinweis.

Auch die Familienklassen werden exportiert. `LibreSAM1`, `LibreSAM2`,
`LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM` und `LibrePicoSAM3` können daher
direkt mit `size=` erstellt werden.

## predict

```python
model.predict(
    source=None,
    *,
    points=None,
    bboxes=None,
    labels=None,
    masks=None,
    text=None,
    conf=None,
    multimask=None,
    max_det=300,
    device=None,
    color_format="auto",
    points_per_side=None,
) -> Results
```

| Argument | Standardwert | Bedeutung |
|---|---|---|
| `source` | `None` | Zu segmentierendes Bild. `None` verwendet das von `set_image()` gespeicherte Bild erneut |
| `points` | `None` | Punkt-Prompt in Pixelkoordinaten |
| `bboxes` | `None` | Box-Prompt als `[x1, y1, x2, y2]` oder eine Liste für eine Maske pro Box |
| `labels` | `None` | Punktlabels, `1` positiv und `0` negativ, passend zur Form von `points`. Ohne Angabe sind alle positiv |
| `masks` | `None` | Reserviert. Die Übergabe löst `NotImplementedError` aus |
| `text` | `None` | Konzept-Prompt, nur SAM 3 |
| `conf` | `None` | Untergrenze für die vorhergesagte Masken-IoU |
| `multimask` | `None` | Alle Mehrdeutigkeitsmasken pro Prompt zurückgeben. Verwendet standardmäßig die Konstruktionseinstellung |
| `max_det` | `300` | Obergrenze für zurückgegebene Masken |
| `device` | `None` | Modell für diesen und spätere Aufrufe verschieben. Gespeicherte Embeddings werden ungültig |
| `color_format` | `"auto"` | Hinweis zum Farbformat von Arrays im Speicher |
| `points_per_side` | `None` | Rasterdichte für Alles-segmentieren, Standardwert 32 |

Die Rückgabe ist ein gewöhnliches `Results`-Objekt mit `masks` und daraus
abgeleiteten engen `boxes`. Klasse `0` heißt `"object"`.

## Prompt-Formen

`points` akzeptiert die verschachtelten Formen `[x, y]` für ein Objekt,
`[[x, y], ...]` für N Objekte und `[[[x, y], ...], ...]` für nach Objekt
gruppierte Punkte. Numpy-Arrays funktionieren überall dort, wo eine Liste
akzeptiert wird. Die Koordinaten sind einfache Pixelwerte im Quellbild.

Wenn jeder räumliche Prompt fehlt, wird Alles-segmentieren ausgeführt. Dieser
automatische Maskengenerator verwendet ein Punktraster, einen Schwellenwert für
die vorhergesagte IoU und eine Deduplizierung anhand der Boxen-IoU. Der
Standardwert 32 für `points_per_side` führt ungefähr 1024 Decoder-Durchläufe
aus und ist auf der CPU langsam. Verringere ihn für interaktive Anwendungen.
Der Generator verwendet keine Stability-Score-Filterung, Multi-Crop oder
Masken-IoU-Deduplizierung. Er ist daher eine Annäherung an den promptbasierten
Pfad und kein identischer Ersatz.

## Confidence

`conf` filtert nach der vorhergesagten Masken-IoU. Dies ist ein Qualitätswert
für Masken und keine Erkennungs-Confidence. Im promptbasierten Pfad behält
`None` jede Maske, bei Alles-segmentieren gilt damit der Rasterschwellenwert
der Familie. `0.0` deaktiviert die Filterung in beiden Modi.

Im Textpfad von SAM 3 ist `conf` stattdessen der Erkennungs-Score der
Promptable Concept Segmentation. `None` steht dort für den normalen
Schwellenwert 0.3. `0.0` behält alle Kandidaten.

## Text-Prompts

`text=` wird nur von SAM 3 unterstützt. Jede Familie mit räumlichen Prompts
löst dafür `NotImplementedError` aus. Text schließt Punkte und Boxen gegenseitig
aus. Im zurückgegebenen `names` wird Klasse `0` dem angeforderten Konzept
zugeordnet. Ein Textaufruf mit `source=None` encodiert das gespeicherte Bild
erneut, weil Tracker und Konzept-Encoder keinen Cache teilen.

Das Keyword `exemplars=` ist für eine zukünftige Erweiterung mit
Bildexemplaren reserviert und nicht implementiert.

## Lebenszyklus für einmalige Encodierung

```python
model.set_image(source, color_format="auto") -> LibreSAMModel
model.reset_image() -> LibreSAMModel
```

`set_image` führt den aufwendigen Bild-Encoder einmal aus und speichert die
Embeddings. Jeder spätere Aufruf von `predict()` mit `source=None` ist dadurch
günstig. Beide Methoden geben das Modell zurück, damit Aufrufe verkettet werden
können. Die Übergabe von `device=` an `predict` verschiebt das Modell und macht
den Cache ungültig.

## PicoSAM3

PicoSAM3 akzeptiert nur `bboxes=`. Punkt-, Text-, Masken-, Multimask- und
Alles-segmentieren-Prompts lösen einen Fehler aus. Die Box wird um 10 %
vergrößert und durch ein 96-Pixel-ROI-Netz geleitet. PicoSAM3 ist die einzige
Familie dieser Stufe, die exportiert werden kann, und unterstützt ausschließlich
ONNX.

## Nicht unterstützte Funktionen

`train()`, `val()` und `track()` lösen bei jeder Familie der Stufe
`NotImplementedError` aus. Promptbasierte Masken besitzen keinen festen
Klassensatz, anhand dessen eine mAP berechnet werden könnte. `export()` löst
bei SAM-1, SAM-2, SAM 3, EdgeTAM und MobileSAM einen Fehler aus.

Video- und Speicherpfade von SAM-2, SAM 3 und EdgeTAM liegen in dieser Version
ebenso außerhalb des Funktionsumfangs wie Bildexemplare für SAM 3 und
Masken-Prompts.

