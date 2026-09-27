---
title: Ensemble-API
seo_title: LibreEnsemble-API und Fusionsoperationen
description: >-
  LibreEnsemble, ExternalDetector und die drei Fusionsoperationen in
  libreyolo.ops: Weighted Boxes Fusion, ihre Seed-Variante und klassenbewusste
  NMS-Fusion.
lead: >-
  LibreEnsemble führt mehrere Detektoren mit demselben Bild aus und fusioniert
  ihre Erkennungen zu einem Results-Objekt. Die Fusion erfolgt nach der eigenen
  Nachverarbeitung jedes Mitglieds. Mitglieder behalten daher Eingabegröße,
  Normalisierung und Unterdrückung.
keywords:
  - LibreEnsemble
  - weighted boxes fusion
  - wbf
  - ExternalDetector
  - libreyolo.ops.fusion
  - min_votes konsens
last_verified: 1.5.0
verification: >-
  Signaturen und Standardwerte aus libreyolo/ensemble/model.py und
  libreyolo/ops/fusion.py für v1.5.0. Designabsicht aus
  docs/adr/0004-model-ensembling.md.
snippets:
  usage:
    - label: 'Zwei Mitglieder, Standardfusion'
      language: python
      code: >
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE


        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])


        # Eine einzelne Bildquelle gibt ein Results-Objekt und keine Liste
        zurück.

        result = ens(SAMPLE_IMAGE, conf=0.25)


        print(result.boxes.xyxy)

        print(result.speed)
    - label: Konsens und Schwellenwerte pro Mitglied
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(
            ["LibreYOLO9t.pt", "LibreYOLO9s.pt"],
            weights=[1.0, 2.0],
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,
        )
        result = ens(SAMPLE_IMAGE, conf=[0.25, 0.4])
        print(len(result))
  ops:
    - label: Fusionsoperation ohne Modell
      language: python
      code: >
        import torch

        from libreyolo.ops import weighted_boxes_fusion


        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [12.0, 11.0, 51.0,
        49.0]])

        scores = torch.tensor([0.9, 0.8])

        labels = torch.tensor([0, 0])

        model_ids = torch.tensor([0, 1])


        fused = weighted_boxes_fusion(
            boxes, scores, labels, model_ids, num_models=2, iou_thr=0.55
        )

        print(fused)
source_hash: 3834f628efb1193d
---

## LibreEnsemble

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

| Argument | Standardwert | Bedeutung |
|---|---|---|
| `members` | | Zwei oder mehr Detektoren |
| `weights` | `None` | Vertrauensfaktoren pro Mitglied. Ohne Angabe jeweils `1.0` |
| `fusion` | `"wbf"` | `"wbf"`, `"wbf_seeded"`, `"nms"` oder ein Callable |
| `fusion_iou` | `0.55` | IoU-Schwellenwert für das Fusion-Clustering |
| `min_votes` | `1` | Nur Boxen behalten, die von mindestens so vielen Mitgliedern bestätigt wurden |

Ein Mitglied ist ein über die Factory `LibreYOLO()` aufgelöster Gewichtspfad,
ein bereits erstelltes Modell, ein exportiertes Backend oder ein
`ExternalDetector`. Jedes Mitglied muss ein Modell für die detect-Aufgabe sein.

<code-tabs name="usage" />

Die Konstruktion lehnt weniger als zwei Mitglieder, eine `weights`-Liste mit
falscher Länge, ein nicht positives Gewicht, einen nicht positiven ganzzahligen
Wert für `min_votes` und einen Wert über der Mitgliederanzahl ab. Auch
`fusion="nms"` zusammen mit `min_votes > 1` löst einen Fehler aus, weil NMS die
Clusterzugehörigkeit verwirft und keine Stimmen zählen kann.

`weights` skaliert das Vertrauen in jedes Mitglied. Ein höheres Gewicht zieht
fusionierte Koordinaten und Scores in Richtung dieses Mitglieds. Üblicherweise
werden die Werte proportional zur Validierungs-mAP gewählt.

## Klassenräume

Mitglieder mit identischen `names` werden direkt verarbeitet. Andernfalls
werden die Klassenräume anhand ihrer Namen vereinigt, die Klassen-IDs der
Mitglieder über Lookup-Tabellen neu zugeordnet und das fusionierte
`Results.names` auf die Vereinigung gesetzt. Die Fusion vereint Boxen nur
innerhalb derselben vereinheitlichten Klasse. Eine Klasse, die nur ein Mitglied
kennt, wird daher unfusioniert weitergegeben. Eine Abweichung löst bei der
Konstruktion eine Warnung aus.

`min_votes` wird pro Klasse auf die Anzahl der Mitglieder begrenzt, deren
Labelraum diese Klasse enthält. Dadurch bleibt der Konsens bei teilweise
gemeinsamen Vokabularen sinnvoll.

## Aufrufen des Ensembles

```python
ens(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    output_path=None,
    color_format="auto",
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    **kwargs,
)
```

`predict` ist ein Alias für `__call__`. Die Rückgabe ist das gewöhnliche
`Results`. Sein Feld `speed` schlüsselt den Aufwand pro Mitglied auf und
ergänzt einen Eintrag `fusion`. Eine einzelne Bildquelle gibt ein solches
Objekt zurück, eine Liste oder ein Verzeichnis eine Liste. `stream=True` gibt
einen Generator zurück.

`conf`, `iou` und `device` werden an jedes Mitglied verteilt und akzeptieren
auch einen Wert pro Mitglied. Mit `conf=[0.25, 0.4]` erhält Mitglied 0 den
Schwellenwert 0.25 und Mitglied 1 den Schwellenwert 0.4. `imgsz` wird als
Ganzzahl oder Tupel an alle Mitglieder verteilt und nur bei einer Liste pro
Mitglied interpretiert. `imgsz=(480, 640)` ist daher eine rechteckige Größe für
alle, während `imgsz=[480, 640]` Mitglied 0 die Größe 480 und Mitglied 1 die
Größe 640 gibt. Jeder Eintrag muss für die Familie des Mitglieds gültig sein.

`augment` wird an Mitglieder verteilt, die Test-Time-Augmentierung unterstützen.
Exportierte Backends ignorieren es. `classes` verwendet Klassen-IDs der
Vereinigung, und `max_det` gilt für das fusionierte Ergebnis. Die Mitglieder
laufen daher großzügig, und das Ensemble beschränkt nur einmal. `batch` wird
aus Gründen der API-Parität akzeptiert. Bilder werden sequenziell verarbeitet.

`val()` und `export()` lösen `NotImplementedError` aus. Validiere und
exportiere die Mitglieder einzeln.

## ExternalDetector

```python
ExternalDetector(fn: Callable, names: dict[int, str])
```

Passt ein beliebiges Erkennungs-Callable als Mitglied an. `fn` nimmt ein
PIL-Bild entgegen und gibt `(boxes, scores, labels)` zurück. Die Boxen sind
xyxy-Koordinaten in Pixeln des Originalbilds. Labels sind in `names` gültige
Klassen-IDs. Tensoren, Arrays und verschachtelte Listen funktionieren.
LibreYOLO importiert nichts aus dem externen Code.

Der Adapter validiert die Rückgabe. Sie muss ein 3-Tupel sein, Boxen müssen die
Form `(N, 4)` besitzen, die drei Arrays dieselbe Länge haben und jede Klassen-ID
in `names` vorkommen. Erkennungen mit einem Wert kleiner oder gleich `conf`
werden vor der Fusion verworfen.

## Fusionsoperationen

Die Fusionsprimitiven sind eigenständige torch-Operationen in `libreyolo.ops`.
Sie sind modellunabhängig und separat importierbar, weshalb sie getrennt vom
Ensemble exportiert werden.

<code-tabs name="ops" />

Alle drei nehmen dieselben Positionsargumente `boxes, scores, labels,
model_ids` entgegen und geben `(boxes, scores, labels)` zurück.

| Operation | Registry-Schlüssel | Verhalten |
|---|---|---|
| `weighted_boxes_fusion` | `wbf` | Sequenzielle, dem Paper entsprechende Weighted Boxes Fusion |
| `wbf_seeded` | `wbf_seeded` | Parallele One-Pass-Variante derselben Reduktion |
| `nms_fusion` | `nms` | Alles verketten und klassenbewusste NMS anwenden |

`FUSIONS` ordnet die drei Registry-Schlüssel den Callables zu.
`LibreEnsemble` schlägt `fusion=` dort nach.

```python
weighted_boxes_fusion(
    boxes, scores, labels, model_ids,
    *,
    weights=None,
    num_models=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type="avg",
    min_votes=1,
    models_per_label=None,
    label_weights=None,
)
```

`wbf_seeded` besitzt dieselbe Signatur. `nms_fusion` nimmt dieselben Argumente
außer `conf_type` entgegen und löst bei `min_votes > 1` einen `ValueError` aus.

In `weighted_boxes_fusion` werden Erkennungen in absteigender Reihenfolge ihrer
gewichtsskalierten Confidence verarbeitet. Jede wird entweder dem bestehenden
Cluster derselben Klasse zugeordnet, dessen laufende fusionierte Box sie bei
einer IoU über `iou_thr` am stärksten überlappt, oder beginnt einen neuen
Cluster. Die fusionierte Box eines Clusters ist der nach Confidence gewichtete
Mittelwert der Koordinaten seiner Mitglieder. Ihr Score ist der gewichtete
Mittelwert oder das Maximum ihrer Confidence-Werte und wird so skaliert, dass
von weniger Modellen bestätigte Boxen einen niedrigeren Score erhalten.

`wbf_seeded` wählt Cluster-Seeds mit klassenbewusster NMS bei `iou_thr`, ordnet
jede Erkennung dem Seed derselben Klasse mit der höchsten IoU zu und reduziert
jeden Cluster anschließend auf dieselbe Weise. Clusterformen verändern sich im
Durchlauf nicht. Die gesamte Operation ist daher Tensorarithmetik mit fester
Form. Beide Varianten stimmen bei eindeutigen Clustern überein und können sich
bei überlappenden Clusterketten leicht unterscheiden.

`nms_fusion` behält die Box mit der höchsten Confidence in jeder
Überlappungsgruppe unverändert. `weights` pro Modell skaliert Confidence-Werte
nur für die Rangfolge der Unterdrückung. Überlebende Boxen behalten ihre
ursprünglichen Scores.

## Benutzerdefinierte Fusion

`fusion=` akzeptiert auch ein Callable mit derselben Signatur wie die oben
genannten Operationen. Sein Name wird in `ens.fusion` gespeichert, oder
`"custom"`, wenn es keinen besitzt. Die Rückgabe wird validiert. Sie muss ein
Tripel `(boxes, scores, labels)` mit konsistenten Formen sein.

