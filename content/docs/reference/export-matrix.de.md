---
title: Vollständige Exportmatrix
seo_title: LibreYOLO-Exportmatrix und ihre Regeln
description: >-
  Wie LibreYOLO entscheidet, ob eine Kombination aus Familie, Aufgabe und Format
  exportierbar ist: zwölf Formate, drei Stufen, Fallback-Regeln und
  Paritätsschwellenwerte.
lead: >-
  Die Exportunterstützung ist eine Abfrage des Tripels (Familie, Aufgabe,
  Format). Diese Seite beschreibt den Aufbau der Matrix, die Regeln für Zellen
  ohne ausdrücklichen Eintrag und die Abfrage einer gewünschten Kombination.
keywords:
  - LibreYOLO Export Unterstützung
  - Exportmatrix
  - ONNX TensorRT OpenVINO TFLite
  - libreyolo formats Befehl
  - Export Parität Schwellenwert
  - NotImplementedError Export
last_verified: 1.5.0
verification: >-
  Formate, Stufen, Fallback-Reihenfolge, Aufgaben- und Familienblöcke sowie
  NCNN-Blöcke aus libreyolo/export/support.py gelesen; Aliasse und gemeinsame
  Argumente aus libreyolo/export/exporter.py; Stufendefinitionen aus
  docs/adr/0011-export-support-tiers.md; Paritätsschwellenwerte aus
  docs/export_support.md, jeweils in v1.5.0. Einzelne Kombinationszellen werden
  hier nicht wiedergegeben; frage sie mit dem folgenden Snippet ab.
snippets:
  usage:
    - label: Matrix ohne Modell abfragen
      language: python
      code: |
        from libreyolo.export.support import (
            EXPORT_FORMATS,
            get_support,
            validated_alternatives,
        )

        print(EXPORT_FORMATS)

        entry = get_support("yolo9", "detect", "onnx")
        print(entry.tier, entry.since)
        print(entry.constraint)

        print(validated_alternatives("yolo9", "detect"))
    - label: CLI
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
        libreyolo formats --family yolo9 --task detect --json
  export:
    - label: Exportieren und eine Ablehnung auslesen
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.export.support import get_support


        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.export(format="onnx"))


        # Vor dem Aufruf prüfen: Eine gesperrte Kombination löst bei der
        Vorabprüfung

        # einen Fehler aus und die Meldung enthält diesen Grund.

        blocked = get_support("domedetr", "detect", "onnx")

        print(blocked.tier)

        print(blocked.reason)
source_hash: 83de3289634888c6
---

## Aufbau der Matrix

Die Matrix wird durch `(family, task, format)` indiziert. Familienschlüssel sind die kanonischen Namen aus der Modell-Registry, Aufgabenschlüssel stammen aus `libreyolo.tasks.TASKS`. Es gibt zwölf Formate:

`onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`, `rknn`, `ncnn`, `tflite`, `coreml`, `coreai`.

`model.export(format=...)` akzeptiert außerdem zwei Aliasse: `engine` für `tensorrt` und `litert` für `tflite`, den aktuellen Namen von TensorFlow Lite. Das Format und die Endung `.tflite` bleiben unverändert.

<code-tabs name="usage" />

Da eine Zelle von drei Schlüsseln abhängt, ist das vollständige Raster groß und ändert sich mit jeder Veröffentlichung. Es wird generiert statt manuell geschrieben und befindet sich im Bibliotheks-Repository unter `docs/export_support.md`. Frage die Matrix aus Python oder der CLI ab, statt eine Kopie zu lesen.

## Drei Stufen

| Stufe | Bedeutung |
|---|---|
| `validated` | Numerische Parität wird in CI oder einem dokumentierten nächtlichen Lauf geprüft |
| `available` | Konvertierung ist implementiert, aber es gibt keinen aufgezeichneten Nachweis numerischer Laufzeitparität |
| `blocked` | Die Vorabprüfung löst vor der Aufzeichnung `NotImplementedError` mit einem Grund aus |

Sowohl validierte als auch verfügbare Kombinationen werden ohne Bestätigung oder pauschale Warnung ausgeführt. Aufgezeichnete Nachweise und Einschränkungen bleiben in der generierten Dokumentation sichtbar. Eine gesperrte Kombination schlägt vor Abhängigkeitsprüfungen, dem Laden von Kalibrierungsdaten, der Aufzeichnung oder der Artefakterstellung fehl.

Ein validierter Eintrag benötigt einen Paritätstest und ein Feld `since`.

Ein `SupportEntry` enthält vier Felder: `tier`, einen String `reason`, die Veröffentlichung `since` und einen String `constraint`. Bei der Integration ist insbesondere die Einschränkung relevant. Ein Häkchen gilt nur unter den darin genannten Bedingungen. Typischerweise sind dies ein fester Eingabe-Canvas, Batchgröße 1, FP32 und eine bestimmte Laufzeitversion.

## Entscheidung über eine Zelle

`get_support(family, task, fmt)` wird in der folgenden Reihenfolge ausgewertet. Die erste passende Regel gewinnt.

1. Eine unbekannte Aufgabe oder ein Format außerhalb der zwölf Formate ergibt `blocked`.
2. Ein ausdrücklicher Eintrag `(family, task, format)` wird unverändert zurückgegeben.
3. Eine familienweite Sperre ergibt `blocked` mit dem Grund der Familie.
4. Eine aufgabenweite Sperre ergibt `blocked` mit dem Grund der Aufgabe.
5. Bei `ncnn` ergibt eine Familie auf der NCNN-Sperrliste `blocked`.
6. `mnn` ergibt `blocked`: Für diese Familie und Aufgabe gibt es keinen Laufzeitvertrag.
7. `rknn` ergibt `blocked`. RKNN ist in dieser Version auf exakt die im Simulator geprüften Erkennungsvarianten beschränkt: YOLO9-t, YOLO9-E2E-t, YOLO-NAS-s und PicoDet-s auf RK3588.
8. `tensorrt` und `openvino` ergeben `available`: Der Konvertierungspfad ist vorhanden, aber die Laufzeitparität für diese Familie und Aufgabe wurde nicht aufgezeichnet.
9. `tflite`, `paddle`, `coreai` und `coreml` ergeben jeweils aus eigenem Grund `blocked`.
10. Alles andere ergibt `available`: Die Konvertierung ist implementiert, numerische Laufzeitparität aber nicht aufgezeichnet.

Die Asymmetrie in den Schritten 8 bis 10 ist beabsichtigt. TensorRT und OpenVINO konvertieren generisch aus ONNX, weshalb sich der Versuch einer nicht aufgeführten Kombination lohnt. TFLite, Paddle, Core AI und CoreML benötigen jeweils einen familienspezifischen Pfad. Eine nicht aufgeführte Kombination wird dort deshalb abgelehnt.

## Gesperrte Aufgaben

Diese Aufgaben sind bei jeder Familie ohne ausdrücklichen Eintrag gesperrt.

| Aufgabe | Grund |
|---|---|
| `ocr` | Zwei Netzwerke mit dynamischem Beschnitt je Region passen nicht in den Exportvertrag eines einzelnen Graphen |
| `point` | Die Familie ist nicht mit dem gemeinsamen Vertrag für Punkt-Heatmaps und Peak-Decodierung im Backend verbunden |
| `semantic` | Die Familie ist nicht mit dem gemeinsamen Vertrag für dichte Logits und Argmax im Backend verbunden |
| `mesh` | Graphausgaben, Metadaten und Laufzeitvertrag für Körper-Meshes sind nicht definiert |
| `normal` | Die Familie ist nicht mit dem Vertrag für dichte Einheitsnormalen auf festem Canvas und Renormalisierung im Backend verbunden |
| `panoptic` | Der panoptische Export besitzt keinen Backend-Laufzeitvertrag |
| `gaze` | Die Familie ist nicht mit dem gemeinsamen Vertrag für Logits aus zwei Köpfen und Erwartungswert-Decodierung im Backend verbunden |

Ein ausdrücklicher Eintrag überschreibt diese Regeln. So kann beispielsweise eine entsprechend verbundene semantische Familie trotzdem exportieren.

## Gesperrte Familien

| Familie | Gesperrt für |
|---|---|
| `depth_anything3` | Jedes Format; sein Tiefengraph ist nicht Bestandteil des Vertrags für exportierte Laufzeitumgebungen |
| `domedetr` | Jedes Format. PAQI legt die Abfrageanzahl je Bild fest, sodass ein aufgezeichneter Graph nur für sein Aufzeichnungsbild gültig ist. Verwende D-FINE als exportierbares DETR |
| `eomt` | Instanz- und panoptischen Export, für die keine Laufzeitanalyse vorhanden ist |
| `l2cs` | Alles außer ONNX, TorchScript, ExecuTorch, TensorRT und OpenVINO |
| `hrnet` | Alles außer ONNX, TorchScript, OpenVINO und TensorRT |
| `sam`, `sam2`, `sam3`, `edgetam`, `mobilesam` | Jedes Format; der Export promptbarer Modelle liegt außerhalb des v1-Laufzeitvertrags |
| `grounding_dino`, `owlv2`, `omdet_turbo`, `ov_deim` | Jedes Format; der Export für Open-Vocabulary-Laufzeitumgebungen liegt außerhalb von v1 |
| `florence2`, `kosmos2`, `lfm2vl`, `internvl3`, `qwen3vl`, `smolvlm2`, `locateanything` | Jedes Format; der Export generativer VLMs liegt außerhalb von v1 |

PicoSAM3 ist die Ausnahme in der promptbaren Stufe. Es exportiert sein rohes ROI-Netzwerk mit 96 px nach ONNX.

## NCNN-Sperren

DETR-artige Decoder benötigen Sampling-Operationen, die NCNN nicht implementiert. Die folgenden Familien sind daher für `ncnn` gesperrt, sofern kein ausdrücklicher Eintrag etwas anderes festlegt: Deformable DETR, DETR, DINO-DETR, D-FINE, LW-DETR, DEIM, DEIMv2, RT-DETR, RT-DETRv2, RT-DETRv4, RF-DETR und EC. Die Ablehnung nennt ONNX, OpenVINO, TorchScript und TensorRT als Alternativen.

## Paritätsschwellenwerte

Eine validierte Zelle bedeutet, dass das exportierte Artefakt das native Modell innerhalb der folgenden Grenzen reproduziert hat:

| Aufgabengruppe | Schwellenwert |
|---|---|
| Objekterkennung und OBB | IoU zugeordneter Boxen über 0,95, Bewertungs-MAE unter 0,01 |
| Segmentierung und Panoptik | Masken-IoU über 0,95 |
| Posenschätzung | Keypoint-L2 unter 2 Pixeln bei nativer Auflösung |
| Klassifizierung | Kosinus der Logits über 0,999 und gleiche Top-1-Klasse |
| Tiefe und Restauration | PSNR über 40 dB gegenüber der nativen Ausgabe |
| Oberflächennormalen | Mittlerer Winkelfehler unter 0,1 Grad |
| Punkt | Peak-Positionen innerhalb einer Ausgabezelle gleich |

DETR-Abfragezeilen bilden eine ungeordnete Menge. Bei der Paritätsprüfung der DETR-Familie werden sie daher als Menge statt nach ihrer Position zugeordnet.

## Export

<code-tabs name="export" />

Eine gesperrte Kombination löst bei der Vorabprüfung `NotImplementedError` aus. Die Meldung enthält den aufgezeichneten Grund. `validated_alternatives(family, task)` gibt die für dieses Paar validierten Formate zurück. Diese Liste lässt sich sinnvoll neben einer Ablehnung anzeigen.

Die gemeinsamen Argumente aller Exporter stehen auf der [Seite zur Modell-API](/docs/reference/model-api). Formatspezifische Argumente findest du auf den Seiten der jeweiligen Formate.

## Interpretation einer Einschränkung

Eine validierte Zelle ist eine Aussage über eine gemessene Konfiguration, nicht über das Format im Allgemeinen. Eine Einschränkung wie `FP32, batch 1, fixed 520x520 input` bedeutet, dass die Parität bei dieser Form und Genauigkeit aufgezeichnet wurde. Ein Export mit einer anderen Auflösung oder Batchgröße erzeugt weiterhin ein Artefakt. Es ist lediglich nicht die Konfiguration, auf der der Wert basiert.
