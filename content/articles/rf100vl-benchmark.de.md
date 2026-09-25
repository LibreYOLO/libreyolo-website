---
title: "RF100-VL Benchmark: Wie gut generalisieren LibreYOLO-Modelle?"
description: RF100-VL-Ergebnisse dazu, wie YOLOv9, YOLOX, YOLO-NAS, EdgeCrafter und RF-DETR nach dem Fine-Tuning auf 100 realen Datensätzen generalisieren.
date: 2026-09-05
author: Xuban
layout: paper
tags: [LibreYOLO, RF100-VL, benchmark, object-detection, roboflow]
---

RF100-VL untersucht, wie gut sich ein Detektor auf andere Daten als COCO anpassen lässt. Jedes Modell wird separat auf 100 sehr unterschiedlichen Datensätzen nachtrainiert, darunter Infrarotbilder, Röntgenaufnahmen, Mikroskopie, Sport, Funkfrequenzbilder, kleine Objekte und Videospiele. Wir haben 17 LibreYOLO-Konfigurationen getestet: **insgesamt 1700 Fine-Tuning-Läufe.**

<div>
<rf100vl-explorer></rf100vl-explorer>
</div>

## Ergebnisse

<div>
<rf100vl-results-chart></rf100vl-results-chart>
</div>

RF-DETR-L führte diesen Vergleich mit **61.76** an. M folgte mit 61.13, danach S mit 60.41. Die vier RF-DETR-Ergebnisse liegen nahe an den [veröffentlichten Zahlen von Roboflow](https://rfdetr.roboflow.com/latest/learn/benchmarks/). Das ist eine nützliche Bestätigung für das RF-DETR-Training in LibreYOLO.

Die Größe sagte nicht jedes Ergebnis voraus. YOLO-NAS-S und M liegen mit 58.00 und 57.99 praktisch gleichauf. EdgeCrafter-M erreicht 57.93 und liegt damit vor S mit 55.99 und L mit 56.11. Diese Rückschritte wollen wir untersuchen. Dieser Durchlauf ist kein kontrolliertes Skalierungsexperiment: Auflösung, vortrainierte Gewichte, Präzision und Trainingsrezepte unterscheiden sich.

RF-DETR-S mit Backbone-LoRA erreicht 57.19, gegenüber 60.41 beim vollständigen Fine-Tuning. Auf 95 Datensätzen gewinnt das vollständige Fine-Tuning. Die aufgezeichnete Medianzeit ist nur sieben Minuten länger, während die aufsummierte Jobzeit nahezu gleich bleibt.

Die YOLOv9-Ergebnisse stammen aus einer Zeit vor wichtigen Korrekturen am Training. Das auffällige Ergebnis für M half dabei, fehlendes PGI, eine andere Letterbox-Konvention und eine Trainingsgrenze von 100 Labels aufzudecken. Die archivierten Zahlen beschreiben den ausgeführten Code. Sie sind kein Urteil über die YOLOv9-Architektur.

Wähle unten ein Modell aus, um seine Werte auf allen 100 Datensätzen anzusehen.

<div>
<rf100vl-results-detail></rf100vl-results-detail>
</div>

## Methodik

Für jeden Datensatz trainierten wir mit `train`, wählten den Checkpoint mit dem besten Validierungswert für AP50:95 aus und werteten ihn einmal auf `test` aus. Der Hauptwert ist der ungewichtete Mittelwert dieser 100 Testergebnisse.

| Einstellung | Regel für den Durchlauf |
|---|---|
| Training | 100 Epochen; kein Early Stopping |
| Effektiver Batch | 16 |
| Seed | 0; ein abgeschlossener Lauf pro Datensatz |
| Checkpoint | Beste Validierungs-AP50:95 mit EMA-Gewichten |
| Testauswertung | pycocotools mit `maxDets=500` |
| Tuning | Ein festgelegtes Trainingsrezept pro Familie; kein Tuning pro Datensatz |

Jede Familie behielt ihr eigenes Trainingsrezept:

| Familie | Auflösung | Präzision | Überblick über die Augmentierung |
|---|---:|---:|---|
| YOLOv9 T/S/M | 640 | FP32 | Mosaic, HSV, Flip; starke Augmentierungen in den letzten 15 Epochen deaktiviert |
| YOLOX Nano/Tiny | 416 | FP32 | Mosaic, Mixup, HSV, affine Transformation, Flip; Auslaufphase über 15 Epochen |
| YOLOX S/M | 640 | FP32 | Dasselbe Familienrezept mit einer größenabhängigen Lernrate |
| YOLO-NAS S/M | 640 | FP32 | Mixup, HSV und Flip; Mosaic deaktiviert |
| EdgeCrafter S/M/L | 640 | FP16 | Flip; LibreYOLO-Standardwerte der Familie |
| RF-DETR N/S/M/L und S LoRA | 384/512/576/704; LoRA bei 512 | BF16 | Multi-Scale, Crop/Resize und Flip |

Das sind Ergebnisse mit einem einzelnen Seed. Kleine Unterschiede belegen keine Verbesserung. Für RF-DETR M und L wurde Gradient Accumulation verwendet, um in den verfügbaren Speicher zu passen; bei einigen dichten Datensätzen waren kleinere physische Batches nötig. Die öffentlichen RF-DETR-Trainingsrezepte starten außerdem mit COCO-Checkpoints, während Roboflow für seine historische Tabelle private Objects365-Checkpoints verwendete.

Die Trainingszeiten sind Aufzeichnungen aus dem Durchlauf, keine kontrollierten Geschwindigkeitsbenchmarks. Jobs teilten sich gelegentlich GPUs, wurden erneut versucht oder fortgesetzt. Das optionale T4-Protokoll zur Latenz eines einzelnen Artefakts haben wir nicht ausgeführt. YOLO-NAS verwendet separat lizenzierte [vortrainierte Gewichte von Deci für nicht kommerzielle Nutzung](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md).

## Was sich durch den Workload verbessert hat

Kleine Tests können belegen, dass das Training startet. Sie bilden nicht wochenlanges kontinuierliches Fine-Tuning über viele Architekturen und Datensätze hinweg ab. Bei diesem Umfang waren langsame Abläufe, Speicherdruck und Korrektheitsprobleme kaum zu übersehen.

- **Bildladen und Validierung.** Wir haben das deterministische Resize vor der Augmentierung zwischengespeichert, Validator-Worker und -Puffer wiederverwendet und, wo unterstützt, Validierungsdurchläufe als Graph dargestellt. [PR #677](https://github.com/LibreYOLO/libreyolo/pull/677), [PR #682](https://github.com/LibreYOLO/libreyolo/pull/682)
- **CUDA-Graphs im Training.** Die Unterstützung für das Capturing wurde von YOLOv9 und RF-DETR auf 24 Familien ausgeweitet. Außerdem haben wir einen Pin-Memory-Race im DataLoader und die Graph-Invalidierung bei Trainingsübergängen behoben. Ein YOLOv9-T-Test sank bei gleicher gemeldeter AP von 428.4 auf 367.7 Sekunden. [PR #671](https://github.com/LibreYOLO/libreyolo/pull/671), [PR #681](https://github.com/LibreYOLO/libreyolo/pull/681), [PR #716](https://github.com/LibreYOLO/libreyolo/pull/716)
- **COCO-Auswertung.** Die Auswertung auf dichten Datensätzen dauerte mitunter länger als das Training. Die Integration von faster-coco-eval wertete gespeicherte Vorhersagen aus allen 100 Test-Splits in 8.4 statt 131.4 Sekunden aus. Von 1,400 Metrikwerten waren 1,381 bitidentisch; die größte Abweichung betrug 2.22e-16, und die Haupt-AP blieb unverändert. [PR #708](https://github.com/LibreYOLO/libreyolo/pull/708)
- **RF-DETR und gemeinsame Trainingspfade.** Schnelleres L1-Matching, weniger Synchronisierungen mit dem Gerät, fusioniertes AdamW und begrenzter Speicherbedarf des Matchers reduzierten in einem aufgezeichneten Test einen RF-DETR-S-Schritt von 266 auf 234 ms. Dieselbe Untersuchung verbesserte fünf weitere Matcher, das YOLOv9-Logging und die Tensor-Wiederverwendung in EdgeCrafter. [PR #761](https://github.com/LibreYOLO/libreyolo/pull/761), [PR #762](https://github.com/LibreYOLO/libreyolo/pull/762), [PR #765](https://github.com/LibreYOLO/libreyolo/pull/765)
- **Attention.** Wir leiteten geeignete Attention über PyTorch SDPA, integrierten eine CUDA-Implementierung unter Apache-2.0 und korrigierten die Ausführung mit Mixed Precision. Außerdem schrieben wir für die Inferenz einen Triton-Kernel für deformierbare Attention direkt im Repository. In den dokumentierten Tests war er isoliert etwa viermal schneller und für RF-DETR-N Ende zu Ende etwa 7% schneller. [PR #712](https://github.com/LibreYOLO/libreyolo/pull/712), [PR #713](https://github.com/LibreYOLO/libreyolo/pull/713), [PR #760](https://github.com/LibreYOLO/libreyolo/pull/760), [PR #784](https://github.com/LibreYOLO/libreyolo/pull/784), [PR #790](https://github.com/LibreYOLO/libreyolo/pull/790)
- **Korrektheit des Trainings.** Wir korrigierten die BatchNorm-Rekonstruktion in YOLOX und stellten PGI, Letterbox-Metadaten, Label-Kapazität und Momentum-Warmup in YOLOv9 wieder her. Der Benchmark lieferte die Checkpoints und Fehlermuster, durch die beide Probleme sichtbar wurden. [PR #700](https://github.com/LibreYOLO/libreyolo/pull/700), [PR #796](https://github.com/LibreYOLO/libreyolo/pull/796)

Diese Zahlen stammen aus separaten Tests mit unterschiedlichen Workloads. Sie lassen sich nicht zu einer einzigen Geschwindigkeitssteigerung multiplizieren. LibreYOLO ist bei realen Trainingsworkloads jetzt schneller und zuverlässiger als vor diesem Durchlauf.

## Harness und Artefakte

Der öffentliche [Trainings-Harness](https://github.com/LibreYOLO/vision-analysis-benchmark/tree/rf100vl-harness) verteilt Datensätze über GPUs, setzt unterbrochene Jobs fort, verarbeitet OOM-Wiederherstellungen und erfasst Trainingsrezepte, Datensatzversionen, Logs, Checkpoints und Vorhersagen.

Der [run-rf100vl-benchmark-Skill](https://github.com/LibreYOLO/libreyolo/blob/release/skills/run-rf100vl-benchmark/SKILL.md) stellt KI-Coding-Agenten das Protokoll und die Betriebsanweisungen für Vast.ai bereit. Das [Ergebnisarchiv](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main) enthält jeden veröffentlichten Lauf.

## Danke, Roboflow

Joseph Nelson bot Unterstützung mit GPUs an, nachdem ich geschrieben hatte, dass ich gern über COCO hinaus benchmarken wollte, mir die Läufe aber nicht leisten konnte. Matvei Popov teilte die Referenzkonfigurationen, erklärte die Auswertungseinstellungen und verfolgte die Ergebnisse, sobald sie eintrafen.

Dank dieser Unterstützung konnten wir das LibreYOLO-Training über 17 Konfigurationen hinweg validieren, Belege veröffentlichen, die Menschen bei der Modellwahl helfen, den Harness freigeben und die Bibliothek so stark belasten, bis ihre Schwachstellen offensichtlich wurden.

Vielen Dank an Joseph, Matvei und das Roboflow-Team für die Rechenleistung, die Zeit und die Unterstützung.
