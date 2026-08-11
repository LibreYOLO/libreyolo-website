---
title: Trainingsperformance
seo_title: 'Schnelleres Training: CUDA Graphs, AMP und Profiler'
description: >-
  Beschleunige einen Trainingslauf: Erfasse den Schritt in CUDA Graphs, wähle
  einen AMP-Datentyp und finde mit dem integrierten Profiler den tatsächlichen
  Zeitverbrauch.
lead: >-
  Drei Stellschrauben beeinflussen die Dauer eines Trainingsschritts: Mixed
  Precision, die CUDA-Graph-Erfassung des Forward und Backward Passes des Netzes
  sowie die vom Profiler ermittelten tatsächlichen Engpässe.
keywords:
  - cuda graphs training
  - training beschleunigen
  - mixed precision training
  - bfloat16 training
  - pytorch profiler
  - dataloader engpass
  - kernel launch overhead
  - gpu-auslastung
last_verified: 1.5.0
snippets:
  profile:
    - label: Profilieren und Training fortsetzen
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Profiliert ein kurzes Fenster echter Schritte, gibt ein Urteil aus
        # und setzt den Lauf nach Entfernen der Hooks fort.
        model.train(data="my-dataset.yaml", epochs=100, profile=True)
    - label: Nur messen und danach beenden
      language: bash
      code: |
        # Setzt no_aug_epochs=0 und führt genug Epochen für das Fenster aus.
        libreyolo profile run coco128 --weights LibreYOLO9s.pt --size s
    - label: Ergebnis untersuchen
      language: bash
      code: |
        libreyolo profile summary runs/profile/prof/profile.json
        libreyolo profile phases runs/profile/prof/profile.json
        libreyolo profile kernels runs/profile/prof/profile.json --top 10
  graph:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 cuda_graph=true
  amp:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", amp=True, amp_dtype="bfloat16")
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          amp_dtype=bfloat16
source_hash: ee5bb727065b6099
---

## Messung vor jeder Änderung

Die drei nachfolgenden Stellschrauben lösen unterschiedliche Probleme. Eine
unpassende ändert nichts. Der Profiler zeigt, welches Problem vorliegt.

<code-tabs name="profile" />

`profile=True` misst ein Fenster echter Trainingsschritte. Standardmäßig werden
fünf verworfen und anschließend 20 gemessen. Danach gibt die Funktion einen
Bericht aus, schreibt ihre Artefakte und setzt das Training ohne Hooks fort.
Bei deaktivierter Option entstehen keine Kosten. Unter verteiltem Training wird
sie ignoriert.

Der Bericht endet mit einem von vier Urteilen:

| Urteil | Bedeutung | Stellschrauben |
|---|---|---|
| `dataloader` | GPU wartet auf Eingabedaten | mehr `workers`, `cache="ram"` oder `"disk"`, leichtere Augmentierung, größerer Batch |
| `host / launch` | GPU wird zu langsam versorgt, viele kleine Kernel | größerer Batch, CUDA Graphs, weniger Host-Synchronisierungen pro Schritt |
| `compute` | GPU ist ausgelastet | AMP oder bfloat16, oder Zustand akzeptieren |
| `memory-pressure` | Allocator-Thrashing, VRAM an der Grenze | kleinerer Batch. Auslastungswerte sind hier unzuverlässig |

Der Auslastungswert ist die Kernel-Busy-Time geteilt durch die nicht
synchronisierte Schrittdauer. Das Fenster wird bewusst geteilt. Die erste
Hälfte läuft ohne zusätzliche Synchronisierung, damit das Urteil die echte
Überlappung widerspiegelt. Nur die zweite Hälfte umschließt jede Phase mit
einer Synchronisierung, um GPU-Zeit zuzuordnen. Eine Synchronisierung jeder
Phase verschafft den Dataloader-Workern Spielraum und verdeckt Unterversorgung.
Die Zusammensetzungswerte werden daher nie zur Auswahl des Urteils verwendet.

Vier Dateien entstehen im Laufverzeichnis: `timeline.html`, das sich direkt in
einem Browser öffnet, `profile_trace.json` für Perfetto oder Nsight,
`profile_summary.json` und die eigenständige Datei `profile.json`, die du
kopieren und an die Unterbefehle von `libreyolo profile` übergeben kannst.

Zwei Eigenschaften von `profile run` sind wichtig. Der Befehl setzt
`no_aug_epochs=0`, weil der Profiler Epoche 0 misst. Ein kurzer Lauf mit dem
normalen `no_aug_epochs` würde sonst den leichteren Dataloader ohne
Augmentierung statt des tatsächlich im Training verwendeten profilieren.
Außerdem meldet `--repeat N` Mittelwert und Standardabweichung. Das ist wichtig,
weil ein durch Starts begrenzter Schritt so stark schwankt, dass ein einzelner
Lauf irreführt. Der Befehl schreibt Versuchsverzeichnisse `prof_1`, `prof_2`
und so weiter sowie eine zusammengefasste Datei `profile_repeat.json`.

## Mixed Precision

`amp=True` ist für die meisten Familien der Standard und führt den Forward Pass
unter CUDA Autocast aus. `amp_dtype` wählt `float16` oder `bfloat16`.

<code-tabs name="amp" />

Float16 benötigt dynamische Loss-Skalierung und erhält einen aktiven
Gradient-Scaler. Der größere Exponentenbereich von bfloat16 benötigt ihn nicht,
weshalb der Scaler deaktiviert ist. Vier Familien werden mit `amp=False`
ausgeliefert: D-FINE, DEIM, YOLO-NAS und FOMO. Die DEIM-Einstellung wird durch
Vererbung auf RT-DETRv4 übertragen. D-FINE nennt den Grund: Sein Decoder
begrenzt Aktivierungen auf 65504, den größten endlichen float16-Wert.

Die Argumentsemantik einschließlich des Verhaltens einer bfloat16-Anfrage auf
Hardware ohne bfloat16-Unterstützung wird unter
[Hyperparameter](/docs/train/hyperparameters) beschrieben.

## CUDA Graphs

`cuda_graph=True` erfasst den Forward und Backward Pass des Netzes in einem
CUDA Graph und entfernt den Overhead der Kernel-Starts pro Schritt.

<code-tabs name="graph" />

Das Flag kann immer sicher übergeben werden. Wenn eine Familie, Aufgabe oder
Konfiguration nicht erfasst werden kann, wird eine Zeile protokolliert und das
Training unverändert im Eager-Modus ausgeführt.

Nur das Netz wird erfasst. Der Loss bleibt bewusst im Eager-Modus, weil
Erkennungs-Loss-Funktionen mit booleschen Masken auswählen, Hungarian Matching
ausführen und anhand von Zuweisungsergebnissen verzweigen. Nichts davon kann
ein Graph aufzeichnen. Auch Optimizer-Schritt, Gradient Clipping, EMA-Update
und Lernratenplan bleiben im Eager-Modus.

Der mögliche Gewinn wird dadurch auf den Netzanteil eines Schritts begrenzt,
der stark variiert. Gemessen auf einer RTX 5070 Ti bei 640 px und Batch-Größe
8 entfallen 84 % eines YOLOv9-t-Schritts auf das Netz, 44 % bei YOLOv7-b, 31 %
bei YOLOX-t und 26 % bei RTMDet-t. Die beiden letzten verbringen den Großteil
des Schritts in ihren Label-Assigner-Funktionen. Die Erfassung des Netzes hilft
ihnen daher am wenigsten.

### Erwartbarer Nutzen

Bedingungen für alle nachfolgenden Werte: RTX 5070 Ti, Windows, AMP, ein
Prozess pro Variante aus einem gemeinsamen gespeicherten Zustand, Wiederholung
eines echten Batches ohne Dataloader, schnellster von 24 Schritten nach dem
Warmup. Erkennung bei 640 px, Klassifikation bei 224 px. Die Batch-Größe gilt
pro Zeile.

| Familie | Größe | Batch | Eager | Graph | Beschleunigung |
|---|---|---:|---:|---:|---:|
| FOMO | s | 16 | 7.0 ms | 1.9 ms | 3.63x |
| MobileNetV4 | s | 16 | 14.5 ms | 5.3 ms | 2.74x |
| EfficientNetV2 | b0 | 16 | 29.0 ms | 11.9 ms | 2.44x |
| YOLOv9 | t | 8 | 93.6 ms | 47.0 ms | 1.99x |
| NAFNet | s | 8 | 132.5 ms | 105.5 ms | 1.26x |
| PicoDet | s | 8 | 145.0 ms | 118.7 ms | 1.22x |
| D-FINE | n | 4 | 185.3 ms | 159.2 ms | 1.16x |
| RF-DETR | n | 4 | 276.3 ms | 239.8 ms | 1.15x |
| YOLOX | t | 8 | 102.2 ms | 90.5 ms | 1.13x |
| RTMDet | t | 8 | 149.7 ms | 136.2 ms | 1.10x |
| YOLOv7 | b | 4 | 102.5 ms | 98.0 ms | 1.05x |

Diese Werte isolieren den GPU-Schritt. Ein vollständiges Fine-Tuning bezahlt
zusätzlich für Dataloader und Validierung. YOLOv9-t auf einem Erkennungsdatensatz
mit 406 Bildern, 20 Epochen, Batch-Größe 8, 640 px und 4 Dataloader-Workern auf
demselben Rechner benötigte im Eager-Modus 428.4 s und mit Graph 367.7 s. Das
entspricht einer Beschleunigung um 1.16x bei identischer mAP50-95 von 0.6394.

Drei Faktoren verändern diese Werte. Kleine Batches sind durch Kernel-Starts
begrenzt, große durch Rechenleistung. RT-DETR-r18 gewinnt daher bei Batch-Größe
2 den Faktor 1.19x und bei Batch-Größe 8 nur 1.04x. Der Start-Overhead ist unter
Windows am höchsten. Unter Linux liegen die Gewinne ungefähr bei einem Drittel
bis der Hälfte der Tabelle. Bei einem durch den Dataloader begrenzten Lauf
ändert sich die Gesamtdauer überhaupt nicht. Deshalb steht der Profiler am
Anfang.

Die Erfassung greift bei `amp=False` auf dieselbe Weise. FP32-Kernel laufen
jedoch länger, sodass ein Schritt weniger durch Starts begrenzt ist und die
meisten Familien weniger gewinnen. Auf derselben Hardware ändert sich
MobileNetV4-s bei Batch-Größe 16 von 2.74x unter AMP auf 3.61x unter FP32,
YOLOv9-t bei Batch-Größe 8 von 1.99x auf 1.69x und RT-DETR-r18 bei Batch-Größe
4 von 1.12x auf 0.99x.

### Geltungsbereich der Erfassung

| Aufgabe | Familien |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Alle anderen Fälle fallen mit einer Protokollzeile auf den Eager-Modus zurück:
andere Aufgaben dieser Familien, nicht aufgeführte Familien, verteilte Läufe
und Distillationsläufe. Auch ein Erfassungsfehler zur Laufzeit stellt den Rest
des Laufs auf den Eager-Modus um, statt fehlzuschlagen.

Bei den Encoder-Decoder-Detektoren D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 und v4
sowie EC werden nur Backbone und Encoder erfasst. Ihr Decoder liest die Ground
Truth, um Contrastive-Denoising-Queries zu erstellen. Die Anzahl dieser Queries
richtet sich nach der größten Ground-Truth-Anzahl im Batch. Seine Tokenanzahl
ändert sich daher von Batch zu Batch.

### Formen

Ein Graph gilt genau für die Eingabeform, mit der er erfasst wurde. Der Trainer
zählt Batch-Formen und erfasst, sobald eine Form dreimal aufgetreten ist.
Batches jeder anderen Form laufen im Eager-Modus. Dies betrifft
Multi-Scale-Batches und den letzten unvollständigen Batch einer Epoche.

Für DETR-Familien, die standardmäßig jeden Batch skalieren, ist dies ein
Fallstrick. Mit `multi_scale=True` sieht ein kurzer Lauf möglicherweise keine
Form oft genug für eine Erfassung. Übergib `multi_scale=False`, wenn die
Beschleunigung das Ziel ist.

YOLOX verändert während eines Laufs den Inhalt des erfassten Bereichs. Wenn
Mosaic bei `no_aug_epochs` endet, wird sein L1-Regressionszweig aktiviert. Der
Trainer macht die Erfassung an dieser Stelle ungültig und erfasst erneut,
sobald sich die neue Form stabilisiert hat.

### Numerik und Speicher

Die meisten Familien reproduzieren ihre Eager-Loss-Trajektorie unter AMP
bitgenau. FOMO und LingBot-Vision unterscheiden sich durch eine andere
Summationsreihenfolge im letzten Bit von float32. Die Detektoren mit
deformierbarer Attention, D-FINE, DEIM, DEIMv2, RT-DETR, RF-DETR und EC,
reproduzieren auch ihre eigenen Eager-Läufe nicht. Ihr Backward Pass akkumuliert
mit atomaren Operationen, und TF32-Convolutions wählen pro Start eine
Reduktionsreihenfolge. Der Graph-Lauf bleibt innerhalb dieser Streuung. RTMDet
unterscheidet sich bei zwei von 139 Gradienten relativ um ungefähr 3e-4. Es
teilt Head-Convolutions über Pyramidenebenen, und die beiden Backward-Pfade
summieren drei Beiträge in unterschiedlicher Reihenfolge. SegFormer besitzt
Stochastic Depth im erfassten Bereich. Ein wiedergegebener Graph zieht daher
einen eigenen Zufallsstrom und ist statistisch zum Eager-Modus äquivalent,
statt identisch. Der Manager protokolliert dies einmal bei der Erfassung.

Bei `amp=False` ist auf dieser Hardware weder mit noch ohne Erfassung
Bitidentität verfügbar. Zwei identisch reproduzierbar initialisierte
YOLOv9-t-Eager-Läufe weichen über 20 Schritte relativ um 36 % voneinander ab,
YOLOX-t um 2.6 %. cuDNN wählt für einige FP32-Convolution-Formen einen
nichtdeterministischen Algorithmus für Gewichtsgradienten.

Ein erfasster Graph hält statische Eingabe-, Ausgabe- und Workspace-Puffer
fest. Der VRAM-Spitzenbedarf steigt daher ungefähr um einen zusätzlichen Satz
Aktivierungen. Über die obigen Familien änderte sich die Spitzenallokation
zwischen -5 und +19 %. Die relativen Kosten sind bei kleinen
Klassifikationsmodellen am höchsten, deren Aktivierungen ohnehin klein sind.
ResNet-18 bei 224 px und Batch-Größe 16 stieg von 0.48 GB im Eager-Modus auf
0.57 GB mit Graph. Wenn ein Lauf dadurch die Grenze überschreitet, verringere
den Batch oder deaktiviere das Flag.

## Verwandte Themen

- Unter [Hyperparameter](/docs/train/hyperparameters) findest du `batch`,
  `nbs`, `cache` und `workers`.
- [Multi-GPU-Training](/docs/train/multi-gpu) unterstützt weder CUDA Graphs
  noch den Profiler.
- [CUDA Graphs](/docs/reference/cuda-graphs) enthält die kombinierte
  Support-Matrix für Inferenz und Training, die Aufteilungspunkte und den
  Numerikvertrag.

