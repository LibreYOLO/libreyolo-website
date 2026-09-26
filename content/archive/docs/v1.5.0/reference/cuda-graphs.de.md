---
title: CUDA-Graphen
seo_title: LibreYOLO-Unterstützungsmatrix für CUDA-Graphen
description: >-
  Welche Familien bei der Vorhersage ihren Vorwärtsdurchlauf und beim Training
  Vorwärts- und Rückwärtsdurchlauf aufzeichnen, welche Numerik garantiert wird,
  wo Aufzeichnungen geteilt werden und warum eine nicht unterstützte Familie
  einen Fehler auslöst.
lead: >-
  Ein CUDA-Graph zeichnet eine Ausführung einer festen Kernelsequenz auf und
  gibt sie mit einem einzigen Start wieder. LibreYOLO zeichnet die Inferenz bei
  39 geprüften Familien und das Training bei 24 auf. Dies geschieht immer
  familienbezogen, erst nach einer bitweisen Paritätsprüfung und nie als stiller
  Fallback.
keywords:
  - LibreYOLO CUDA Graph
  - cuda_graph=True
  - CUDA Graph Unterstützungsmatrix
  - Torch CUDA Graph Training
  - capture_error_mode thread_local
  - CUDA Graph bitidentisch
last_verified: 1.5.0
verification: >-
  Liste der Inferenzfamilien aus der CAPTURABLE-Matrix in
  tests/e2e/test_cuda_graph_families.py in v1.5.0 abgeleitet. Trainingsfamilien,
  Paritätsklassen und Zeitmessungen aus docs/training_cuda_graphs.md. API und
  NotImplementedError aus BaseModel._require_cuda_graph_support,
  cuda_graph_scope und capture_graph in libreyolo/models/base/model.py mit der
  Klassenvariable SUPPORTS_CUDA_GRAPH. Schnittstellenaufteilungen aus den
  Überschreibungen von _get_graph_runner in den Familien depth_anything3,
  birefnet, ppocr, sam und sensenova sowie aus
  libreyolo/models/base/detr_cuda_graph.py gelesen. capture_error_mode aus
  libreyolo/models/base/cuda_graph.py und libreyolo/training/cuda_graph.py.
  Trainings-Fallback aus libreyolo/training/trainer.py und Flag --cuda-graph aus
  libreyolo/cli/commands/train.py.
meta:
  - label: Inferenzfamilien
    value: '39'
  - label: Trainingsfamilien
    value: '24'
  - label: Inferenz-Flag
    value: predict(cuda_graph=True)
    mono: true
  - label: Trainings-Flag
    value: train(cuda_graph=True)
    mono: true
snippets:
  usage:
    - label: Vorhersage
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # True zeichnet bei der ersten Verwendung jeder Eingabeform auf.

        # "auto" wartet auf eine Wiederholung der Form, bevor
        Aufzeichnungskosten entstehen.

        result = model(SAMPLE_IMAGE, cuda_graph=True)
    - label: Training
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: Training über die CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=my-dataset.yaml \
          epochs=100 --cuda-graph
source_hash: 67c46199939278f2
---

## Aufgezeichnete Arbeit

Ein Graph zeichnet eine feste Kernelsequenz und die von ihr gelesenen und geschriebenen Speicheradressen auf. Er zeichnet weder Werte noch Formen oder Kontrollfluss auf. Die Wiedergabe benötigt einen einzigen Start statt Hunderten. Der Vorteil ist deshalb bei kleinen Netzwerken und kleinen Batchgrößen am größten, wenn der Start-Overhead statt der Arithmetik einen Schritt dominiert.

Die beiden Einstiegspunkte zeichnen unterschiedlich viel Arbeit auf.

| | Im Graphen | Eager |
|---|---|---|
| Inferenz | Netzwerk-Vorwärtsdurchlauf, `model._forward(x)` | Vorverarbeitung, NMS, gesamte Nachverarbeitung |
| Training | Vorwärts- und Rückwärtsdurchlauf des Netzwerks | Loss, Optimiererschritt, Gradientenbegrenzung, EMA, Lernratenschedule |

Weder NMS noch Erkennungs-Loss eignen sich dafür. Beide wählen mit booleschen Masken aus, führen ungarische Zuordnung oder einen Assigner aus und verzweigen anhand des Ergebnisses. Genau das kann ein Graph nicht aufzeichnen. Sie außerhalb zu halten, macht die Aufzeichnung sicher und ist keine zu umgehende Einschränkung.

<code-tabs name="usage" />

`cuda_graph` akzeptiert bei der Vorhersage drei Werte. `False` ist der Standard. `True` zeichnet bei der ersten Verwendung jeder Eingabeform auf. `"auto"` wartet, bis sich eine Form wiederholt. Einmalige Aufgaben und wechselnde Formen tragen dadurch keine Aufzeichnungskosten ohne spätere Wiederverwendung. `capture_graph(imgsz=None, batch=1, dtype=None)` verschiebt die Kosten aus der ersten Anfrage. `graph_info()` meldet aufgezeichnete Graphen und Wiedergabezähler, `release_graphs()` gibt sie frei.

Beim Training ist das Flag ein einfacher boolescher Wert und heißt in der CLI `--cuda-graph`. Die umgebenden Einstellungen beschreiben [Vorhersageleistung](/docs/predict/performance) und [Trainingsleistung](/docs/train/performance).

## Inferenzunterstützung

Die Unterstützung ist familienbezogen und wird über die Klassenvariable `SUPPORTS_CUDA_GRAPH` deklariert. Eine Familie wird erst markiert, nachdem Aufzeichnung und Wiedergabe bei zwei Testeingaben aus unterschiedlichen Verteilungen bitidentisch waren. Diese gemeinsame Paritätsmatrix deckt 39 Familien über neun Aufgaben ab.

| Aufgabe | Familien |
|---|---|
| detect | yolo1, yolo2, yolo3, yolo4, yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, rfdetr, ec |
| segment | dfine, rtmdet, rfdetr, ec |
| pose | ec, yolonas, rfdetr |
| point | fomo |
| classify | resnet, convnext, mobilenetv4, efficientnetv2, clip, dinov2, siglip2 |
| semantic | eomt, dinov2, segformer, pidnet, lingbotvision |
| depth | depth_anything, depth_anything3, zipdepth |
| restore | nafnet, realesrgan, swinir |
| matte | birefnet |

Mehrere Familien erscheinen unter mehr als einer Aufgabe. Die Matrix führt daher mehr Zeilen aus, als sie unterschiedliche Familien enthält. Drei weitere Familien zeichnen über familienspezifische Codepfade mit eigenen Tests statt über die gemeinsame Matrix auf und gehören nicht zu den 39: PP-OCR, SAM und SenseNova.

Die Prüfung erfolgt bitweise und nicht näherungsweise. Eine frühere Version des Protokolls bewertete Parität anhand der relativen Größe und stufte drei funktionierende Familien fälschlich herab: YOLOX, EfficientNetV2 und YOLOv7. Ihre Differenz zwischen Eager und Graph lag bei etwa 1e-7, obwohl sie bei der relevanten Testeingabe bitidentisch waren.

## Trainingsunterstützung

Die Trainingsaufzeichnung wurde in dieser Veröffentlichung von zwei auf 24 Familien über fünf Aufgaben erweitert.

| Aufgabe | Familien |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Alles andere wird im Eager-Modus trainiert: andere Aufgaben derselben Familien, nicht aufgeführte Familien, verteilte Läufe und Destillationsläufe. Solange eine Form noch neu ist, wird die Aufzeichnung ebenfalls übersprungen. Der Trainingspfad wartet auf drei Wiederholungen einer Eingabeform. Bei `multi_scale=True` findet deshalb möglicherweise überhaupt keine Aufzeichnung statt.

## Zwei Reaktionen auf eine nicht unterstützte Familie

Der Inferenzpfad löst einen Fehler aus. `predict(cuda_graph=True)` bei einer nicht aktivierten Familie löst `NotImplementedError` unter Angabe der Familie aus, statt Eager auszuführen und eine nicht vorhandene Beschleunigung vorzutäuschen. Eine fehlerhafte Aufzeichnung schlägt nicht unbedingt sichtbar fehl. Die Wiedergabe eines Vorwärtsdurchlaufs mit nicht aufzeichenbarer Arbeit kann unbemerkt falsche Zahlen liefern. Unterstützung muss daher eine ausdrückliche familienbezogene Zusicherung und darf kein Versuch mit Fallback sein.

Der Trainingspfad protokolliert dagegen. `train(cuda_graph=True)` kann immer sicher übergeben werden. Eine nicht aufzeichenbare Familie, Aufgabe oder Konfiguration schreibt eine Zeile und trainiert unverändert im Eager-Modus. Schlägt eine Aufzeichnung während eines Laufs fehl, wechselt auch der Rest des Laufs zu Eager, statt ihn abzubrechen. Die Asymmetrie ist beabsichtigt. Eine Vorhersage lässt sich an ihrer Aufrufstelle korrigieren, während ein Trainingslauf nicht in der sechsten Stunde wegen einer optionalen Optimierung enden soll.

## Aufteilung an Schnittstellen

Einige Familien lassen sich nicht vollständig aufzeichnen, da eine Stufe tatsächlich nicht aufzeichenbare Arbeit ausführt. Statt die Familie zu verwerfen, wird an einer geprüften Schnittstelle geteilt. Der aufzeichenbare Teil wird wiedergegeben, der Rest im Eager-Modus ausgeführt. Die kombinierte Ausgabe entspricht der vollständig im Eager-Modus ausgeführten Variante.

| Familie | Aufgezeichnet | Eager und Grund |
|---|---|---|
| Depth Anything 3 | Netzwerk | Sky-Schritt mit für den Host sichtbarer Arbeit nach dem Vorwärtsdurchlauf |
| BiRefNet | Encoder, `forward_enc` | Decoder, dessen `deform_conv2d` unter Aufzeichnung ein anderes Ergebnis wiedergibt |
| PP-OCR | Erkennungsstufe, `forward_det` | Erkennung, da die Beschnittbreite je Zeile variiert |
| SAM | Bildencoder | Prompt-Pfad, der je Codierung mehrfach ausgeführt wird |
| SenseNova | Vision-Tower | Autoregressive Generierung mit einem bei jedem Schritt wachsenden KV-Cache |
| Encoder-Decoder-Detektoren | Backbone und Encoder | Decoder und ungarisches Kriterium |

Die Aufteilung von BiRefNet ist besonders wichtig. Das Fehlverhalten von `deform_conv2d` bei der Wiedergabe lässt sich mit einem einfachen Aufruf außerhalb jedes Modells reproduzieren. Ein Ersatz durch eine reine PyTorch-Variante wurde verworfen, da dies auch Eager-Vorhersagen verändert hätte. Die Eager-Werte sind der Vertrag.

Der Encoder-Decoder-Fall gilt für D-FINE, DEIM, DEIMv2, RT-DETR, RT-DETRv2, RT-DETRv4 und EC. Ihr Decoder erstellt Contrastive-Denoising-Abfragen aus der Ground Truth. Die Abfrageanzahl stammt von der größten Ground-Truth-Anzahl im Batch, sodass sich die Tokenanzahl des Decoders zwischen Batches ändert. Genau das kann ein Graph nicht tolerieren. Backbone und Encoder beanspruchen bei diesen Familien etwa ein Fünftel bis ein Viertel eines Schritts. Deshalb stehen sie am unteren Ende der Beschleunigungstabelle.

PP-OCR zeichnet je Eingabeform der Erkennungsstufe einen Graphen auf. Die Anzahl wird durch die Cache-Obergrenze des Runners begrenzt. Wenn kein Aufzeichnungsbereich aktiv ist, wird das Eager-Ergebnis zurückgegeben.

## Numerik

Die meisten Familien sind bitidentisch. Wo sie es nicht sind, wird der Grund konkret benannt. Bei Schritt null des Trainings ist der Loss für alle 24 Familien bitidentisch und kein BatchNorm-Puffer unterscheidet sich. Die Kategorien werden durch den Gradientenvergleich getrennt.

| Klasse | Familien | Bedeutung |
|---|---|---|
| Exakt | Die meisten der 24 | Jedes Gradientenbit ist identisch |
| 1 ULP | fomo, lingbotvision | Letztes Bit von float32, relativ etwa 1e-7, aufgrund einer anderen Summierungsreihenfolge |
| Eager-Rauschen | DETR-Abstammungslinie | Graph und Eager unterscheiden sich höchstens so stark wie zwei Eager-Läufe voneinander |
| Gleitkommarundung | rtmdet | 137 von 139 Gradienten bitidentisch, zwei unterscheiden sich um etwa 3e-4 |
| Eigener RNG-Stream | segformer | Stochastic Depth liegt innerhalb des aufgezeichneten Bereichs |

Die Klasse Eager-Rauschen muss korrekt interpretiert werden. Bei diesen Familien unterscheiden sich bereits zwei Eager-Läufe mit demselben Seed. Bitidentische Ausgabe ist daher keine Hürde, die nur der Graph verfehlt. Niemand erfüllt sie. Bei `amp=False` gilt dies noch breiter. Eine gemessene relative Nichtdeterministik von 3,2e-7 in einem FP32-Gewichtsgradienten verstärkt sich. Zwei Eager-Läufe von YOLOv9-t mit demselben Seed weichen nach 20 Schritten um 36 Prozent voneinander ab. Die Deaktivierung von TF32 behebt dies nicht.

## Angehefteter Speicher

Die Aufzeichnung läuft mit `capture_error_mode="thread_local"`. Im PyTorch-Standardmodus `"global"` ruft ein Pin-Memory-Thread des DataLoaders bei der Vorbereitung des nächsten Batches `cudaHostAlloc` auf. Dies macht die laufende Aufzeichnung ungültig und wird zugleich durch sie beschädigt. Der Lauf bricht beim Abruf des nächsten Batches mit einem Fehler aus dem Pin-Memory-Thread ab. Dieses Zusammenspiel wurde in einer echten Trainingskampagne zweimal beobachtet, bevor die Ursache gefunden wurde.

Der Thread-lokale Modus beschränkt nur den aufzeichnenden Thread. Der Pin-Thread berührt den Aufzeichnungsstream nie, daher gehört keine seiner Operationen in den Graphen. Das Training geht weiter und ersetzt vorübergehend `torch.cuda.CUDAGraph` durch eine Unterklasse, die den Modus erzwingt. Das ist nötig, weil `make_graphed_callables` dafür kein Argument bereitstellt. Eine Sperre verhindert, dass zwei gleichzeitige Aufzeichnungen die Ersetzung installiert lassen.

## Nutzen

Gemessen auf einer RTX 5070 Ti unter AMP mit einem Prozess je Variante und wiederholter Ausführung eines echten Batches, sodass der DataLoader nicht beteiligt ist. Angegeben ist der schnellste von 24 Schritten nach dem Warmup. Erkennung bei 640 px, Klassifizierung bei 224 px.

| Familie | Batch | Beschleunigung |
|---|---:|---:|
| FOMO s | 16 | 3,63x |
| MobileNetV4 s | 16 | 2,74x |
| EfficientNetV2 b0 | 16 | 2,44x |
| YOLOv9-t | 8 | 1,99x |
| YOLOv9 e2e | 8 | 1,76x |
| YOLOv9 p2 | 8 | 1,49x |
| Alle übrigen | unterschiedlich | 1,04x bis 1,26x |

Ein vollständiger Lauf profitiert weniger, da ein Graph weder DataLoader noch Validierung beschleunigen kann. Ein Fine-Tuning von YOLOv9-t über 20 Epochen auf 406 Bildern sank von 428,4 s auf 367,7 s. Das entspricht einer 1,16-fachen Beschleunigung im Gesamtlauf. In beiden Varianten waren mAP50-95 mit 0,6394 und die Losses je Epoche identisch.

Die Obergrenze hängt vom Netzwerkanteil eines Schritts ab. Auf derselben Hardware bei 640 px und Batchgröße 8 beträgt er 84 Prozent für YOLOv9-t, aber nur 26 Prozent für RTMDet-t. Letzteres verbringt den größten Teil eines Schritts im Label-Assigner. Der Start-Overhead ist unter Windows am höchsten. Unter Linux liegen die Vorteile bei etwa einem Drittel bis der Hälfte dieser Tabelle. Ein durch den DataLoader begrenzter Lauf zeigt überhaupt keine Änderung der Gesamtdauer. Der Spitzenspeicher reicht von 5 Prozent weniger bis 19 Prozent mehr.

## Einschränkungen

Ein Graph zeichnet Adressen und keine Werte auf. Jede Aktion, die Parameter verschiebt, verwirft ihn. Dazu gehören der Gerätewechsel über `predict(device=...)`, Quantisierung und Dequantisierung.

Die Batchgröße ist wichtiger als die Familie. RT-DETR-r18 wird bei Batchgröße 2 um den Faktor 1,19 und bei Batchgröße 8 nur um 1,04 beschleunigt. Ein großer Batch ist durch Rechenleistung begrenzt und besitzt weniger entfernbaren Start-Overhead.

Die Inferenz-Paritätssuite wurde ohne das optionale Paket `kernels` ausgeführt. Die Aufzeichnungssicherheit bei aktiven kompilierten Hub-Kerneln wird daher nicht abgedeckt. Setze `LIBREYOLO_HUB_KERNELS=0`, um sie bei der Eingrenzung eines Aufzeichnungsproblems auszuschließen. Weitere Informationen findest du unter [Kernel](/docs/reference/kernels).
