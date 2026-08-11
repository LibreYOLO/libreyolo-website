---
title: Open-Vocabulary-Erkennung
seo_title: Open-Vocabulary-Erkennung in LibreYOLO
description: >-
  Erkenne in LibreYOLO Objekte anhand eines Textvokabulars. Lade Grounding DINO,
  OWLv2, OMDet-Turbo oder OV-DEIM über LibreOpenVocab und lege die Klassen zur
  Laufzeit fest.
lead: >-
  Die Open-Vocabulary-Erkennung ersetzt die feste Klassenliste eines Checkpoints
  durch Wörter, die du beim Aufruf auswählst. In LibreYOLO ist sie keine eigene
  Aufgabe. Sie verwendet die detect-Aufgabe einer eigenen Modellstufe, die über
  LibreOpenVocab statt LibreYOLO geladen wird.
keywords:
  - open-vocabulary-erkennung
  - zero shot objekterkennung
  - open set detection
  - grounding dino python
  - owlv2
  - omdet turbo
  - text prompt objekterkennung
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        print(result.names)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Vokabular wechseln
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("owlv2-b16")


        # set_classes bleibt bis zum nächsten Aufruf aktiv. Nach Umwandlung in

        # Kleinbuchstaben und Entfernung von Artikeln müssen Labels eindeutig
        sein.

        model.set_classes(["a red backpack", "traffic cone"])

        result = model.predict(SAMPLE_IMAGE)


        model.set_classes(["bicycle wheel"])

        result = model.predict(SAMPLE_IMAGE)
    - label: Grounding-DINO-Textschwellenwert
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf filtert nach Boxen-Score, text_threshold nach Token-Score der
        # decodierten Phrase. Ohne Angabe sind beide 0.25. Nur Grounding DINO
        # akzeptiert text_threshold, die anderen lösen einen Fehler aus.
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
source_hash: 17197cf4d80f3d6f
---

## Definition

Die Open-Vocabulary-Erkennung gibt normale `Results` einer Erkennung zurück:
Boxen, Confidence-Werte und Klassenindizes. `result.names` ordnet diese Indizes
den angeforderten Strings zu. Der Unterschied liegt in der Herkunft der
Klassenliste. Ein herkömmlicher Detektor wird mit einem festen Kategoriensatz
trainiert und kann nie eine Kategorie außerhalb davon ausgeben. Diese Modelle
nehmen das Vokabular bei der Inferenz als Text entgegen. Mit
`set_classes(["forklift", "safety cone"])` werden diese Begriffe daher zu den
Klassen.

LibreYOLO besitzt keinen Aufgabenschlüssel `open-vocabulary`. Diese Modelle
deklarieren wie jeder andere Detektor `SUPPORTED_TASKS = ("detect",)`. Sie
unterscheiden sich durch den Ladepfad. Da es sich um Snapshots von Hugging Face
und nicht um State-Dict-Checkpoints von LibreYOLO handelt, werden sie nicht von
der Factory `LibreYOLO()` geladen, sondern über `LibreOpenVocab()` erstellt.
Diese Factory steht neben `LibreSAM()` und `LibreVLM()` und ersetzt
`LibreYOLO()` nicht.

Die Scores sind echte Erkennungs-Scores und keine nachträglich geparsten
generierten Bildbeschreibungen. Jede Familie bewertet Bildregionen anhand des
Text-Embeddings jedes Prompts.

## Modelle

Vier Familien bilden diese Stufe und unterstützen ausschließlich Vorhersagen.
Lade jede davon über ihren Alias mit `LibreOpenVocab`.

[Grounding DINO](/docs/models/grounding-dino) von IDEA Research gibt es in den
Größen `t` und `b`. Es ist der Standard dieser Stufe und die einzige Familie,
die `text_threshold` akzeptiert. Dieser zweite Grenzwert gilt für den
Token-Score der decodierten Phrase.

[OWLv2](/docs/models/owlv2) von Google Research gibt es in den Größen `b16` und
`l14`. Die Familie bewertet Bildregionen anhand von Text-Embeddings aus einem
CLIP-artigen Encoder.

[OMDet-Turbo](/docs/models/omdet-turbo) vom Om AI Lab gibt es in einer Größe
`t`. Die Familie trennt Klassen-Embeddings von einem sprachlichen
Aufgaben-Prompt. Als einzige Familie dieser Stufe unterdrückt sie überlappende
Boxen in der eigenen Nachverarbeitung, weshalb `iou=` berücksichtigt wird.

[OV-DEIM](/docs/models/ov-deim) gibt es in den Größen `s`, `m` und `l`. Dieser
DETR-artige Detektor gleicht Decoder-Queries mit Text-Embeddings eines
gebündelten MobileCLIP-Text-Towers ab. Er verwendet eine eindeutige Zuordnung
mit Top-K-Auswahl, sodass keine NMS ausgeführt wird.

Die Gewichte von OV-DEIM bilden den eingeschränkten Fall dieser Stufe. Die
Detektorgewichte stehen unter CC BY-NC 4.0 und sind nicht kommerziell nutzbar.
Der gebündelte Text-Tower steht unter Apples Machine Learning Research Model
License und ist nur für Forschungszwecke bestimmt. Der Checkpoint `l` ergänzt
ein nachtrainiertes DINOv3-S-Backbone unter Metas DINOv3 License. Alle drei
Lizenztexte werden im Gewichts-Repository ausgeliefert. Die Bibliothek
protokolliert dieselbe Zusammenfassung beim Auflösen der Gewichte, bevor das
Modell erstellt wird. Lies vor dem Deployment die Seite zu
[OV-DEIM](/docs/models/ov-deim).

Die Stufe benötigt ein Extra:

```bash
pip install "libreyolo[openvocab]"
```

Es umfasst `transformers` und `timm` für die drei eingebundenen Familien sowie
`huggingface_hub`, `safetensors`, `regex` und `ftfy`, die OV-DEIM als native
Portierung benötigt.

Eine zweite Stufe nimmt ebenfalls ein Textvokabular entgegen: `LibreVLM()`
lädt generative Vision-Language-Modelle wie
[Qwen3-VL](/docs/models/qwen3-vl) und
[Florence-2](/docs/models/florence-2) und wandelt ihre Ausgabe in dieselben
`Results` um. Sie stellt ebenfalls `set_classes()` bereit. Der Unterschied
liegt in der Erzeugung der Boxen. Die Familien dieser Seite sind
diskriminative Detektoren, die Scores direkt ausgeben. Die VLM-Stufe generiert
die Boxen dagegen.

## Vorhersage

<code-tabs name="predict" />

`set_classes()` nimmt eine nicht leere Liste von Labelstrings entgegen und
bleibt bis zum nächsten Aufruf aktiv. Nach der Umwandlung in Kleinbuchstaben
und dem Entfernen vorangestellter Artikel müssen Labels eindeutig sein. Daher
können `"a bus"` und `"bus"` nicht gemeinsam in einem Vokabular stehen.
Phrasen aus mehreren Wörtern sind Labels wie alle anderen. Jede Familie
wandelt die Liste vor der Tokenisierung in ihre eigene Texteingabe um. Deshalb
ist `"traffic cone"` eine andere Query als `"cone"`.

Drei Vorhersageargumente verhalten sich hier anders als bei einem nativen
Detektor. `imgsz=` wird abgelehnt, da der Processor die Größenänderung für
diese Familien steuert. `augment=True` wird ebenfalls abgelehnt, weil
Test-Time-Augmentierung nicht zum Funktionsumfang dieser Stufe gehört. `iou=`
gilt nur für die Familie, deren Processor eine eigene Unterdrückung ausführt.
Wenn keine Unterdrückung stattfindet, wird das Argument mit einer Warnung
ignoriert.

Ohne Angabe verwendet `conf` den eigenen Standardwert der geladenen Familie
und nicht den üblichen Wert 0.25 von `predict()`. Dieser Standardwert ist nicht
in der gesamten Stufe gleich. Setze ihn explizit, wenn du zwei Familien mit
demselben Bild vergleichst.

`track()` löst in der gesamten Stufe einen Fehler aus. Führe stattdessen pro
Frame `predict()` aus. Unter [Vorhersage](/docs/predict) findest du
Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Training

Keine Familie dieser Stufe wird innerhalb von LibreYOLO trainiert. `train()`
löst einen Fehler aus. Führe das Fine-Tuning im Upstream-Projekt durch und lade
die resultierenden Gewichte. Das an `set_classes()` übergebene Vokabular ist
die einzige Einstellung, die beeinflusst, was ein geladenes Modell erkennt.

## Validierung

Für diese Stufe gibt es keinen Validator und `val()` löst einen Fehler aus.
Die Open-Vocabulary-Validierung benötigt eine eigene Implementierung, weil der
normale Erkennungsvalidator Bildtensoren direkt an das Modell übergibt, während
diese Familien parallel dazu erstellte textkonditionierte Eingaben benötigen.

## Export

Der Export liegt außerhalb des Funktionsumfangs dieser Stufe und `export()`
löst einen Fehler aus. Diese Modelle werden mit `predict()` in PyTorch
ausgeführt.

