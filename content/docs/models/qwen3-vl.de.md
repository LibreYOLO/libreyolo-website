---
title: Qwen3-VL
families:
  - qwen3vl
seo_title: 'Qwen3-VL in LibreYOLO: Open-Vocabulary-Erkennung'
description: >-
  Qwen3-VL in LibreYOLO: Installiere Alibabas Vision-Language-Modell unter
  Apache-2.0, lege ein offenes Vokabular fest und sage vorher oder chatte.
lead: >-
  Qwen3-VL ist Alibabas Vision-Language-Modell mit nativem 2D-Grounding.
  LibreYOLO bindet es als Open-Vocabulary-Objektdetektor ein und stellt seinen
  freien Chat direkt bereit: Übergib eine Klassenliste zur Erkennung oder stelle
  eine Frage.
keywords:
  - qwen3-vl python
  - vision-language model
  - open-vocabulary-erkennung
  - grounding
  - alibaba qwen
  - vlm
last_verified: "1.6.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-4b")
        model.set_classes(["forklift", "pallet", "safety vest"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat
      language: python
      code: >
        from libreyolo import LibreVLM, SAMPLE_IMAGE


        model = LibreVLM("qwen3-vl-4b")


        # Der direkte Weg unterhalb der Erkennungsfunktion: jede beliebige
        Frage,

        # nicht nur eine Anfrage nach Bounding Boxes.

        answer = model.chat(SAMPLE_IMAGE, "How many people are wearing a safety
        vest?")

        print(answer)
source_hash: "801d97d089f1f957"
---

## Installation

Qwen3-VL gehört zum VLM-as-Detector-Tier von LibreYOLO. Dieses besitzt eine
eigene Factory und eine getrennte Produktoberfläche neben den
Checkpoint-basierten Familien. Es benötigt das Zusatzpaket `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen
und lokal zwischengespeichert. Ein Aufruf von `LibreVLM()` ohne Argument
verwendet standardmäßig Qwen3-VL-4B.

<code-tabs name="predict" />

Diese Familie wird über die Factory `LibreVLM()` und nicht über `LibreYOLO()`
geladen. VLM-Familien deklarieren keinen Checkpoint-Loader. Das auf anderen
Modellseiten beschriebene Routing über die Dateiendung gilt daher nicht.
`set_classes()` legt das Vokabular fest, nach dem Qwen3-VL suchen soll. Es
bleibt wirksam, bis du es erneut festlegst, und gilt somit für jeden späteren
Aufruf von `predict()` oder `track()`. Jede Erkennung trägt dieselbe
Platzhalter-Confidence. Die Filterung mit `conf` ist daher eine
Alles-oder-nichts-Entscheidung statt einer Rangfolge. `iou` hat bei dieser
Familie eine Wirkung: Eine spätere Box derselben Klasse wird verworfen, wenn
sie eine bereits beibehaltene Box stärker als der Schwellenwert überlappt.
Dadurch werden nahezu identische Boxen entfernt, die ein sich wiederholender
Generator sonst für ein Objekt ausgeben kann. Anders als Florence-2 und
Kosmos-2 beantwortet Qwen3-VL auch freie Fragen über `chat()`, denselben
direkten Weg, der bei der Factory `LibreVLM` dokumentiert ist. Die CLI von
LibreYOLO unterstützt diesen Tier nicht. Es gibt keine Form
`libreyolo predict model=...` dafür. Unter [Vorhersage](/docs/predict) findest
du Quellen, Streaming und die Verarbeitung von Ergebnissen.

## Varianten

Es gibt drei Größen: Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct und
Qwen3-VL-8B-Instruct, geladen als `LibreVLM("qwen3-vl-2b")`,
`LibreVLM("qwen3-vl-4b")` und `LibreVLM("qwen3-vl-8b")`. Alle drei geben eine
nominelle Eingabe von 1024 px an. Die eigene intelligente Größenänderung des
Qwen-Prozessors bestimmt jedoch die tatsächlich an das Netzwerk übergebene
Arbeitsfläche. Dieser Wert ist daher keine feste Betriebsauflösung wie bei den
anderen Familien auf dieser Website. LibreYOLO hat keinen Benchmark zum
Accuracy-Vergleich der drei Größen veröffentlicht.

## Training

Installiere `libreyolo[vlm-train]`, um LoRA-Adapter für die Erkennung über `LibreVLM("qwen3-vl-2b").train(data=...)` zu trainieren. Der Vision-Tower bleibt eingefroren; der Validierungs-Loss bestimmt das beste Checkpoint-Verzeichnis. Das Fortsetzen mit Optimizer-Zustand und die Validierung der Erkennungs-mAP werden nicht unterstützt. Siehe [VLM-Fine-Tuning](/docs/train/vlm-fine-tuning) für Standardwerte und Hinweise zum erneuten Laden.

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
