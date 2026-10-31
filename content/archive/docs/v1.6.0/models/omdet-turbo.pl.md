---
title: OMDet-Turbo
families:
  - omdet_turbo
seo_title: 'OMDet-Turbo w LibreYOLO: detekcja zero-shot w czasie rzeczywistym'
description: >-
  Używaj OMDet-Turbo w LibreYOLO do detekcji z otwartym słownikiem w czasie
  rzeczywistym. Zainstaluj dodatek openvocab i uruchamiaj predykcję ze swobodnym
  słownikiem tekstowym.
lead: >-
  OMDet-Turbo to detektor obiektów z otwartym słownikiem czasu rzeczywistego
  opracowany przez Om AI Lab, który rozdziela embeddingi klas od promptu zadania
  językowego. LibreYOLO udostępnia go jako rodzinę wyłącznie do predykcji na
  poziomie detektorów z otwartym słownikiem.
keywords:
  - OMDet-Turbo
  - OmDet
  - detekcja obiektów z otwartym słownikiem
  - detekcja w czasie rzeczywistym
  - detekcja zero-shot
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.3)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Niestandardowy próg NMS
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("omdet-turbo")

        model.set_classes(["traffic light", "bicycle"])


        # OMDet-Turbo jest jedyną rodziną tego poziomu, która respektuje iou=.
        Jej

        # własne przetwarzanie końcowe przyjmuje próg tłumienia jako argument,

        # domyślnie 0.5, gdy iou= nie zostanie ustawione.

        result = model.predict(SAMPLE_IMAGE, conf=0.3, iou=0.7)

        print(result.names, len(result))
source_hash: c2a375d234341b7e
---

## Instalacja

OMDet-Turbo jest wczytywany przez poziom detektorów z otwartym słownikiem w
LibreYOLO, który wymaga dodatku `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Ten dodatek instaluje `transformers` i `timm`, biblioteki Hugging Face używane
przez ten poziom. Backbone Swin modelu OMDet-Turbo jest wczytywany przez wrapper
`TimmBackbone` z `transformers`.

## Predykcja

OMDet-Turbo nie jest checkpointem wczytywanym przez LibreYOLO za pomocą
`LibreYOLO()`. Korzysta z pokrewnej fabryki `LibreOpenVocab`, która przy
pierwszym użyciu pobiera snapshot Hugging Face i zapisuje go w pamięci
podręcznej w `weights/`.

<code-tabs name="predict" />

`set_classes()` ustawia trwały słownik tekstowy. Ponowne wywołanie całkowicie
zastępuje listę, a pominięcie zachowuje domyślne etykiety COCO-80. Pusty wynik
jest poprawnym rezultatem, a nie błędem. W przeciwieństwie do Grounding DINO,
OMDet-Turbo oddziela embeddingi klas od promptu zadania językowego, dlatego
przetwarzanie końcowe `transformers` zwraca etykiety mapowane bezpośrednio na
odpytywaną listę klas, bez etapu rozstrzygania fraz.

OMDet-Turbo nie ma progu tokenów tekstowych. Detekcje filtruje wyłącznie `conf`,
a przekazanie `text_threshold` zgłasza błąd. Jest to jedyna rodzina tego poziomu,
która uruchamia własne tłumienie niemaksymalne wewnątrz
`post_process_grounded_object_detection`, dlatego `iou` jest tutaj respektowane,
a nie ignorowane z ostrzeżeniem. Argumenty `imgsz` i `augment=True` są
odrzucane. Procesor `transformers` sam kontroluje zmianę rozmiaru, a augmentacja
podczas testowania pozostaje poza zakresem tego poziomu. `predict()` dla jednego
obrazu zwraca jeden obiekt `Results`, a nie listę. Katalog, listę obrazów lub
`stream=True` ze źródłem wideo należy przekazać, aby uzyskać kilka wyników. Dla
tej rodziny nie ma ścieżki CLI. `libreyolo predict` wczytuje przez
`LibreYOLO()` wyłącznie checkpointy `.pt`, dlatego rodziny `LibreOpenVocab`
uruchamia się z Pythona. Informacje o typach źródeł i streamingu zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępny jest jeden checkpoint, `t`, jedyny rozmiar tego poziomu. Jest kopią
lustrzaną `omlab/omdet-turbo-swin-tiny-hf` przy przypiętej rewizji źródłowej,
utworzoną za pośrednictwem `OmDetTurboForObjectDetection` z `transformers`.
Plik wag w kopii lustrzanej jest identyczny bajt po bajcie ze źródłowym
snapshotem. Nie opublikowano jeszcze wyników dokładności ani opóźnienia tej
rodziny.

Trenowanie, walidacja zbioru danych i eksport pozostają poza zakresem tego
poziomu. `train()`, `val()` i `export()` zawsze zgłaszają
`NotImplementedError`. Jest to wrapper przeznaczony wyłącznie do predykcji z
opublikowanym checkpointem.

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
