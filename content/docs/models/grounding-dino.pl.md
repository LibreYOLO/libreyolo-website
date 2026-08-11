---
title: Grounding DINO
families:
  - grounding_dino
seo_title: 'Grounding DINO w LibreYOLO: detekcja zbioru otwartego'
description: >-
  Używaj Grounding DINO w LibreYOLO do wykrywania dowolnego obiektu opisanego
  tekstem. Zainstaluj dodatek openvocab i uruchamiaj predykcję ze swobodnym
  słownikiem tekstowym.
lead: >-
  Grounding DINO to detektor obiektów ze zbiorem otwartym opracowany przez IDEA
  Research, który ocenia obraz względem swobodnego promptu tekstowego zamiast
  stałej listy klas. LibreYOLO udostępnia go jako rodzinę wyłącznie do predykcji
  na poziomie detektorów z otwartym słownikiem.
keywords:
  - Grounding DINO
  - detekcja obiektów z otwartym słownikiem
  - detekcja zbioru otwartego
  - detekcja zero-shot
  - detektor sterowany tekstem
  - LibreOpenVocab
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
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Próg tekstowy
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("grounding-dino-b")

        model.set_classes(["remote control", "school bus"])


        # conf filtruje według wyniku ramki, a text_threshold według wyniku
        tokenu

        # zdekodowanej frazy. Oba mają domyślnie 0.25, gdy nie zostaną
        ustawione.

        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)

        print(result.names)
source_hash: 06bd13b8e6a66038
---

## Instalacja

Grounding DINO jest wczytywany przez poziom detektorów z otwartym słownikiem w
LibreYOLO, który wymaga dodatku `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Ten dodatek instaluje `transformers` i `timm`, biblioteki Hugging Face używane
przez ten poziom.

## Predykcja

Grounding DINO nie jest checkpointem wczytywanym przez LibreYOLO za pomocą
`LibreYOLO()`. Korzysta z pokrewnej fabryki `LibreOpenVocab`, która przy
pierwszym użyciu pobiera snapshot Hugging Face i zapisuje go w pamięci
podręcznej w `weights/`.

<code-tabs name="predict" />

`set_classes()` ustawia trwały słownik tekstowy. Ponowne wywołanie zastępuje
listę, a pominięcie zachowuje domyślne etykiety COCO-80. Grounding DINO dekoduje
swobodne frazy z własnego wyjścia tekstowego i samodzielnie mapuje je z powrotem
na ten słownik. Wygrywa dokładne znormalizowane dopasowanie, akceptowane jest
dopasowanie całego tokenu, a niejednoznaczna lub niedopasowana fraza jest
odrzucana zamiast zgadywana. Dlatego `school bus` nigdy nie jest mapowane
wyłącznie na `bus` ani `school`. Słownik na tyle długi, że przekracza limit
tokenów enkodera tekstowego, jest dzielony na kilka promptów uruchamianych w
osobnych przebiegach w przód i ponownie łączonych w jeden zbiór detekcji
ograniczony przez `max_det`.

Argument `iou` jest przyjmowany dla zgodności API, ale powoduje ostrzeżenie i
nic nie robi, ponieważ nie jest tu uruchamiane tłumienie niemaksymalne.
Argumenty `imgsz` i `augment=True` są odrzucane. Procesor `transformers` sam
kontroluje zmianę rozmiaru, a augmentacja podczas testowania pozostaje poza
zakresem tego poziomu. `predict()` dla jednego obrazu zwraca jeden obiekt
`Results`, a nie listę. Katalog, listę obrazów lub `stream=True` ze źródłem
wideo należy przekazać, aby uzyskać kilka wyników. Dla tej rodziny nie ma
ścieżki CLI. `libreyolo predict` wczytuje przez `LibreYOLO()` wyłącznie
checkpointy `.pt`, dlatego rodziny `LibreOpenVocab` uruchamia się z Pythona.
Informacje o typach źródeł i streamingu zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępne są dwa checkpointy, `t` i `b`. `t` jest domyślnym rozmiarem tego
poziomu, gdy nie podano żadnego. Oba tworzą kopię lustrzaną oficjalnego wydania
IDEA Research za pośrednictwem `GroundingDinoForObjectDetection` z
`transformers`, pobieraną jednorazowo do hostowanego przez LibreYOLO snapshotu
Hugging Face, który zachowuje pliki źródłowe. Nie opublikowano jeszcze wyników
dokładności ani opóźnienia tej rodziny.

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
