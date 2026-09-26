---
title: SenseNova-Vision
families:
  - sensenovavision
seo_title: 'SenseNova-Vision w LibreYOLO: 7 zadań, jeden checkpoint'
description: >-
  Używaj SenseNova-Vision w LibreYOLO do detekcji, segmentacji, segmentacji
  panoptycznej, estymacji pozy, punktów, głębi i OCR z jednego sterowanego
  promptami checkpointu generatywnego.
lead: >-
  SenseNova-Vision to zunifikowany model multimodalny, który przedstawia zadania
  wizyjne jako generowanie sterowane promptami we wspólnym dekoderze: ramki,
  punkty, punkty kluczowe i słowa OCR powstają jako oznakowany tekst, a mapy
  głębi, masek i segmentacji panoptycznej jako obrazy renderowane przez dekoder.
  LibreYOLO wczytuje go przez LibreVLM i obsługuje siedem zadań z jednego
  checkpointu 7B.
keywords:
  - SenseNova-Vision
  - SenseTime
  - zunifikowany model multimodalny
  - Bagel
  - detekcja sterowana promptami
  - gęsta percepcja
  - segmentacja referencyjna
  - segmentacja panoptyczna
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="detect")
        model.set_classes(["bird", "boat"])
        result = model.predict("image.jpg")
        print(result.boxes.xyxy)

        # set_task() przełącza zadania w tym samym wczytanym modelu.
        model.set_task("depth")
        result = model.predict("image.jpg")
        depth = result.depth_map.data
    - label: Segmentacja referencyjna i panoptyczna
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("sensenova-vision", task="segment")

        # Segmentacja jest referencyjna: wymaga frazy docelowej, a nie listy
        klas.

        model.set_classes(["the person furthest to the right"])

        result = model.predict("street.jpg")

        mask = result.masks.data[0]


        model.set_task("panoptic")

        # Bez niestandardowego słownika segmentacja panoptyczna używa kategorii

        # COCO panoptic, do których dostrojono checkpoint.

        result = model.predict("street.jpg")

        segment_map = result.panoptic.data

        for segment in result.panoptic.segments_info:
            print(segment)
    - label: 'Punkty, estymacja pozy i OCR'
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="point")
        model.set_classes(["screw"])
        result = model.predict("board.jpg")
        print(result.points.xy)

        # Bez ustawionego słownika estymacja pozy domyślnie używa "person".
        model.set_task("pose")
        result = model.predict("gym.jpg")
        print(result.boxes.xyxy, result.keypoints.data.shape)

        model.set_task("ocr")
        result = model.predict("sign.jpg")
        print(result.ocr.texts)
source_hash: 8749277e1910baa4
---

## Instalacja

SenseNova-Vision wymaga własnego dodatku, który instaluje `accelerate` do
obsługi dużego modelu wymaganej przez ten checkpoint, a na platformach innych
niż macOS także `bitsandbytes` do wczytywania 4-bitowego.

```bash
pip install "libreyolo[sensenova]"
```

Checkpoint ma kopię lustrzaną na Hugging Face we własnej organizacji LibreYOLO
i jest pobierany automatycznie przy pierwszym użyciu. Podlega licencji CC BY-NC
4.0 wyłącznie do użytku niekomercyjnego, a loader wypisuje tę informację przed
każdym automatycznym pobraniem. Szczegóły zawiera poniższa sekcja Licencja.

## Predykcja

<code-tabs name="predict" />

Każda predykcja jest dekodowaniem dyfuzyjnym na wspólnym backbone Bagel-MoT,
jest to więc model funkcjonalny, a nie czasu rzeczywistego. Należy oczekiwać
wyraźnie większego opóźnienia na obraz niż w specjalizowanym detektorze lub
segmenterze. `dtype="auto"` (wartość domyślna) wczytuje bf16 na GPU z
wystarczającą pamięcią, a w przeciwnym razie używa kwantyzacji 4-bitowej NF4,
która wymaga `bitsandbytes`. Aby wymusić pełną precyzję na odpowiednio dużym
GPU, należy przekazać `dtype="bf16"`. Ustawienie `noise_seed=42` podczas
tworzenia modelu inicjuje sampler dyfuzyjny, zapewniając powtarzalne gęste
wyniki. Przekazanie `noise_seed=None` wyłącza inicjalizację.

Siedem zadań współdzieli jeden wczytany checkpoint. `set_task()` przełącza je
bez ponownego wczytywania. `set_classes()` ustawia aktywny słownik. Detekcja,
punkty, estymacja pozy i segmentacja panoptyczna przyjmują listę klas, natomiast
segmentacja jest referencyjna i wymaga dokładnej frazy opisującej izolowany
element. Każde zadanie zwraca standardowy obiekt `Results` z innym wypełnionym
elementem: `boxes` dla detekcji, `points` dla punktów, `boxes` i `keypoints` dla
estymacji pozy, `ocr` dla OCR, `depth_map` dla głębi, `masks` dla segmentacji
oraz `panoptic` (z `segments_info`) dla segmentacji panoptycznej. Więcej
informacji o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Checkpointy

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
