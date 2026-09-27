---
title: PP-OCRv5
families:
  - ppocr
seo_title: 'PP-OCRv5: detekcja i rozpoznawanie tekstu w LibreYOLO'
description: >-
  Używaj PP-OCRv5 w LibreYOLO do wielojęzycznego OCR tekstu scen. Instaluj,
  przewiduj i waliduj checkpointy t i l na licencji Apache-2.0.
lead: >-
  PP-OCRv5 to pipeline detekcji i rozpoznawania tekstu z PaddleOCR: detektor z
  różniczkowalną binaryzacją lokalizuje czworokąty tekstu, a rozpoznawanie
  SVTR/CTC je odczytuje. LibreYOLO przenosi go do PyTorch w dwóch poziomach.
keywords:
  - PP-OCRv5
  - PaddleOCR
  - OCR
  - detekcja tekstu
  - rozpoznawanie tekstu
  - tekst na zdjęciu
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for text, conf in zip(result.ocr.texts, result.ocr.conf):
            print(text, float(conf))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePPOCRl-ocr.pt source=receipt.jpg save=True
    - label: Czworokąty
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePPOCRl-ocr.pt")

        result = model(SAMPLE_IMAGE)


        # Wielokąty (N, 4, 2) w kolejności odczytu: lewy górny, prawy górny,

        # prawy dolny, lewy dolny. Czworokąty detekcji są prawdziwymi
        wielokątami

        # (obrócony tekst), dlatego trafiają do result.ocr, a nie result.boxes.

        print(result.ocr.data.shape)

        print(result.ocr.det_conf)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        metrics = model.val(data="my-dataset")

        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # główna metryka
        print(metrics["metrics/rec_1-NED"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePPOCRl-ocr.pt data=my-dataset
source_hash: 9835057f8bd95bc1
---

## Instalacja

PP-OCRv5 nie wymaga żadnego dodatku poza pakietem podstawowym.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Każdy checkpoint zawiera oba etapy, detekcję i rozpoznawanie, w jednym pliku
`.pt`, a zestaw znaków rozpoznawania i wartości domyślne pipeline'u są zapisane
w metadanych checkpointu. Rozpoznawanie odczytuje jednym słownikiem chiński
uproszczony i tradycyjny, angielski, japoński oraz pinyin. `result.ocr` jest
elementem `OCRRegions`: `.data` zawiera czteropunktowe wielokąty, `.texts`
transkrypcje, `.conf` wynik rozpoznawania dla każdego regionu, a `.det_conf`
wynik detekcji. Źródła z wieloma obrazami działają sekwencyjnie. Dwustopniowy
pipeline nie tworzy batcha z wielu obrazów. Więcej informacji o źródłach,
streamingu i obsłudze wyników zawiera strona [predykcji](/docs/predict).

## Warianty

Dostępne są dwa poziomy: `t`, oparty na lżejszych backbone
PP-LCNetV3/PP-OCRv5_mobile do zastosowań CPU, oraz `l`, oparty na serwerowych
backbone PP-HGNetV2 zapewniających większą dokładność. Oba poziomy uruchamiają
detekcję ze stałym limitem dłuższego boku i rozpoznają przycięcia w batchach.
`rec_batch` określa liczbę przycięć przekazywanych do rozpoznawania w jednym
przebiegu w przód.

## Walidacja

`val()` mierzy pipeline względem katalogu obrazów z plikiem
`labels/<split>.jsonl` lub równoważnego pliku YAML zbioru danych. Każda etykieta
wymienia wielokąty regionów tekstowych obrazu i ich transkrypcje. Zwracane są:
hmean detekcji (precision/recall/F1 dopasowane przez IoU), F1 od początku do
końca (hmean połączone z dokładnym dopasowaniem znormalizowanej transkrypcji,
metryka fitness checkpointu) oraz 1-NED, czyli średnia znormalizowana odległość
edycyjna dla dopasowanych par.

<code-tabs name="val" />

## Eksport

<export-matrix />

PP-OCRv5 jest pipelinem dwóch sieci, detekcji i rozpoznawania działających
wspólnie, a nie pojedynczym grafem możliwym do śledzenia. Eksport nie jest dla
niego zaimplementowany i nie jest jeszcze obsługiwany żaden format. Jeśli
potrzebny jest checkpoint poza tym formatem, należy bezpośrednio dostroić kod
trenowania projektu źródłowego na licencji Apache-2.0 i przekonwertować wynik za
pomocą `weights/convert_ppocr_weights.py`.

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
