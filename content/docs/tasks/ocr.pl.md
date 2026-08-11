---
title: OCR
seo_title: 'OCR: detekcja i rozpoznawanie tekstu w LibreYOLO'
description: >-
  Znajduj i odczytuj tekst na obrazach za pomocą LibreYOLO. Przewiduj czworokąty
  i transkrypcje, oznaczaj zbiory danych JSONL oraz waliduj wyniki metrykami
  hmean, kompleksową F1 i 1-NED.
lead: >-
  OCR lokalizuje tekst na obrazie i go odczytuje. LibreYOLO udostępnia tę
  funkcję jako zadanie ocr, które dla każdego obszaru tekstowego zwraca jeden
  wielokąt z czterema punktami i jedną transkrypcję, w kolejności czytania.
keywords:
  - biblioteka ocr python
  - rozpoznawanie tekstu na obrazach
  - detekcja tekstu czworokąty
  - PP-OCRv5 python
  - end-to-end text spotting
last_verified: 1.5.0
snippets:
  predict:
    - label: Odczytywanie tekstu z obrazu
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Wariant t jest lżejszy z dwóch i przeznaczony do CPU. SAMPLE_IMAGE
        # pozwala uruchomić przykład; można też wskazać własny obraz z tekstem.
        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        regions = result.ocr
        print(len(regions), "regions")
        for text, score in zip(regions.texts, regions.conf):
            print(repr(text), float(score))
    - label: Odczytywanie czworokątów
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        regions = result.ocr
        print(regions.data.shape)   # wielokąty (N, 4, 2), LG PG PD LD
        print(regions.xyxy)         # obwiednie wielokątów wyrównane do osi
        print(regions.det_conf)     # wskaźnik detekcji, niezależny od .conf
    - label: Filtrowanie według pewności rozpoznawania
      language: python
      code: |
        import numpy as np
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        # Indeksowanie odbywa się po pozycjach, nie masce logicznej: wycinanie
        # zachowuje transkrypcje i obie tablice wyników razem z geometrią.
        regions = result.ocr.numpy()
        keep = regions[np.flatnonzero(regions.conf >= 0.9)]
        print(keep.texts)
  val:
    - label: Walidacja i odczytywanie kluczy metryk
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        metrics = model.val(data="my-ocr-dataset")

        print(metrics["metrics/det_precision"], metrics["metrics/det_recall"])
        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # fitness
        print(metrics["metrics/rec_1-NED"])
source_hash: 58ad5305c9dd458c
---

## Definicja

Zadanie `ocr` wykonuje dwie czynności w jednym wywołaniu: lokalizuje każdy
obszar tekstowy na obrazie i tworzy jego transkrypcję. Obszary są zwracane jako
wielokąty z czterema punktami, a nie ramki wyrównane do osi, ponieważ tekst na
scenach jest często obrócony. Zachowana jest kolejność czytania, od góry do dołu,
a następnie od lewej do prawej.

Predykcja wypełnia `result.ocr`, czyli strukturę `OCRRegions`. Pole `.data` jest
tablicą zmiennoprzecinkową `(N, 4, 2)` wielokątów we współrzędnych pikselowych
oryginalnego obrazu, w kolejności lewy górny, prawy górny, prawy dolny, lewy
dolny. Pole `.texts` jest listą N transkrypcji, `.conf` to wynik rozpoznawania
dla każdego obszaru, a `.det_conf` to wynik detekcji. Pole `.xyxy` zawiera
wyrównaną do osi obwiednię każdego wielokąta. Ponieważ czworokąty są
rzeczywistymi wielokątami, nie trafiają do `result.boxes`. Wycinanie obiektu
`OCRRegions` zachowuje transkrypcje i obie tablice wyników razem z geometrią.

## Modele

Zadanie `ocr` obsługują dwie rodziny.

[PP-OCRv5](/docs/models/pp-ocrv5) to wyspecjalizowany pipeline: detektor z
różniczkowalną binaryzacją znajduje czworokąty tekstu, a moduł rozpoznawania
SVTR/CTC je odczytuje. Oba etapy wraz z zestawem znaków do rozpoznawania są
umieszczone w jednym pliku `.pt`. Dostępne są dwa warianty, lżejszy do CPU oraz
serwerowy o większej dokładności. Jeden słownik obejmuje chiński uproszczony i
tradycyjny, angielski, japoński oraz pinyin.

[SenseNova-Vision](/docs/models/sensenova-vision) realizuje OCR przez generowanie
słów jako oznaczonego tekstu z tego samego checkpointu 7B, który obsługuje sześć
pozostałych zadań. Wczytuje się go za pomocą
`LibreVLM("sensenova-vision", task="ocr")`. Wymaga dodatku `sensenova`, a jego
wagi są ograniczone do użytku niekomercyjnego. Informacje o licencji znajdują
się na stronie modelu.

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane lokalnie w
pamięci podręcznej.

<code-tabs name="predict" />

PP-OCRv5 uruchamia detekcję ze stałym limitem dłuższego boku, a następnie
rozpoznaje wycięte obszary w batchach. Parametr `rec_batch` określa, ile wycinków
przechodzi przez moduł rozpoznawania w jednym przebiegu. Źródła z wieloma
obrazami są przetwarzane kolejno, ponieważ dwuetapowy pipeline nie tworzy batchy
obejmujących różne obrazy. Informacje o źródłach, streamingu i obsłudze wyników
znajdują się w sekcji [predykcja](/docs/predict).

## Format zbioru danych

Etykiety OCR mają postać jednego pliku JSONL na każdy podział. Zawiera on po
jednym obiekcie JSON na obraz i znajduje się obok samych obrazów.

```text
my-ocr-dataset/
  images/
    val/receipt.jpg
  labels/
    val.jsonl
```

Każdy wiersz wskazuje obraz i wymienia jego obszary:

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` jest czworokątem w bezwzględnych współrzędnych pikselowych, z punktami
w kolejności lewy górny, prawy górny, prawy dolny, lewy dolny. Obszar, którego
tekstu nie można odczytać, otrzymuje etykietę `"text": "###"` zgodnie z
konwencją ignorowania ICDAR. Jest wyłączony z oceny rozpoznawania, a nakładająca
się na niego predykcja jest ignorowana zamiast uznawana za fałszywie dodatnią.

Wystarczy przekazać katalog główny jako `data=`. Alternatywą jest plik YAML
zbioru danych z polem `path`, opcjonalnymi nazwami katalogów `images` i `labels`
oraz `nc: 1` i `names: {0: text}` jako symbolami zastępczymi schematu, ponieważ
model OCR zwraca `Results.ocr`, a nie detekcje. Pełny kontrakt opisano w sekcji
[formaty zbiorów danych](/docs/reference/dataset-formats).

## Trenowanie

Żadna rodzina OCR nie ma implementacji trenowania: `train()` zgłasza
`NotImplementedError` w obu przypadkach, a obsługa OCR obejmuje tylko predykcję
i walidację. Strona PP-OCRv5 wskazuje nadrzędny kod trenowania na licencji
Apache-2.0 oraz skrypt konwersji, który przenosi dostrojony checkpoint z
powrotem do LibreYOLO.

## Walidacja

Metoda `val()` ocenia cały pipeline, jednocześnie detekcję i rozpoznawanie,
dopasowując przewidziane wielokąty jeden do jednego do wielokątów danych
referencyjnych (ground truth) przy IoU powyżej 0.5.

<code-tabs name="val" />

Metryki `metrics/det_precision`, `metrics/det_recall` i `metrics/det_hmean`
oceniają wyłącznie lokalizację. Dopasowanie wymaga jedynie nałożenia się
wielokątów, niezależnie od treści transkrypcji. Metryki `metrics/e2e_precision`,
`metrics/e2e_recall` i `metrics/e2e_f1` uwzględniają również odczyt. Dopasowanie
wymaga takiego samego nałożenia wielokątów i dokładnej zgodności transkrypcji po
normalizacji NFKC oraz usunięciu białych znaków, przy czym porównanie rozróżnia
wielkość liter. `metrics/e2e_f1` jest także wartością `fitness`, używaną przy
wyborze najlepszego checkpointu.

Metryka `metrics/rec_1-NED` ocenia sam moduł rozpoznawania na parach już
dopasowanych przez detekcję. Jest równa jedności pomniejszonej o znormalizowaną
odległość edycyjną, więc transkrypcja różniąca się o jeden znak uzyskuje wynik bliski 1,
podczas gdy kompleksowa F1 przypisuje jej 0.

## Eksport

Dla tego zadania nie jest dostępny żaden format eksportu. PP-OCRv5 składa się z
dwóch współdziałających sieci, a nie jednego grafu możliwego do prześledzenia,
natomiast `export()` zgłasza błąd dla każdego formatu w obu rodzinach. Aby
wdrożyć model poza LibreYOLO, należy dostroić go w projekcie nadrzędnym i użyć
nadrzędnej ścieżki wdrożenia.
