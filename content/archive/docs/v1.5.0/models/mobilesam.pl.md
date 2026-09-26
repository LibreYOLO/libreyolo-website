---
title: MobileSAM
families:
  - mobilesam
seo_title: 'MobileSAM: lekka segmentacja sterowana promptami w LibreYOLO'
description: >-
  Używaj MobileSAM w LibreYOLO do segmentacji sterowanej punktami i ramkami z
  enkoderem TinyViT. Zainstaluj model i uruchamiaj predykcję z małym
  checkpointem na licencji Apache-2.0.
lead: >-
  MobileSAM zastępuje enkoder obrazu ViT-H modelu SAM destylowanym enkoderem
  TinyViT, dzięki czemu ten sam proces sterowany promptami punktowymi i ramkami
  działa na lżejszym sprzęcie. LibreYOLO udostępnia jego natywny port przez
  osobną fabrykę LibreSAM, niezależną od fabryki detektorów LibreYOLO().
keywords:
  - MobileSAM
  - Segment Anything
  - TinyViT
  - segmentacja sterowana promptami
  - segmentacja interaktywna
  - prompt punktowy
  - prompt ramką
  - lekka segmentacja
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompty punktowe i ramkowe
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # MobileSAM ma jeden rozmiar, "tiny", dlatego nie jest potrzebny inny
        alias.

        model = LibreSAM("mobilesam")


        # Prompt punktowy: [x, y] we współrzędnych pikseli, etykieta 1 =
        pierwszy plan.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # wielokąt dla każdej maski

        print(result.boxes.xyxy)    # ciasna ramka wyznaczona z maski


        # Prompt ramkowy zamiast punktowego.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])


        # Brak promptu segmentuje cały obraz (uproszczony automatyczny

        # generator masek, a nie pełny generator referencyjny).

        result = model.predict(SAMPLE_IMAGE)
    - label: 'Jedno kodowanie, wiele promptów'
      language: python
      code: >
        from libreyolo import LibreMobileSAM, SAMPLE_IMAGE


        model = LibreMobileSAM()


        # Enkoder obrazu jest kosztowną częścią. set_image() uruchamia go raz,

        # a każde kolejne wywołanie predict() ponownie wykorzystuje embedding z
        pamięci podręcznej.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: f96e885d93f72bdd
---

## Instalacja

MobileSAM wymaga dodatku `sam`. Własne pobieranie wag LibreYOLO nadal korzysta
z narzędzi snapshotów Hugging Face w `transformers`, mimo że inferencja działa
na natywnym dekoderze niezależnym od `transformers`.

```bash
pip install "libreyolo[sam]"
```

## Predykcja

`LibreSAM(...)` (lub właściwy dla tej rodziny `LibreMobileSAM(...)`) jest
osobnym punktem wejścia niż `LibreYOLO(...)`. Zwraca segmenter sterowany
promptami, a nie detektor, ponieważ przebieg w przód nie ma tu znaczenia bez
promptu przestrzennego. Dla tej rodziny nie istnieje polecenie CLI
`libreyolo predict`. Należy użyć API Pythona.

<code-tabs name="predict" />

Prompt punktowy przyjmuje `[x, y]` dla jednego obiektu, `[[x, y], ...]` dla
kilku obiektów lub tablice numpy. Argument `labels` oznacza każdy punkt jako `1`
(pierwszy plan) lub `0` (tło), a domyślnie wszystkie punkty należą do pierwszego
planu. Prompt ramkowy przyjmuje `[x1, y1, x2, y2]` albo listę ramek i zwraca
jedną maskę dla każdej ramki. Pominięcie obu promptów segmentuje cały obraz za
pomocą gęstej siatki promptów, zachowując pewne i nienakładające się maski. Ten
tryb „segmentuj wszystko” jest uproszczony względem referencyjnego automatycznego
generatora masek i może niedostatecznie segmentować zatłoczone sceny, dlatego
prawdziwy prompt punktowy lub ramkowy zapewnia precyzyjniejszą ścieżkę. Argument
`conf` filtruje według przewidywanej jakości maski (IoU), a nie pewności
detekcji. Przekazanie `0.0` zachowuje każdego kandydata. Ustawienie
`multimask=True` zwraca dla każdego promptu wszystkie trzy maski SAM
odpowiadające niejednoznaczności całość względem części, zamiast tylko jednej
najlepszej. Argument `device=` przenosi model oraz, jeśli sesja `set_image()`
jest aktywna, jej embedding z pamięci podręcznej. Każda maska ma identyfikator
klasy `0` o nazwie `"object"`, ponieważ maska sterowana promptem nie ma stałego
zbioru klas. Wywołania `train()`, `val()`, `export()` i `track()` zawsze
zgłaszają dla tej rodziny `NotImplementedError`. MobileSAM służy w LibreYOLO
wyłącznie do predykcji. Informacje o typach źródeł zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępny jest jeden rozmiar, tiny, ze stałym wejściem 1024 px. MobileSAM
udostępnia pojedynczy enkoder TinyViT zamiast poziomów base/large/huge z SAM-1.

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
