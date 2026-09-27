---
title: SAM 2
families:
  - sam2
seo_title: 'SAM 2: segmentacja obrazów sterowana promptami w LibreYOLO'
description: >-
  Używaj SAM 2 w LibreYOLO do segmentacji sterowanej punktami i ramkami.
  Instaluj i uruchamiaj predykcję z checkpointami tiny, small, base-plus i large
  na licencji Apache-2.0.
lead: >-
  SAM 2 rozszerza SAM o architekturę pamięci strumieniowej zbudowaną do wideo
  oraz zamienia kliknięcie punktu lub ramki w maskę obiektu. LibreYOLO obsługuje
  jego ścieżkę segmentacji obrazów przez osobną fabrykę LibreSAM, niezależną od
  fabryki detektorów LibreYOLO().
keywords:
  - SAM 2
  - Segment Anything
  - segmentacja sterowana promptami
  - segmentacja interaktywna
  - prompt punktowy
  - prompt ramką
  - Meta AI
  - Hiera
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompty punktowe i ramkowe
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # Aliasy rozmiarów: "sam2-tiny", "sam2-small", "sam2-base-plus",

        # "sam2-large" (także krótkie formy
        "sam2-t"/"sam2-s"/"sam2-bp"/"sam2-l").

        model = LibreSAM("sam2-large")


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
        from libreyolo import LibreSAM2, SAMPLE_IMAGE


        # Klasa właściwa dla rodziny przyjmuje rozmiar bez prefiksu "sam2-".

        model = LibreSAM2("large")


        # Enkoder obrazu jest kosztowną częścią. set_image() uruchamia go raz,

        # a każde kolejne wywołanie predict() ponownie wykorzystuje embedding z
        pamięci podręcznej.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: 2a3090d7ecd533b0
---

## Instalacja

SAM 2 wymaga dodatku `sam`, który instaluje zależności `transformers` i `timm`.

```bash
pip install "libreyolo[sam]"
```

## Predykcja

`LibreSAM(...)` (lub właściwy dla tej rodziny `LibreSAM2(...)`) jest osobnym
punktem wejścia niż `LibreYOLO(...)`. Zwraca segmenter sterowany promptami, a
nie detektor, ponieważ przebieg w przód nie ma tu znaczenia bez promptu
przestrzennego. Dla tej rodziny nie istnieje polecenie CLI `libreyolo predict`.
Należy użyć API Pythona. Obsługiwana jest tylko segmentacja obrazów. Śledzenie
z pamięcią wideo w SAM 2 pozostaje poza zakresem tej implementacji.

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
zgłaszają dla tej rodziny `NotImplementedError`. LibreYOLO obsługuje tu
inferencję obrazów. Informacje o typach źródeł zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępne są cztery rozmiary z backbone Hiera: tiny, small, base-plus i large,
wszystkie przy tej samej rozdzielczości wejściowej. Nie opublikowano jeszcze
benchmarku dokładności ani opóźnienia tej rodziny, dlatego wybór rozmiaru
bezpośrednio wymienia ciężar enkodera na jakość maski. Tiny koduje najszybciej,
a large jest najcięższy.

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
