---
title: SAM
families:
  - sam
seo_title: 'SAM (Segment Anything): predykcja masek w LibreYOLO'
description: >-
  Używaj SAM w LibreYOLO do segmentacji sterowanej punktami i ramkami. Instaluj
  i uruchamiaj predykcję z checkpointami base, large i huge na licencji
  Apache-2.0.
lead: >-
  SAM (Segment Anything) zamienia kliknięcie punktu lub ramki w maskę obiektu.
  LibreYOLO wczytuje go przez osobną fabrykę LibreSAM, niezależną od fabryki
  detektorów LibreYOLO(), ponieważ model sterowany promptami wymaga innego
  sposobu wywołania.
keywords:
  - SAM
  - Segment Anything
  - segmentacja sterowana promptami
  - segmentacja interaktywna
  - prompt punktowy
  - prompt ramką
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompty punktowe i ramkowe
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # "base" automatycznie pobiera facebook/sam-vit-base przy pierwszym
        użyciu.

        # Inne rozmiary: "large", "huge" (także "b"/"l"/"h").

        model = LibreSAM("base")


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
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # Enkoder obrazu jest kosztowną częścią. set_image() uruchamia go raz,

        # a każde kolejne wywołanie predict() ponownie wykorzystuje embedding z
        pamięci podręcznej.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: f8904d241ef8a929
---

## Instalacja

SAM wymaga dodatku `sam`, który instaluje zależności `transformers` i `timm`.

```bash
pip install "libreyolo[sam]"
```

## Predykcja

`LibreSAM(...)` jest osobnym punktem wejścia niż `LibreYOLO(...)`. Zwraca
segmenter sterowany promptami, a nie detektor, ponieważ przebieg w przód nie ma
tu znaczenia bez promptu przestrzennego. Dla tej rodziny nie istnieje polecenie
CLI `libreyolo predict`. Należy użyć API Pythona.

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
zgłaszają dla tej rodziny `NotImplementedError`. SAM służy w LibreYOLO wyłącznie
do predykcji, a śledzenie wideo pozostaje poza zakresem. Informacje o typach
źródeł zawiera strona [predykcji](/docs/predict).

## Warianty

Dostępne są trzy rozmiary enkodera obrazu ViT: base, large i huge, wszystkie ze
stałym wejściem 1024 px. Nie opublikowano jeszcze benchmarku dokładności ani
opóźnienia tej rodziny, dlatego wybór rozmiaru bezpośrednio wymienia ciężar
enkodera na jakość maski. Base koduje najszybciej, a huge jest najcięższy.

## Licencja

<provenance-box>

LibreYOLO nie hostuje własnej kopii wag SAM-1. `LibreSAM("base")`, `"large"` i
`"huge"` pobierają je bezpośrednio z własnych repozytoriów Meta
`facebook/sam-vit-base`, `facebook/sam-vit-large` i `facebook/sam-vit-huge` na
Hugging Face. Każde z nich jest tam oznaczone licencją Apache-2.0 niezależnie od
LibreYOLO.

</provenance-box>

## Cytowanie

<citation-block />
