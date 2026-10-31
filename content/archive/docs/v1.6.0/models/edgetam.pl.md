---
title: EdgeTAM
families:
  - edgetam
seo_title: 'EdgeTAM: segmentacja sterowana promptami na urządzeniu w LibreYOLO'
description: >-
  Używaj EdgeTAM w LibreYOLO do szybkiej segmentacji na urządzeniu, sterowanej
  punktami i ramkami. Zainstaluj model i uruchom predykcję z checkpointu na
  licencji Apache-2.0.
lead: >-
  EdgeTAM to wariant SAM 2 przeznaczony do pracy na urządzeniu. Zapewnia szybką
  inferencję mobilną, zachowując ten sam pipeline segmentacji sterowanej
  punktami i ramkami. LibreYOLO obsługuje jego ścieżkę segmentacji obrazów przez
  osobną fabrykę LibreSAM, niezależną od fabryki detektorów LibreYOLO().
keywords:
  - EdgeTAM
  - SAM 2
  - segmentacja sterowana promptami
  - segmentacja interaktywna
  - segmentacja na urządzeniu
  - prompt punktowy
  - prompt ramki
  - Meta Reality Labs
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompty punktowe i ramkowe
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # EdgeTAM ma jeden rozmiar, "edge". Aliasy: "edgetam", "edge-tam",

        # "edgetam-edge".

        model = LibreSAM("edgetam")


        # Prompt punktowy: [x, y] we współrzędnych pikselowych, etykieta 1 =
        pierwszy plan.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # wielokąt dla każdej maski

        print(result.boxes.xyxy)    # ciasna ramka wyznaczona z maski


        # Prompt ramkowy zamiast punktowego.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])


        # Brak promptu powoduje segmentację całego obrazu (uproszczony
        automatyczny

        # generator masek, a nie pełna implementacja referencyjna).

        result = model.predict(SAMPLE_IMAGE)
    - label: 'Jedno kodowanie, wiele promptów'
      language: python
      code: >
        from libreyolo import LibreEdgeTAM, SAMPLE_IMAGE


        model = LibreEdgeTAM()


        # Koder obrazu jest kosztowną częścią. set_image() uruchamia go raz;

        # każde późniejsze wywołanie predict() korzysta z embeddingu w pamięci
        podręcznej.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: e6cce8faad18e73d
---

## Instalacja

EdgeTAM wymaga dodatku `sam`, który instaluje pakiety `transformers` i `timm`.

```bash
pip install "libreyolo[sam]"
```

## Predykcja

`LibreSAM(...)` (lub właściwa dla tej rodziny klasa `LibreEdgeTAM(...)`) jest
osobnym punktem wejścia niż `LibreYOLO(...)`. Zwraca segmenter sterowany
promptami, a nie detektor, ponieważ przejście w przód nie ma tutaj znaczenia bez
promptu przestrzennego. Dla tej rodziny nie ma polecenia CLI
`libreyolo predict`; należy użyć API Pythona. Obsługiwana jest tylko segmentacja
obrazów. Śledzenie wideo z EdgeTAM nie jest tu dostępne.

<code-tabs name="predict" />

Prompt punktowy przyjmuje `[x, y]` dla jednego obiektu, `[[x, y], ...]` dla
kilku obiektów albo tablice numpy. Argument `labels` oznacza każdy punkt jako
`1` (pierwszy plan) lub `0` (tło), a domyślnie wszystkie punkty należą do
pierwszego planu. Prompt ramkowy przyjmuje `[x1, y1, x2, y2]` albo listę ramek
i zwraca po jednej masce na ramkę. Pominięcie obu promptów powoduje segmentację
całego obrazu przez użycie gęstej siatki promptów i zachowanie pewnych,
niezachodzących na siebie masek. Ten tryb „segmentuj wszystko” jest uproszczony
względem referencyjnego automatycznego generatora masek i może dzielić zatłoczone
sceny na zbyt małą liczbę segmentów, dlatego dokładną ścieżką jest prawdziwy
prompt punktowy lub ramkowy. Argument `conf` filtruje według przewidywanej
jakości maski (IoU), a nie pewności detekcji. Wartość `0.0` zachowuje wszystkich
kandydatów. Ustawienie `multimask=True` zwraca dla każdego promptu wszystkie
trzy maski SAM reprezentujące niejednoznaczność całość-część, zamiast tylko
najlepszej. Argument `device=` przenosi model, a podczas aktywnej sesji
`set_image()` również embedding zapisany w pamięci podręcznej. Każda maska ma
identyfikator klasy `0` o nazwie `"object"`, ponieważ maska sterowana promptem
nie ma stałego zestawu klas. Metody `train()`, `val()`, `export()` i `track()`
zgłaszają `NotImplementedError` dla tej rodziny. LibreYOLO obsługuje tutaj
inferencję obrazów. Typy źródeł opisano na stronie
[predykcji](/docs/predict).

## Warianty

Dostępny jest jeden rozmiar, edge, ze stałą rozdzielczością wejściową. Wybór
tej rodziny zamiast pozostałych modeli z grupy SAM jest więc decyzją sprzętową,
a nie wyborem rozmiaru. EdgeTAM zaprojektowano specjalnie do inferencji na
urządzeniach o ograniczonych zasobach.

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
