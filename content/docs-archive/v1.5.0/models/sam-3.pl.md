---
title: SAM 3
families:
  - sam3
seo_title: 'SAM 3: segmentacja sterowana promptami i pojęciami w LibreYOLO'
description: >-
  Używaj SAM 3 w LibreYOLO do segmentacji sterowanej punktami, ramkami i
  pojęciami tekstowymi. Instaluj i uruchamiaj predykcję z checkpointem large,
  kontrolowanym licencją SAM License firmy Meta.
lead: >-
  SAM 3 rozszerza SAM o prompt pojęcia tekstowego obok zwykłych punktów i ramek,
  dzięki czemu fraza taka jak „yellow school bus” zwraca każdą pasującą
  instancję. LibreYOLO obsługuje jego ścieżkę obrazową przez osobną fabrykę
  LibreSAM, niezależną od fabryki detektorów LibreYOLO().
keywords:
  - SAM 3
  - Segment Anything
  - segmentacja sterowana promptami
  - segmentacja pojęć
  - prompt tekstowy
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


        # "sam3" to jedyny rozmiar ("large"). Aliasy: "sam3", "sam-3",
        "sam3-large".

        model = LibreSAM("sam3")


        # Prompt punktowy: [x, y] we współrzędnych pikseli, etykieta 1 =
        pierwszy plan.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # wielokąt dla każdej maski

        print(result.boxes.xyxy)    # ciasna ramka wyznaczona z maski


        # Prompt ramkowy zamiast punktowego.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: Prompt tekstowy (pojęcie)
      language: python
      code: |
        from libreyolo import LibreSAM3, SAMPLE_IMAGE

        model = LibreSAM3("large")

        # Znajduje każdą instancję pasującą do frazy, a nie tylko jeden obiekt.
        # text= wzajemnie wyklucza się z points, bboxes, labels i masks.
        result = model.predict(SAMPLE_IMAGE, text="a person")
        print(result.names)         # {0: "a person"}
        print(result.boxes.conf)    # wynik detekcji PCS dla każdej instancji
    - label: 'Jedno kodowanie, wiele promptów'
      language: python
      code: >
        from libreyolo import LibreSAM3, SAMPLE_IMAGE


        model = LibreSAM3("large")


        # Enkoder obrazu jest kosztowną częścią. set_image() uruchamia go raz,

        # a każde kolejne wywołanie predict() ponownie wykorzystuje embedding z
        pamięci podręcznej.

        # Wywołanie text= ponownie koduje dane wewnętrznie, ponieważ tracker i
        enkoder

        # segmentacji pojęć nie współdzielą pamięci podręcznej.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: c4fb6d5a622f99ff
---

## Instalacja

SAM 3 wymaga dodatku `sam`, który instaluje zależności `transformers` i `timm`.

```bash
pip install "libreyolo[sam]"
```

Dostęp do wag jest kontrolowany. Należy odwiedzić stronę
[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), zaakceptować
licencję SAM License firmy Meta, a następnie uruchomić `hf auth login` (lub
ustawić `HF_TOKEN`) przed pierwszym pobraniem. LibreYOLO wypisuje informację o
licencji przy pierwszym pobraniu tej rodziny.

## Predykcja

`LibreSAM(...)` (lub właściwy dla tej rodziny `LibreSAM3(...)`) jest osobnym
punktem wejścia niż `LibreYOLO(...)`. Zwraca segmenter sterowany promptami, a
nie detektor, ponieważ przebieg w przód nie ma tu znaczenia bez promptu. Dla tej
rodziny nie istnieje polecenie CLI `libreyolo predict`. Należy użyć API Pythona.
Obsługiwana jest tylko inferencja obrazów. Modele wideo SAM 3 pozostają poza
zakresem tej implementacji.

<code-tabs name="predict" />

Ścieżka punktów i ramek odpowiada pozostałym modelom rodziny SAM. Prompt
punktowy przyjmuje `[x, y]` dla jednego obiektu lub `[[x, y], ...]` dla kilku,
`labels` oznacza każdy punkt jako `1` (pierwszy plan) lub `0` (tło), a prompt
ramkowy przyjmuje `[x1, y1, x2, y2]` lub listę ramek. `conf` w tej ścieżce
filtruje według przewidywanej jakości maski (IoU), a nie pewności detekcji.

Ścieżka `text=` jest rozszerzeniem SAM 3. Ciąg pojęcia zwraca każdą pasującą
instancję na obrazie przez Promptable Concept Segmentation i nie może być
łączony z punktami, ramkami, etykietami ani maskami. `conf` oznacza w niej wynik
detekcji PCS zamiast IoU maski. Pozostawienie wartości domyślnej stosuje własny
próg modelu 0.3, a `conf=0.0` zachowuje każdego kandydata. Zwracane `names`
mapuje identyfikator klasy `0` na żądany ciąg pojęcia, ponieważ maska sterowana
promptem nie ma innego stałego zbioru klas. Argument `device=` przenosi model
oraz, jeśli sesja `set_image()` jest aktywna, jej embedding z pamięci
podręcznej. Wywołania `train()`, `val()`, `export()` i `track()` zawsze
zgłaszają dla tej rodziny `NotImplementedError`. SAM 3 służy w LibreYOLO
wyłącznie do predykcji, a śledzenie wideo pozostaje poza zakresem. Informacje o
typach źródeł zawiera strona [predykcji](/docs/predict).

## Warianty

Dostępny jest jeden rozmiar, large, ze stałym wejściem 1008 px. SAM 3.1 nie
jest obsługiwany. Jego implementacja podlega niestandardowej licencji, która nie
może zostać dołączona do tego repozytorium MIT, a wersja Transformers, od której
zależy LibreYOLO, nie wczytuje jeszcze formatu jego checkpointu.

## Licencja

<provenance-box>

LibreYOLO nie hostuje własnej kopii wag SAM 3 ani ich nie redystrybuuje.
`LibreSAM("sam3")` pobiera je bezpośrednio z kontrolowanego repozytorium Meta
`facebook/sam3` na Hugging Face, które wymaga zaakceptowania licencji SAM
License firmy Meta oraz uwierzytelnienia przed pierwszym pobraniem.

</provenance-box>

## Cytowanie

<citation-block />
