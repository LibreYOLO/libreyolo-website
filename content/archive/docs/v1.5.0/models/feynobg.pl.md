---
title: FeyNobg
families:
  - feynobg
seo_title: 'FeyNobg: usuwanie tła w LibreYOLO'
description: >-
  Używaj FeyNobg w LibreYOLO do usuwania tła i mattingu alfa, w pogłębionym
  wariancie BiRefNet firmy Feyn Inc. Instaluj, przewiduj i waliduj.
lead: >-
  Model do usuwania tła firmy Feyn Inc., który pogłębia architekturę BiRefNet i
  ponownie ją trenuje. LibreYOLO udostępnia inferencję i walidację dla zadania
  mattingu FeyNobg.
keywords:
  - FeyNobg
  - usuwanie tła
  - dychotomiczna segmentacja obrazu
  - matting alfa
  - wycinanie tła
  - przezroczyste tło
  - nobg
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFeyNobgl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Wycięcie
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: źródłowe RGB oraz matting jako kanał alfa.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreFeyNobgl-matte.pt")


        # Zamiast pliku YAML zbioru danych można użyć katalogu zawierającego
        images/

        # i automatycznie wykrywany katalog mattingu (mattes/, matte/, gt/,

        # masks/, mask/ lub alpha/).

        metrics = model.val(data="my-matte-dataset/")


        print(metrics["metrics/MAE"])

        print(metrics["metrics/Smeasure"])
source_hash: 45de3b578d7ebbf2
---

## Instalacja

FeyNobg nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu checkpoint jest pobierany z organizacji LibreYOLO na
Hugging Face i zapisywany w lokalnej pamięci podręcznej, tak samo jak dla każdej
innej rodziny, choć nie jest jeszcze wymieniony w tabeli checkpointów na tej
stronie.

<code-tabs name="predict" />

Wynik mattingu nie zawiera ramek. `result.matte` to gęsta tablica float32
`(H, W)` z wartościami w zakresie `[0, 1]`, gdzie 1 oznacza pełny pierwszy plan,
a 0 pełne tło. W przeciwieństwie do maski binarnej miękki matting zachowuje
wygładzone szczegóły krawędzi, takie jak włosy i sierść. `result.cutout()` łączy
obraz źródłowy z tym kanałem alfa w tablicę RGBA, a `result.save(path)` (lub
`save=True` w wywołaniu predykcji) zapisuje ją bezpośrednio jako plik PNG z
przezroczystym tłem. Model działa na stałym natywnym obszarze 1024x1024. Inna
rozdzielczość nie jest obsługiwana, ponieważ tablice pozycji względnych backbone
Swin są z nią powiązane, a niedopasowanie powoduje ich nieprawidłową interpolację
zamiast błędu. Więcej informacji o źródłach, streamingu i obsłudze wyników
zawiera strona [predykcji](/docs/predict).

## Warianty

Opublikowano jeden rozmiar, `l`, z backbone poziomu Swin-L. FeyNobg wykorzystuje
architekturę BiRefNet i pogłębia jej trzeci etap Swin z 18 do 24 bloków przed
ponownym trenowaniem. Port LibreYOLO ponownie używa więc ścieżki w przód,
przetwarzania wstępnego i kontraktu wyjścia pojedynczego logitu z BiRefNet.
Predykcja, walidacja i obsługa checkpointów działają tak samo jak w rodzinie
`birefnet`.

## Walidacja

`val()` zwraca dwie metryki dla sparowanego folderu obrazów i mattingu, obie w
zakresie `[0, 1]` i niezależne od rozdzielczości. MAE to średni błąd bezwzględny
względem referencyjnego kanału alfa (mniej znaczy lepiej), a S-measure (Fan i
in., ICCV 2017) to podobieństwo strukturalne nagradzające zachowanie kształtu i
otworów obiektu, których sam pikselowy MAE nie uwzględnia (więcej znaczy lepiej).
Walidacja korzysta z własnej metody `predict` modelu, dlatego używa dokładnie
przetwarzania wstępnego tej rodziny.

<code-tabs name="val" />

Walidacja służy wyłącznie do inferencji. Źródłowa biblioteka `nobg` udostępnia
kod trenowania na licencji Apache-2.0. Obecnie dostrajanie oznacza trenowanie w
tej bibliotece i konwersję wyniku własnym skryptem LibreYOLO, a nie wywołanie
`train()` dla tej rodziny, które zgłasza błąd zamiast uruchamiać niepełnego
trenera.

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
