---
title: ZipDepth
families:
  - zipdepth
seo_title: 'ZipDepth: lekka głębia monokularna w LibreYOLO'
description: >-
  Używaj ZipDepth w LibreYOLO do lekkiej estymacji głębi monokularnej. Instaluj,
  przewiduj, waliduj i eksportuj dwa checkpointy na licencji MIT.
lead: >-
  ZipDepth to kompaktowa, reparametryzowalna sieć CNN destylowana z Depth
  Anything V2 Large, która przewiduje gęstą mapę względnej odwrotności głębi.
  LibreYOLO obsługuje ją w zadaniu estymacji głębi: do predykcji i walidacji
  zero-shot, bez ścieżki trenowania.
keywords:
  - ZipDepth
  - estymacja głębi monokularnej
  - model głębi na urządzenie brzegowe
  - głębia względna
  - mapa głębi
  - reparametryzowalna CNN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreZipDepthb-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Checkpoint NPU/edge
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Ten sam enkoder z głowicą skalowania bez unfold dla kompilatorów bez
        obsługi

        # gather/unfold. Wynik jest wizualnie równoważny checkpointowi b.

        model = LibreYOLO("LibreZipDepthbnpu-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreZipDepthb-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        model.export(format="onnx")
        model.export(format="ncnn")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreZipDepthb-depth.pt format=onnx
        libreyolo export model=LibreZipDepthbnpu-depth.pt format=ncnn
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreZipDepthb-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: 891eaa1a42795a4c
---

## Instalacja

ZipDepth nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

`result.depth_map` zawiera gęstą mapę względnej odwrotności głębi: wyższe
wartości oznaczają mniejszą odległość od kamery, a wartości nie mają jednostki
metrycznej ani skali wspólnej dla obrazów. Ustawienie `save=True` zapisuje na
dysku wizualizację tej mapy z nałożoną paletą kolorów. `Results.plot()` nie
obsługuje tej rodziny, ponieważ zdefiniowano go tylko dla normalnych powierzchni
i krawędzi. Więcej informacji o źródłach, streamingu i obsłudze wyników zawiera
strona [predykcji](/docs/predict).

## Warianty

Dwa checkpointy mają tę samą pojemność enkodera, a różnią się tylko wytrenowaną
głowicą skalowania. `b` używa skalowania wypukłego i działa na GPU lub CPU.
`bnpu` zastępuje je dekoderem bez unfold dla NPU i kompilatorów brzegowych bez
obsługi gather/unfold. Jego wynik jest opisany jako wizualnie równoważny `b`.
`bnpu` należy wybrać dla ograniczonego docelowego środowiska eksportu, a w
pozostałych przypadkach `b`.

Oba checkpointy destylowano z pseudoetykiet Depth Anything V2 Large. Ta rodzina
jest zatem kompaktowym poziomem zadania głębi LibreYOLO przeznaczonym do urządzeń
brzegowych, obok większych enkoderów Depth Anything V2.

Ta rodzina nie obsługuje trenowania. `LibreZipDepth.train()` zawsze zgłasza
`NotImplementedError`. Procedura źródłowa destyluje pseudoetykiety na dużym
zbiorze obrazów, czego nie można odtworzyć jako przebiegu trenowania LibreYOLO.
Model należy wytrenować w projekcie źródłowym
[fabiotosi92/ZipDepth](https://github.com/fabiotosi92/ZipDepth), a wynik
przekonwertować za pomocą `weights/convert_zipdepth_weights.py`.

## Walidacja

`val()` uruchamia wspólny walidator głębi. Dopasowuje każdą predykcję do jej
danych referencyjnych (ground truth), obliczając dla każdego obrazu skalę i
przesunięcie metodą najmniejszych kwadratów, po czym zwraca standardowe metryki
głębi względnej zero-shot: AbsRel, RMSE i trzy progi delta.

<code-tabs name="val" />

## Eksport

<export-matrix />

Eksport korzysta ze stałego kontraktu gęstej rozdzielczości. Obraz źródłowy
jest rozciągany do wyeksportowanego obszaru, a zwrócona mapa głębi jest potem
skalowana do oryginalnego obszaru. Wyeksportowany artefakt jest ponownie
ładowany przez `LibreYOLO()` na podstawie rozszerzenia pliku, dlatego plik
`.onnx` lub `.ncnn` zachowuje się jak checkpoint i zwraca ten sam obiekt
`Results`, z `depth_map` zamiast ramek.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
