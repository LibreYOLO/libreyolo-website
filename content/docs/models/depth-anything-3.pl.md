---
title: Depth Anything 3
families:
  - depth_anything3
seo_title: 'Depth Anything 3: przewiduj głębię monokularną w LibreYOLO'
description: >-
  Użyj Depth Anything 3 w LibreYOLO do monocularnej estymacji głębokości.
  Zainstaluj, przewiduj, zweryfikuj i wyeksportuj checkpoint DA3MONO-LARGE,
  Apache-2.0.
lead: >-
  Depth Anything 3 to zwykły transformator DINOv2 wytrenowany do przewidywania
  głębokości i geometrii kamery z jednego lub więcej widoków, bez specjalizacji
  architektonicznej. LibreYOLO przenosi swój checkpoint DA3MONO-LARGE dla
  zadania głębokości: predykcja i walidacja zero-shot, bez ścieżki szkoleniowej.
keywords:
  - Depth Anything 3
  - DA3
  - jednooczna estymacja głębokości
  - DINOv2
  - wzglębna głębokość
  - mapa głębi
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnything3l-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Odczytaj mapę głębi
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnything3l-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map    # DepthMap: gęste (H, W), wyższe = bliżej

        raw = depth.data                # tensor, brak jednostki miary lub skali
        między obrazami

        normalized = depth.normalized() # przeskalowano do [0, 1] w celu
        wizualizacji
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnything3l-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnything3l-depth.pt format=onnx

        libreyolo export model=LibreDepthAnything3l-depth.pt format=tensorrt
        half=True
    - label: Użyj wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka kieruje na podstawie sufiksu pliku, więc eksportowany artefakt
        się ładuje

        # jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreDepthAnything3l-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: 0ac96180165c4891
---
## Instalacja

Depth Anything 3 nie potrzebuje żadnych dodatkowych opcji. Wszystko, co importuje, znajduje się w instalacji bazowej.

```bash
pip install libreyolo
```

## Predykcja

Wagi są pobierane z Hugging Face przy pierwszym użyciu i są przechowywane w pamięci lokalnej.

<code-tabs name="predict" />

`result.depth_map` zawiera gęstą względną mapę odwrotnej głębokości: wyższe wartości oznaczają bliżej kamery, a wartości te nie mają jednostki metrycznej ani skali międzyobrazowej. Checkpoint upstream generuje dodatnią względną głębokość; opakowanie sieci LibreYOLO odwraca ją i odwzorowuje oficjalne przetwarzanie nieba, tak aby wynik odpowiadał udostępnionemu kontraktowi głębokości LibreYOLO. `save=True` zapisuje wizualizację tej mapy w kolorze na dysku; `Results.plot()` nie obejmuje tej rodziny, ponieważ definiowana jest tylko dla normalnych powierzchni i krawędzi. Zobacz [predykcja](/docs/predict) dla źródeł, streaming i przetwarzania wyników.

## Warianty

Jeden rozmiar, `l`, przy stałej rozdzielczości wejściowej. Upstream DA3 publikuje również checkpointy Small i Base any-view, checkpoint o metrycznej głębokości oraz checkpointy Nested i Giant; LibreYOLO nie udostępnia żadnego z nich. Głębokość metryczna wymaga innego publicznego kontraktu niż zadanie względnej odwrotnej głębokości LibreYOLO, a checkpointy any-view i Nested wymagają kamery wieloobrazowej API, której LibreYOLO nie oferuje. Checkpointy Large i Giant any-view są również CC-BY-NC-4.0 i nie są odwoływane przez żadną ścieżkę pobierania LibreYOLO.

Trenowanie nie jest oferowane dla tej rodziny. `LibreDepthAnything3.train()` zgłasza `NotImplementedError` bezwarunkowo; trenuj upstream i skonwertuj kompatybilny checkpoint DA3MONO-LARGE za pomocą `weights/convert_depth_anything3_weights.py`.

## Walidacja

`val()` uruchamia wspólny walidator głębokości: dopasowuje każdą prognozę do swojej prawdziwej wartości przy użyciu skalowania i przesunięcia najmniejszych kwadratów dla każdego obrazu, a następnie raportuje standardowe metryki względnej głębokości zero-shot, AbsRel, RMSE oraz trzy progi delta.

<code-tabs name="val" />

## Eksport

<export-matrix />

Eksport jest ograniczony do pięciu formatów dla tej rodziny: ONNX, TorchScript, ExecuTorch, TensorRT i OpenVINO. Poproszenie o jakikolwiek inny format wywołuje `NotImplementedError` zamiast próby niezweryfikowanej konwersji. Wyeksportowany artefakt ładuje się z powrotem przez `LibreYOLO()` na podstawie jego rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint i zwraca ten sam `Results`, z `depth_map` zamiast pudełek.

<code-tabs name="export" />

## Checkpointy

Każdy opublikowany plik wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
