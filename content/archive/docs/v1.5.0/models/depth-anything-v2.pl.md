---
title: Depth Anything V2
families:
  - depth_anything
seo_title: 'Depth Anything V2: przewidywać i weryfikować głębię z jednego oka'
description: >-
  Użyj Depth Anything V2 w LibreYOLO do jednookularnej estymacji głębi.
  Zainstaluj, przewiduj i zweryfikuj; Małe statki Apache-2.0, Podstawowe i Duże
  to CC-BY-NC-4.0.
lead: >-
  Depth Anything V2 to enkoder DINOv2 sparowany z dekoderem DPT, który
  przewiduje gęstą względną mapę odwrotnej głębi z pojedynczego obrazu.
  LibreYOLO obsługuje go dla zadania głębi: predykcja i walidacja zero-shot, bez
  ścieżki uczenia.
keywords:
  - Depth Anything V2
  - jednooczna estymacja głębokości
  - DPT
  - DINOv2
  - względna głębokość
  - mapa głębi
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnythingV2s-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Odczytaj mapę głębi
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")

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

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnythingV2s-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=onnx

        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=tensorrt
        half=True
    - label: Użyj wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka kieruje na podstawie rozszerzenia pliku, więc eksportowany
        artefakt się ładuje

        # jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: e1043aba1b70b65c
---
## Instalacja

Depth Anything V2 nie potrzebuje żadnych dodatkowych opcji. Wszystko, co importuje, znajduje się w instalacji bazowej.

```bash
pip install libreyolo
```

## Predykcja

Wagi są pobierane z Hugging Face przy pierwszym użyciu i są przechowywane w pamięci lokalnej.

<code-tabs name="predict" />

`result.depth_map` przenosi gęstą względną mapę odwrotnej głębokości: wyższe wartości oznaczają bliżej kamery, a wartości te nie mają jednostki metrycznej ani skali międzyobrazowej. `save=True` zapisuje wizualizację tej mapy z kolorami na dysk; `Results.plot()` nie obejmuje tej rodziny, ponieważ jest zdefiniowana tylko dla normalnych powierzchni i krawędzi. Rozdzielczość wejściowa musi dzielić się dokładnie przez 14, czyli przez siatkę patchy DINOv2, na której buduje się głowica DPT; LibreYOLO sprawdza to przed uruchomieniem i zgłasza błąd, jeśli tak nie jest. Zobacz [predykcja](/docs/predict) dla źródeł, streaming i obsługi wyników.

## Warianty

Cztery rozmiary enkodera, s/b/l/g, odpowiadające ViT-S/B/L/G. Poniższa tabela checkpointów wymienia tylko s, b i l; checkpoint Giant nie został opublikowany. Wszystkie cztery mają taką samą rozdzielczość wejściową, więc wybór rozmiaru dotyczy pojemności enkodera, a nie rozmiaru obrazu. Licencjonowanie również ma znaczenie: checkpoint Small to Apache-2.0, natomiast Base i Large to CC-BY-NC-4.0, zobacz poniżej w sekcji Licencjonowanie.

Trenowanie i dostrajanie nie są oferowane dla tej rodziny. `LibreDepthAnythingV2.train()` zgłasza `NotImplementedError` bezwarunkowo; zamiast tego skonwertuj zgodny checkpoint upstream, z `weights/convert_depth_anything_v2_weights.py`.

## Walidacja

`val()` uruchamia wspólny walidator głębokości: dopasowuje każdą prognozę do jej prawdziwej wartości przy użyciu skalowania i przesunięcia najmniejszych kwadratów dla każdego obrazu, a następnie raportuje standardowe metryki względnej głębokości zero-shot, AbsRel, RMSE oraz trzy progi delta.

<code-tabs name="val" />

## Eksport

<export-matrix />

Eksportowany artefakt ładuje się ponownie przez `LibreYOLO()` na jego rozszerzeniu pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint i zwraca ten sam `Results`, z `depth_map` zamiast skrzynek. [Eksport](/docs/export) wymienia argumenty, które akceptuje każdy format.

<code-tabs name="export" />

## Checkpointy

Każdy opublikowany plik wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
