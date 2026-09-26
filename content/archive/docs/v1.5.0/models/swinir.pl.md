---
title: SwinIR
families:
  - swinir
seo_title: 'SwinIR: superrozdzielczość obrazów 4x w LibreYOLO'
description: >-
  Używaj SwinIR w LibreYOLO do superrozdzielczości obrazów 4x. Instaluj,
  przewiduj, waliduj i eksportuj checkpointy lightweight, medium i large.
lead: >-
  Sieć Swin Transformer do odtwarzania obrazów. LibreYOLO udostępnia inferencję
  i walidację dla jej checkpointów superrozdzielczości 4x: oficjalnego lekkiego
  generatora oraz generatorów medium i large do rzeczywistych obrazów.
keywords:
  - SwinIR
  - Swin Transformer
  - superrozdzielczość obrazów
  - odtwarzanie obrazów
  - residual Swin Transformer block
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwinIRm-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Kafelki dla dużych obrazów
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSwinIRl-restore.pt")


        # tile dzieli przebieg w przód na nakładające się kafelki i scala ich
        szwy.

        # tile_pad to obwódka dodawana wokół każdego kafelka przed ponownym
        przycięciem.

        # Oba są argumentami nazwanymi tylko dla Pythona, a nie flagami CLI.

        result = model("large-photo.jpg", tile=512, tile_pad=16, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwinIRm-restore.pt data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSwinIRm-restore.pt")


        # Gdy imgsz pominięto, domyślnie przyjmuje mały wewnętrzny rozmiar
        patcha,

        # a nie rozdzielczość roboczą. Podaj rozmiar faktycznie przekazywany
        modelowi

        # przez wdrożenie.

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwinIRm-restore.pt format=onnx imgsz=512
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreSwinIRm-restore.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.restored.array.shape)
source_hash: 87fc3d5524480eec
---

## Instalacja

SwinIR nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Wynik odtwarzania nie zawiera ramek. `result.restored` jest gęstym obrazem RGB
uint8 `(H, W, 3)` na obszarze 4 razy większym od wejścia w każdym wymiarze.
Ustawienie `save=True` zapisuje bezpośrednio ten obraz zamiast wykresu z
adnotacjami. Wejście jest dopełniane do wielokrotności 8 zamiast skalowane,
dlatego predykcja działa w oryginalnej rozdzielczości zdjęcia. Źródło większe
niż dostępna pamięć można podzielić argumentami `tile` i `tile_pad`, które
scalają szwy kafelków w wyniku. Więcej informacji o źródłach, streamingu i
obsłudze wyników zawiera strona [predykcji](/docs/predict).

## Warianty

Dostępne są trzy rozmiary, wszystkie ze stałym skalowaniem 4x. `s` jest
oficjalnym lekkim generatorem z czterema etapami residual Swin Transformer block
(RSTB) i skalowaniem pixel-shuffle-direct. `m` i `l` są generatorami medium i
large do rzeczywistych obrazów, z sześcioma i dziewięcioma etapami RSTB oraz
skalowaniem najbliższy sąsiad plus splot, zbudowanym dla rzeczywistych degradacji
zamiast wyłącznie zmniejszania dwusześciennego.

## Walidacja

`val()` mierzy PSNR i SSIM między odtworzonym wynikiem a czystym obrazem
docelowym. Obie metryki są obliczane w RGB na oryginalnym obszarze, bez
przycinania granic i bez zmiany rozmiaru. SSIM używa okna Gaussa 11x11 z sigma
1.5, uśrednionego w trzech kanałach koloru.

<code-tabs name="val" />

Argument zbioru danych to plik YAML łączący katalog zdegradowanych obrazów
wejściowych z katalogiem czystych obrazów docelowych o zgodnej rozdzielczości.
Dokładne klucze opisują [formaty zbiorów danych](/docs/reference/dataset-formats).

## Eksport

<export-matrix />

Wyeksportowany artefakt jest ponownie ładowany przez `LibreYOLO()` na podstawie
rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint
i zwraca ten sam obiekt `Results`. ExecuTorch i każdy format oznaczony w macierzy
jako zablokowany nie są dostępne dla tej rodziny. Obsługiwane są ONNX,
TorchScript, TensorRT, OpenVINO i TFLite. Strona [eksportu](/docs/export)
wymienia argumenty obsługiwane przez każdy format oraz dodatki wymagane przez
niektóre z nich.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
