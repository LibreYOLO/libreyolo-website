---
title: Real-ESRGAN
families:
  - realesrgan
seo_title: 'Real-ESRGAN: superrozdzielczość obrazów w LibreYOLO'
description: >-
  Używaj Real-ESRGAN w LibreYOLO do praktycznej superrozdzielczości obrazów 4x,
  2x i szybkiego poziomu 4x. Instaluj, przewiduj, waliduj i eksportuj.
lead: >-
  Praktyczny upscaler ślepej superrozdzielczości wytrenowany na syntetycznych
  degradacjach zamiast wyłącznie zmniejszaniu dwusześciennym. LibreYOLO
  udostępnia inferencję i walidację dla checkpointów 4x, 2x i szybkiego 4x.
keywords:
  - Real-ESRGAN
  - RRDBNet
  - SRVGGNetCompact
  - superrozdzielczość obrazów
  - odtwarzanie obrazów
  - ślepa superrozdzielczość
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRealESRGANx4-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Kafelki dla dużych obrazów
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRealESRGANx4-restore.pt")


        # tile dzieli przebieg w przód na nakładające się kafelki i scala ich
        szwy.

        # tile_pad to obwódka dodawana wokół każdego kafelka przed ponownym
        przycięciem.

        # Oba są argumentami nazwanymi tylko dla Pythona, a nie flagami CLI.

        result = model("large-photo.jpg", tile=512, tile_pad=10, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: >
        libreyolo val model=LibreRealESRGANx4-restore.pt
        data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRealESRGANx4-restore.pt")


        # Gdy imgsz pominięto, domyślnie przyjmuje mały wewnętrzny rozmiar
        patcha,

        # a nie rozdzielczość roboczą. Podaj rozmiar faktycznie przekazywany
        modelowi

        # przez wdrożenie.

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRealESRGANx4-restore.pt format=onnx
        imgsz=512
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreRealESRGANx4-restore.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.restored.array.shape)
source_hash: f0efb4f65d38e22d
---

## Instalacja

Real-ESRGAN nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane
elementy znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Wynik odtwarzania nie zawiera ramek. `result.restored` jest gęstym obrazem RGB
uint8 `(H, W, 3)` na obszarze `Results.restore_scale` razy większym od wejścia
w każdym wymiarze. Ustawienie `save=True` zapisuje bezpośrednio ten obraz zamiast
wykresu z adnotacjami. Wejście jest konwertowane do RGB, a kanał alfa jest
usuwany. Źródło większe niż dostępna pamięć można podzielić argumentami `tile` i
`tile_pad`, które scalają szwy kafelków w wyniku. Więcej informacji o źródłach,
streamingu i obsłudze wyników zawiera strona [predykcji](/docs/predict).

## Warianty

Dostępne są trzy checkpointy nazwane według współczynnika skalowania. `x4` to
RRDBNet (`RealESRGAN_x4plus`) z 23 gęstymi blokami residual-in-residual,
domyślny wariant jakościowy 4x. `x2` to ta sama architektura RRDBNet przy 2x.
`x4t` to SRVGGNetCompact (`realesr-general-x4v3`), mniejszy i szybszy generator
zbudowany do wideo i zastosowań o mniejszym opóźnieniu przy 4x. Uniwersalny
model źródłowy udostępnia również sparowaną sieć siły odszumiania mieszaną
podczas inferencji. To ustawienie siły nie jest częścią tego portu, który
uruchamia podstawowy generator `x4t`.

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
i zwraca ten sam obiekt `Results`. Strona [eksportu](/docs/export) wymienia
argumenty obsługiwane przez każdy format oraz dodatki wymagane przez niektóre z
nich.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
