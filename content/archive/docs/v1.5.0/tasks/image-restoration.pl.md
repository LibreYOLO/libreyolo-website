---
title: Odtwarzanie obrazów
seo_title: Odtwarzanie i powiększanie obrazów w LibreYOLO
description: >-
  Odszumiaj, usuwaj rozmycie i powiększaj obrazy w LibreYOLO. Przewiduj
  odtworzony obraz RGB, trenuj NAFNet na sparowanych danych i odczytuj klucze
  PSNR oraz SSIM.
lead: >-
  Odtwarzanie obrazu przyjmuje obraz zdegradowany i zwraca czysty. LibreYOLO
  udostępnia je jako zadanie restore, które obejmuje odszumianie, usuwanie
  rozmycia i superrozdzielczość za jednym kontraktem wyjściowym: jeden obraz RGB
  na wejściu i jeden obraz RGB na wyjściu.
keywords:
  - odtwarzanie obrazów python
  - model odszumiania obrazu
  - super resolution python
  - model usuwania rozmycia
  - walidacja PSNR SSIM
last_verified: 1.5.0
snippets:
  predict:
    - label: Powiększenie obrazu
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Kompaktowy generator 4x; tile ogranicza szczytowe zużycie pamięci dla
        dużego źródła.

        model = LibreYOLO("LibreRealESRGANx4t-restore.pt")

        result = model(SAMPLE_IMAGE, tile=512, tile_pad=10)


        result.restored.save("upscaled.png")

        print(result.restored.array.shape)   # 4x względem wejścia w każdej osi
    - label: Odszumianie obrazu
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Wytrenowany na szumie rzeczywistych obrazów SIDD; wyjście zachowuje
        rozmiar wejścia.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        result = model(SAMPLE_IMAGE)


        result.restored.save("denoised.png")

        print(result.restore_scale)   # 1: ten checkpoint nie powiększa obrazu
  train:
    - label: Dostrajanie NAFNet na sparowanych obrazach
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: Zapis pochodzenia w checkpointcie
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # degradation i dataset są zapisywane w checkpointcie jako informacje
        # o pochodzeniu; nie wpływają na trenowanie.
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
  val:
    - label: Walidacja i odczyt kluczy metryk
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() zwraca zwykły słownik, a nie obiekt.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])   # fitness
        print(metrics["metrics/SSIM"])
  export:
    - label: Eksport
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # imgsz jest utrwalany w grafie, dlatego przekaż rozmiar rzeczywiście
        # podawany modelowi we wdrożeniu.
        model.export(format="onnx", imgsz=256)
    - label: Uruchomienie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Funkcja fabrykująca wybiera ścieżkę na podstawie sufiksu pliku, więc

        # wyeksportowany artefakt wczytuje się jak checkpoint i zwraca ten sam
        obiekt Results.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")

        result = model(SAMPLE_IMAGE)


        result.restored.save("denoised.png")
source_hash: 9dc81cadb3ebf18b
---

## Definicja

Zadanie `restore` mapuje jeden obraz na drugi. Odszumianie, usuwanie rozmycia i
superrozdzielczość są tutaj tym samym zadaniem, ponieważ współdzielą jeden
kontrakt. Model przyjmuje obraz RGB i zwraca obraz RGB, a degradacja, której
usuwania go wytrenowano, jest właściwością checkpointu, a nie API.

Predykcja wypełnia `result.restored`, czyli strukturę `RestoredImage` zawierającą
tablicę RGB uint8 `(H, W, 3)`. `.array` zwraca ją jako NumPy, a `.save(path)`
zapisuje na dysku. `result.restore_scale` przechowuje współczynnik powiększenia
płótna wyjściowego, który wynosi `1` dla checkpointu zachowującego
rozdzielczość. `result.boxes` pozostaje pusty, więc `conf`, `iou` i `max_det` są
przyjmowane dla zgodności sygnatury, ale nie mają wpływu. `save=True` zapisuje
bezpośrednio odtworzony obraz zamiast zdjęcia z adnotacjami.

## Modele

Zadanie `restore` obsługują trzy rodziny podzielone według usuwanej degradacji.

[NAFNet](/docs/models/nafnet) jest modelem odszumiającym i jedyną rodziną
odtwarzania, którą LibreYOLO może trenować. Jego architektura zastępuje
nieliniowe aktywacje bloku UNet mnożeniem element po elemencie, a opublikowany
checkpoint wytrenowano na szumie rzeczywistych obrazów SIDD. Wyjście zachowuje
rozdzielczość wejściową.

[Real-ESRGAN](/docs/models/real-esrgan) jest praktycznym modelem powiększającym.
Ma trzy checkpointy wytrenowane względem syntetycznych degradacji, a nie tylko
próbkowania w dół metodą bikubicznej interpolacji. Obejmują one wariant 4x,
wariant 2x i mniejszy, szybszy generator 4x zbudowany pod kątem niższego
opóźnienia.

[SwinIR](/docs/models/swinir) powiększa obraz 4x przy użyciu backbone Swin
Transformer. Występuje w trzech rozmiarach obejmujących oficjalny lekki
generator i dwa generatory do rzeczywistych obrazów.

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Odtwarzanie działa w rozdzielczości obrazu źródłowego, a nie na stałym płótnie
sieci. Dopełnienie jest stosowane tylko do współczynnika próbkowania w dół
sieci, dlatego czas i pamięć rosną wraz z liczbą pikseli wejściowych. `tile`
dzieli przebieg forward na nakładające się kafelki i ponownie łączy ich szwy, a
`tile_pad` jest obramowaniem dodawanym wokół każdego kafelka przed ponownym
przycięciem. Oba są argumentami nazwanymi Pythona. Informacje o źródłach,
streamingu i obsłudze wyników zawiera strona [predykcji](/docs/predict).

## Format zbioru danych

Odtwarzanie łączy każdy zdegradowany obraz wejściowy z czystym obrazem docelowym
o dokładnie tej samej rozdzielczości, dopasowanym według nazwy bazowej pliku.

```text
dataset/
  data.yaml
  inputs/
    train/photo.jpg
    val/photo.jpg
  targets/
    train/photo.jpg
    val/photo.jpg
```

```yaml
path: dataset
train: inputs/train
val: inputs/val
input_dir: inputs
target_dir: targets
degradation: denoise
dataset: MyDataset
nc: 1
names: {0: image}
```

`nc` i `names` są polami wymaganymi przez schemat. Model restore zwraca
`Results.restored`, a nie detekcje. `degradation` i `dataset` są opcjonalnymi
etykietami pochodzenia. `target_stem_suffix` obsługuje zbiory danych, które
nazywają czysty obraz inaczej niż jego zdegradowaną parę. Walidacja zachowuje
natywną rozdzielczość i dopełnia tylko tyle, ile potrzeba do ułożenia batcha,
dlatego metryki są obliczane na oryginalnym płótnie. Pełny kontrakt opisują
[formaty zbiorów danych](/docs/reference/dataset-formats).

## Trenowanie

NAFNet jest jedyną rodziną restore z implementacją trenowania. Zarówno
`Real-ESRGAN.train()`, jak i `SwinIR.train()` zgłaszają `NotImplementedError`.
Checkpointy te pochodzą z trenowania GAN na pipeline'ach syntetycznych
degradacji, a moduł trenujący restore na sparowanych danych działałby bez
odtworzenia tej procedury.

<code-tabs name="train" />

Moduł trenujący pobiera powiązane wycinki pary wejścia i celu, dzięki czemu obie
strony pozostają wyrównane. Informacje o zbiorach danych, wielu GPU i modułach
rejestrujących zawiera strona [trenowania](/docs/train), a wartości domyślne tej
rodziny oraz pooling odłączany podczas trenowania po stronie inferencji opisuje
[strona NAFNet](/docs/models/nafnet).

## Walidacja

Funkcja `val()` porównuje odtworzone wyjście z czystym celem w RGB, na
oryginalnym płótnie, bez przycinania krawędzi i zmiany rozmiaru.

<code-tabs name="val" />

`metrics/PSNR` jest szczytowym stosunkiem sygnału do szumu w decybelach oraz
wartością `fitness`, używaną przy wyborze najlepszego checkpointu.
`metrics/SSIM` jest podobieństwem strukturalnym w zakresie `[0, 1]`, obliczanym
za pomocą okna Gaussa 11x11 przy sigma 1.5 i uśrednianym w trzech kanałach
kolorów. Dla obu metryk większa wartość jest lepsza.

## Eksport

Wyeksportowany model restore wczytuje się ponownie przez `LibreYOLO()` na
podstawie sufiksu pliku. Plik `.onnx` lub `.engine` działa więc jak checkpoint i
zwraca ten sam obiekt `Results`, w którym `restored` przechowuje obraz wyjściowy.

<code-tabs name="export" />

Eksport restore utrwala rozdzielczość przestrzenną w grafie, dlatego należy
przekazać `imgsz` rzeczywiście podawany modelowi przez wdrożenie. W przypadku
NAFNet rozmiar musi dzielić się przez współczynnik próbkowania w dół sieci, a
przy `dynamic=True` dynamiczny pozostaje tylko wymiar batcha. W przypadku
Real-ESRGAN i SwinIR pominięcie `imgsz` powoduje użycie małego wewnętrznego
rozmiaru fragmentu zamiast rozdzielczości roboczej. Zakres poszczególnych
formatów podano na stronach modeli i w [pełnej macierzy
eksportu](/docs/reference/export-matrix). Strona [Eksport](/docs/export)
wymienia argumenty przyjmowane przez każdy format.
