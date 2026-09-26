---
title: NAFNet
families:
  - nafnet
seo_title: 'NAFNet: odszumianie, trenowanie i eksport na licencji MIT'
description: >-
  Używaj modelu NAFNet w LibreYOLO do odszumiania i rekonstrukcji obrazów.
  Instalacja, predykcja, trenowanie, walidacja i eksport punktu kontrolnego SIDD
  na licencji MIT.
lead: >-
  NAFNet to sieć konwolucyjna do rekonstrukcji obrazów, która usuwa nieliniowe
  funkcje aktywacji z typowego bloku UNet i zastępuje je mnożeniem element po
  elemencie. LibreYOLO obsługuje ją w jednym zadaniu, rekonstrukcji, z
  opublikowanym punktem kontrolnym do odszumiania rzeczywistych obrazów
  wytrenowanym na SIDD.
keywords:
  - NAFNet
  - rekonstrukcja obrazu
  - odszumianie obrazu
  - usuwanie rozmycia obrazu
  - sieć bez nieliniowych aktywacji
  - SIDD
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model("noisy.jpg", save=True)

        restored = result.restored
        print(restored.array.shape)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreNAFNetl-restore-sidd.pt source=noisy.jpg
        save=True
    - label: Zapis zrekonstruowanego obrazu
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model.predict("noisy.jpg")

        result.restored.save("denoised.png")
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 imgsz=256 batch=16 lr0=1e-3
    - label: Pochodzenie punktu kontrolnego
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Rodzaj degradacji i zbiór danych są zapisywane w punkcie kontrolnym;
        # nie zmieniają trenowanych elementów.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
    - label: Wiele GPU
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() zwraca zwykły słownik, a nie obiekt
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.export(format="onnx", imgsz=256)
        model.export(format="tensorrt", imgsz=256, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=onnx
        imgsz=256

        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=tensorrt
        imgsz=256 half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, dlatego

        # artefakt ładuje się jak punkt kontrolny i zwraca ten sam obiekt
        Results.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")

        result = model("noisy.jpg")


        result.restored.save("denoised.png")
source_hash: 9bae9f82bee741bf
---

## Instalacja

NAFNet nie wymaga opcjonalnych dodatków. Wszystkie importowane przez nią elementy znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` zawiera dla tej rodziny jedno pole, `restored`, czyli
gęsty obraz RGB uint8 w układzie HWC na oryginalnym płótnie. Nie ma ramek do
iterowania. Ustawienie `save=True` zapisuje zrekonstruowany obraz bezpośrednio
na dysku zamiast rysować adnotację na obrazie wejściowym. Parametry `conf`,
`iou` i `max_det` są przyjmowane dla zgodności sygnatury z każdą inną rodziną,
ale nie mają wpływu na wynik, ponieważ rekonstrukcja nie tworzy detekcji do
filtrowania. Informacje o źródłach, strumieniowaniu i obsłudze wyników znajdziesz
w sekcji [predykcja](/docs/predict).

## Warianty

Ta architektura ma dwie szerokości: `s` (szerokość 32) i `l` (szerokość 64),
obie zbudowane wokół fragmentu treningowego 256 px. Predykcja i walidacja
działają w natywnej rozdzielczości obrazu niezależnie od rozmiaru, z dopełnieniem
wyłącznie do współczynnika zmniejszania rozdzielczości sieci. Obecnie opublikowano
tylko szerokość `l`, jako punkt kontrolny do odszumiania rzeczywistych obrazów
wytrenowany na SIDD.

## Trenowanie

NAFNet dostraja się na własnych parach obrazów zdegradowanych i czystych. Plik
YAML zbioru danych wskazuje folder `inputs/<split>/` ze zdegradowanymi obrazami
oraz folder `targets/<split>/` z czystymi obrazami docelowymi, dopasowanymi według
rdzenia nazwy pliku. `degradation` i `dataset` to opcjonalne ciągi zapisywane
w punkcie kontrolnym jako informacja o pochodzeniu. Nie biorą udziału w trenowaniu.

<code-tabs name="train" />

Przy ustawieniach domyślnych trener wykonuje 100 epok z optymalizatorem AdamW,
`lr0=1e-3`, partiami po 16, wycinkami 256 px oraz wczesnym zatrzymaniem po 50
epokach bez poprawy PSNR. Ta rodzina nie ma ścieżki LoRA: `lora=True` zgłasza
błąd zamiast rozpoczynać działanie, ponieważ `NAFNetTrainer` nie obsługuje
dostrajania adapterów.

Podczas trenowania sieć używa zwykłego globalnego uśredniania. Przeznaczone
wyłącznie do wnioskowania lokalne uśrednianie okienkowe NAFNet (Test-time Local
Converter) jest odłączane przed pierwszą epoką i ponownie dołączane po zakończeniu
trenowania, ponieważ propagacja wsteczna przez lokalne uśrednianie o stałym oknie
nie odpowiadałaby sposobowi użycia punktu kontrolnego podczas wnioskowania.

Informacje o zbiorach danych, augmentacji, wielu GPU i loggerach znajdziesz w sekcji [trenowanie](/docs/train).

## Walidacja

Metoda `val()` zwraca słownik z `metrics/PSNR` i `metrics/SSIM`, obliczanymi w RGB
na całym prawidłowym płótnie. SSIM używa okna Gaussa 11x11 z sigma 1,5, a `fitness`
do wyboru najlepszego punktu kontrolnego jest wartością PSNR. Parametr `data`
wskazuje ten sam format zbioru sparowanych obrazów, którego używa trenowanie.

<code-tabs name="val" />

## Eksport

<export-matrix />

Wyeksportowany artefakt jest ponownie ładowany przez `LibreYOLO()` na podstawie
rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak punkt
kontrolny i zwraca ten sam obiekt `Results`, a `restored` zawiera obraz wynikowy.
NAFNet jest eksportowany ze stałą rozdzielczością przestrzenną: `imgsz` musi być
podzielne przez współczynnik zmniejszania rozdzielczości sieci (16 dla obu
szerokości architektury), a przy `dynamic=True` dynamiczny jest tylko wymiar partii.
Wysokość i szerokość są ustalane podczas eksportu.

<code-tabs name="export" />

## Punkty kontrolne

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box></provenance-box>

## Cytowanie

<citation-block />

