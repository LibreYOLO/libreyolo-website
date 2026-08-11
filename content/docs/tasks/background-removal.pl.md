---
title: Usuwanie tła
seo_title: Usuwanie tła w LibreYOLO
description: >-
  Wytnij obiekt z tła w LibreYOLO. Przewiduj miękką maskę alfa, zapisuj
  przezroczyste pliki PNG i waliduj za pomocą MAE oraz S-measure.
lead: >-
  Usuwanie tła oddziela obiekt od wszystkiego, co znajduje się za nim. LibreYOLO
  udostępnia je jako zadanie matte, które zwraca miękką wartość alfa dla każdego
  piksela zamiast twardej maski pierwszego planu.
keywords:
  - usuwanie tła python
  - model alpha matting
  - segmentacja obrazu obiekt tło
  - wycinanie do przezroczystego png
  - miękka maska alfa
last_verified: 1.5.0
snippets:
  predict:
    - label: Predykcja mapy alfa
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        result = model(SAMPLE_IMAGE)


        matte = result.matte

        print(matte.array.shape, matte.array.dtype)   # (H, W) float32 w
        zakresie [0, 1]
    - label: Zapis przezroczystego pliku PNG
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # save() łączy obraz źródłowy z mapą alfa jako kanałem alfa.
        result.save("subject.png")

        rgba = result.cutout()   # ta sama tablica uint8 (H, W, 4) w pamięci
        print(rgba.shape)
    - label: Kompozycja na nowym tle
      language: python
      code: >
        import numpy as np

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        result = model(SAMPLE_IMAGE)


        rgba = result.cutout()

        alpha = rgba[..., 3:4].astype(np.float32) / 255.0

        backdrop = np.full_like(rgba[..., :3], 255)          # biel

        composited = (rgba[..., :3] * alpha + backdrop * (1 -
        alpha)).astype(np.uint8)

        print(composited.shape)
  val:
    - label: Walidacja i odczyt kluczy metryk
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Zamiast pliku YAML zbioru danych można użyć katalogu zawierającego
        # podkatalogi images/ i matte.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])        # im mniej, tym lepiej
        print(metrics["metrics/Smeasure"])   # fitness, im więcej, tym lepiej
  export:
    - label: Eksport
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="torchscript")
    - label: Uruchomienie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Funkcja fabrykująca wybiera ścieżkę na podstawie sufiksu pliku, więc

        # wyeksportowany artefakt wczytuje się jak checkpoint i zwraca ten sam
        obiekt Results.

        model = LibreYOLO("LibreBiRefNetl-matte.torchscript")

        result = model(SAMPLE_IMAGE)


        print(result.matte.array.shape)
source_hash: f7d88c74d9729268
---

## Definicja

Zadanie `matte` przewiduje jedną wartość alfa na piksel pojedynczego obrazu RGB:
`1` oznacza w całości pierwszy plan, a `0` w całości tło. Wartość jest ciągła,
a nie binarna, co stanowi istotę zadania. Twardą maskę można uzyskać przez
zastosowanie progu 0.5, natomiast miękka mapa alfa zachowuje dodatkowo częściowe
pokrycie włosów, sierści i krawędzi rozmytych ruchem, które maska binarna odrzuca.

Predykcja wypełnia `result.matte`, czyli strukturę `Matte` zawierającą tablicę
float32 `(H, W)` w zakresie `[0, 1]` na płótnie oryginalnego obrazu. Przez
`.array` jest ona dostępna jako tablica NumPy. Funkcja `result.cutout()` łączy
obraz źródłowy z tą wartością alfa w tablicę RGBA uint8 `(H, W, 4)`, a
`result.save(path)` zapisuje ten sam wynik jako plik PNG z przezroczystym tłem.
`result.boxes` pozostaje pusty, więc `conf`, `iou` i `max_det` nie mają wpływu.

## Modele

Zadanie `matte` obsługują dwie rodziny, które korzystają ze wspólnej ścieżki
forward.

[BiRefNet](/docs/models/birefnet) to sieć bilateral-reference, wokół której
zbudowano zadanie. Jest tutaj publikowana jako jeden checkpoint poziomu Swin-L.

[FeyNobg](/docs/models/feynobg) to pogłębiony wariant firmy Feyn Inc.:
architektura BiRefNet, w której trzeci etap Swin zwiększono z 18 do 24 bloków,
a następnie ponownie przeprowadzono trenowanie. LibreYOLO ponownie wykorzystuje
dla niego ścieżkę forward, przetwarzanie wstępne i wyjście z jednym logitem z
BiRefNet. Dzięki temu predykcja, walidacja i obsługa checkpointów działają
identycznie, natomiast wagi i tożsamość rodziny należą do FeyNobg.

Obie rodziny mają inne licencje wag. Podano je na stronach modeli, a
rozstrzygająca jest licencja w repozytorium Hugging Face konkretnego checkpointu.

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Obie rodziny działają na stałym natywnym płótnie 1024x1024 i zmieniają rozmiar
mapy alfa z powrotem do oryginalnego obrazu. Inna rozdzielczość nie jest
obsługiwana, ponieważ tablice pozycji względnych w backbone Swin są związane z
tym rozmiarem, a niezgodność powoduje ich błędną interpolację zamiast zgłoszenia
wyjątku. `Results.save()` jest zdefiniowane wyłącznie dla wyników matte i wymaga
obrazu źródłowego, który wczytuje ponownie z `Results.path`, chyba że zostanie
przekazany bezpośrednio. Informacje o źródłach, streamingu i obsłudze wyników
zawiera strona [predykcji](/docs/predict).

## Format zbioru danych

Walidacja matte łączy każdy obraz RGB z jednokanałową referencyjną mapą alfa o
tej samej nazwie bazowej. Wartość 0 oznacza tło, a 255 pierwszy plan.

```text
my-matte-dataset/
  images/
    subject.jpg
  mattes/
    subject.png
```

Wystarczy przekazać ten katalog główny jako `data=`. Katalog matte jest
automatycznie wykrywany wśród `mattes/`, `matte/`, `gt/`, `masks/`, `mask/` i
`alpha/`. Alternatywą jest plik YAML zbioru danych zawierający `path` oraz
`val_images` i `val_mattes` wskazujące katalogi względem tej ścieżki:

```yaml
path: my-matte-dataset
val_images: images
val_mattes: mattes
nc: 1
names: {0: matte}
```

`nc` i `names` są polami wymaganymi przez schemat. Model matte zwraca
`Results.matte`, a nie detekcje. Wartości mapy alfa są odczytywane w zakresie
`[0, 1]` przez podzielenie przez 255. Jeśli kształt mapy różni się od płótna
predykcji, jest ona przeskalowywana biliniowo. Pełny kontrakt opisują [formaty
zbiorów danych](/docs/reference/dataset-formats).

## Trenowanie

Żadna z rodzin matte nie ma implementacji trenowania. Funkcja `train()` zgłasza
`NotImplementedError` dla obu, a obsługa matte obejmuje tylko predykcję,
walidację i eksport. Strona każdego modelu wskazuje projekt źródłowy zawierający
kod trenowania oraz skrypt konwersji do ponownego wczytania checkpointu.

## Walidacja

Funkcja `val()` steruje własną funkcją `predict` modelu, dlatego walidacja używa
dokładnego przetwarzania wstępnego danej rodziny, a obie metryki są obliczane na
płótnie oryginalnego obrazu.

<code-tabs name="val" />

`metrics/MAE` jest średnim błędem bezwzględnym względem referencyjnej wartości
alfa w zakresie `[0, 1]`. Im mniej, tym lepiej. `metrics/Smeasure` jest miarą
S-measure autorstwa Fana i współautorów (ICCV 2017), czyli podobieństwem
strukturalnym oceniającym poprawność kształtu obiektu i otworów w jego wnętrzu,
których nie uwzględnia sama średnia dla pikseli. Im więcej, tym lepiej.
S-measure jest również wartością `fitness`, używaną przy wyborze najlepszego
checkpointu. Żadna z metryk nie zależy od rozdzielczości.

## Eksport

Wyeksportowany model matte wczytuje się ponownie przez `LibreYOLO()` na
podstawie sufiksu pliku, więc artefakt działa jak checkpoint i zwraca ten sam
obiekt `Results`.

<code-tabs name="export" />

TorchScript jest zweryfikowaną ścieżką dla tego zadania. Konwersja ONNX działa,
ale nie osiągnęła tego samego poziomu zgodności, a pozostałe formaty są
niedostępne. Zakres poszczególnych formatów podano na stronach
[BiRefNet](/docs/models/birefnet) i [FeyNobg](/docs/models/feynobg) oraz w
[pełnej macierzy eksportu](/docs/reference/export-matrix).
