---
title: BiRefNet
families:
  - birefnet
seo_title: 'BiRefNet: usuwanie tła i matting w LibreYOLO'
description: >-
  Używaj BiRefNet w LibreYOLO do usuwania tła i dychotomicznej segmentacji
  obrazów. Zainstaluj model, uruchamiaj predykcję i walidację oraz eksportuj
  ogólny checkpoint.
lead: >-
  Sieć z dwustronnymi odniesieniami, która przewiduje miękką matę alfa
  oddzielającą obiekt od tła. LibreYOLO udostępnia inferencję i walidację dla
  zadania matting modelu BiRefNet.
keywords:
  - BiRefNet
  - usuwanie tła
  - segmentacja obrazu
  - mata alfa
  - image matting
  - wycinanie tła
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreBiRefNetl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Wycięcie
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: źródłowe RGB oraz mata jako kanał alfa.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Zamiast pliku YAML zbioru danych można też podać katalog zawierający
        # images/ oraz automatycznie wykrywany katalog z matami
        # (mattes/, matte/, gt/, masks/, mask/ lub alpha/).
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreBiRefNetl-matte.pt format=onnx
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie sufiksu pliku, więc
        wyeksportowany

        # artefakt wczytuje się jak każdy checkpoint i zwraca ten sam obiekt
        Results.

        model = LibreYOLO("LibreBiRefNetl-matte.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.matte.array.shape)
source_hash: 1af1bd7f4f905081
---

## Instalacja

BiRefNet nie wymaga opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji bazowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane lokalnie
w pamięci podręcznej.

<code-tabs name="predict" />

Wynik matting nie zawiera ramek. `result.matte` to gęsta tablica float32
o kształcie `(H, W)` i wartościach w zakresie `[0, 1]`, gdzie 1 oznacza
w pełni pierwszy plan, a 0 w pełni tło. W przeciwieństwie do maski binarnej
miękka mata zachowuje wygładzone szczegóły krawędzi, na przykład włosy i sierść.
Metoda `result.cutout()` łączy obraz źródłowy z tym kanałem alfa w tablicę RGBA,
a `result.save(path)` (lub `save=True` w wywołaniu predykcji) zapisuje ją
bezpośrednio jako plik PNG z przezroczystym tłem. Model działa na stałym,
natywnym obszarze 1024x1024. Inna rozdzielczość nie jest obsługiwana, ponieważ
tabele pozycji względnych w backbone Swin są z nią związane, a przy niezgodności
zostają nieprawidłowo interpolowane zamiast zgłoszenia błędu. Zobacz stronę
[predykcji](/docs/predict), aby poznać źródła, streaming i obsługę wyników.

## Warianty

Opublikowano jeden checkpoint `l`: model BiRefNet-general z warstwą Swin-L,
który jest domyślnym wariantem upstream zapewniającym najwyższą jakość. Kod
rodziny obsługuje też lekki wariant Swin-T oznaczony jako `t`, ale jego konwersja
do LibreYOLO nie została jeszcze opublikowana.

## Walidacja

Metoda `val()` raportuje dwie metryki dla sparowanego folderu obrazów i mat.
Obie mieszczą się w zakresie `[0, 1]` i są niezależne od rozdzielczości: MAE,
czyli średni błąd bezwzględny względem referencyjnego kanału alfa (niższa wartość
jest lepsza), oraz S-measure (Fan i in., ICCV 2017), czyli podobieństwo
strukturalne uwzględniające zachowanie kształtu obiektu i otworów, których nie
obejmuje sam pikselowy błąd MAE (wyższa wartość jest lepsza). Walidacja korzysta
z własnej metody `predict` modelu, dlatego stosuje dokładnie jego przetwarzanie
wstępne.

<code-tabs name="val" />

Walidacja obejmuje tylko inferencję. Dostrajanie jest udokumentowanym planem
rozwoju, a nie dostępną funkcją. Dokładne ograniczenie rozdzielczości, które
odziedziczyłby przyszły moduł trenowania, opisano w sekcji Predykcja.

## Eksport

<export-matrix />

Wyeksportowany artefakt jest ponownie wczytywany przez `LibreYOLO()` na
podstawie sufiksu pliku, dlatego plik `.onnx` zachowuje się jak checkpoint
i zwraca ten sam obiekt `Results`. Zweryfikowaną ścieżką jest TorchScript.
Konwersja ONNX działa, lecz nie spełniła jeszcze takiego samego kryterium
zgodności. Strona [Eksport](/docs/export) zawiera argumenty obsługiwane przez
każdy format oraz dodatki wymagane przez niektóre z nich.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
