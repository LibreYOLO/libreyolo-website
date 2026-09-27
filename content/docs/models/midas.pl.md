---
title: MiDaS
families:
  - midas
seo_title: 'MiDaS: estymacja głębi monokularnej w LibreYOLO'
description: >-
  Inferencja względnej głębi MiDaS w LibreYOLO. Checkpointy s i l używają kopii
  LibreYOLO na licencji MIT wydawcy.
lead: >-
  MiDaS estymuje monokularną głębię względną i jest trenowany z funkcją straty
  niezmienną względem skali i przesunięcia na mieszanych zbiorach danych. To
  linia prac, która ustanowiła protokół transferu głębi zero-shot używany przez
  późniejsze rodziny. LibreYOLO obsługuje go w zadaniu estymacji głębi: do
  predykcji i walidacji zero-shot, bez ścieżki trenowania.
keywords:
  - MiDaS
  - estymacja głębi monokularnej
  - DPT
  - głębia względna
  - mapa głębi
  - głębia zero-shot
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Przy pierwszym użyciu pobiera kopię checkpointu.
        model = LibreYOLO("LibreMiDaSl-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMiDaSl-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Mały wariant
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Enkoder EfficientNet-Lite3, mniejszy i szybszy niż rozmiar DPT-Large
        l.

        model = LibreYOLO("LibreMiDaSs-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMiDaSl-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMiDaSl-depth.pt format=onnx
        libreyolo export model=LibreMiDaSl-depth.pt format=tensorrt half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreMiDaSl-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: 64537aa3ad3a6f8f
---

## Instalacja

MiDaS wymaga dodatku `midas` dla enkoderów timm.

```bash
pip install "libreyolo[midas]"
```

## Predykcja

Checkpointy s i l są pobierane z kopii w repozytoriach LibreYOLO na licencji MIT wydawcy i przechowywane w lokalnej pamięci podręcznej.

<code-tabs name="predict" />

`result.depth_map` zawiera gęstą mapę względnej odwrotności głębi: większe wartości oznaczają mniejszą odległość od kamery, a wartości nie mają jednostki metrycznej ani wspólnej skali między obrazami. `save=True` zapisuje na dysku wizualizację mapy z nałożonymi kolorami; `Results.plot()` renderuje mapę głębi. Źródła, streaming i obsługę wyników opisano w sekcji [predykcji](/docs/predict).

## Warianty

Dostępne są dwa warianty z różnymi enkoderami, a nie tylko różne skale tej samej
architektury. `s` to MiDaS v2.1 Small z enkoderem EfficientNet-Lite3. `l` to
DPT-Large z enkoderem ViT-L/16 i dekoderem DPT wprowadzonym przez MiDaS do
gęstej predykcji. Różni je także przetwarzanie wstępne. `s` stosuje zmianę
rozmiaru proporcji z górną granicą oraz normalizację średnią/odchyleniem
standardowym ImageNet. `l` stosuje minimalną zmianę rozmiaru proporcji ze
średnią i odchyleniem standardowym 0.5. `s` należy wybrać jako lżejszą sieć CNN,
a `l` dla dokładności dekodera transformerowego.

Ta rodzina nie obsługuje trenowania. `LibreMiDaS.train()` zawsze zgłasza
`NotImplementedError`.

## Walidacja

`val()` uruchamia wspólny walidator głębi. Dopasowuje każdą predykcję do jej
danych referencyjnych (ground truth), obliczając dla każdego obrazu skalę i
przesunięcie metodą najmniejszych kwadratów, po czym zwraca standardowe metryki
głębi względnej zero-shot: AbsRel, RMSE i trzy progi delta.

<code-tabs name="val" />

## Eksport

<export-matrix />

Wyeksportowany artefakt jest ponownie ładowany przez `LibreYOLO()` na podstawie
rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint
i zwraca ten sam obiekt `Results`, z `depth_map` zamiast ramek.

<code-tabs name="export" />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
