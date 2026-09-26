---
title: PicoSAM3
families:
  - picosam3
seo_title: 'PicoSAM3: segmentacja krawędziowa sterowana ramką w LibreYOLO'
description: >-
  Używaj PicoSAM3 w LibreYOLO do segmentacji regionu sterowanej ramką na
  czujnikach brzegowych. Instaluj, przewiduj i eksportuj checkpoint pico na
  licencji Apache-2.0.
lead: >-
  PicoSAM3 to kompaktowa sieć CNN destylowana z SAM 2.1 i SAM 3, zbudowana do
  segmentacji obszaru zainteresowania sterowanej ramką na czujnikach takich jak
  Sony IMX500. LibreYOLO obsługuje ją przez osobną fabrykę LibreSAM, niezależną
  od fabryki detektorów LibreYOLO(), wyłącznie z promptami ramkowymi.
keywords:
  - PicoSAM3
  - Segment Anything
  - segmentacja brzegowa
  - obszar zainteresowania
  - prompt ramką
  - inferencja w czujniku
  - IMX500
  - destylacja wiedzy
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt ramkowy
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # PicoSAM3 ma jeden rozmiar, "pico", dlatego nie jest potrzebny inny
        alias.

        model = LibreSAM("picosam3")


        # bboxes= jest jedynym obsługiwanym promptem: [x1, y1, x2, y2] lub lista

        # ramek, jedna maska na ramkę. Każda ramka jest rozszerzana o 10%,
        zamieniana

        # na kwadrat, przycinana do obrazu i skalowana do 96x96 przed
        uruchomieniem CNN.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        print(result.masks.xy)      # wielokąt dla każdej maski

        print(result.boxes.xyxy)    # ciasna ramka wyznaczona z maski
    - label: 'Jedno kodowanie, wiele promptów'
      language: python
      code: >
        from libreyolo import LibrePicoSAM3, SAMPLE_IMAGE


        model = LibrePicoSAM3()


        # set_image() zapisuje obraz źródłowy w pamięci podręcznej. PicoSAM3
        wykonuje pełny

        # przebieg CNN dla każdej ramki, więc oszczędza to
        wczytywanie/dekodowanie obrazu,

        # a nie przebieg enkodera jak w pozostałych rodzinach SAM.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(bboxes=[300, 200, 900, 700])

        b = model.predict(bboxes=[100, 100, 400, 400])

        model.reset_image()
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibrePicoSAM3

        model = LibrePicoSAM3()
        model.export(format="onnx", output_path="LibrePicoSAM3pico.onnx")

        # opset (domyślnie 13) i dynamic (domyślnie True, tylko oś batcha) są
        # jedynymi argumentami eksportu przyjmowanymi przez tę rodzinę.
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # PicoSAM3 eksportuje surową sieć CNN ROI 96x96: roi_image ->
        mask_logits.

        # Nie ma tu przetwarzania wstępnego/końcowego LibreYOLO do ponownego
        użycia,

        # ponieważ export() nie kieruje wyniku z powrotem przez LibreYOLO() tak
        jak

        # checkpoint detektora.

        session = ort.InferenceSession("LibrePicoSAM3pico.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 96, 96),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 5d60ff14fe61ba29
---

## Instalacja

PicoSAM3 wymaga dodatku `sam`. Własne pobieranie wag LibreYOLO nadal korzysta z
narzędzi Hugging Face w `transformers`, mimo że inferencja działa na natywnej
sieci CNN niezależnej od `transformers`.

```bash
pip install "libreyolo[sam]"
```

## Predykcja

`LibreSAM(...)` (lub właściwy dla tej rodziny `LibrePicoSAM3(...)`) jest
osobnym punktem wejścia niż `LibreYOLO(...)`. Zwraca segmenter sterowany
promptami, a nie detektor, ponieważ przebieg w przód nie ma tu znaczenia bez
promptu. Dla tej rodziny nie istnieje polecenie CLI `libreyolo predict`. Należy
użyć API Pythona.

<code-tabs name="predict" />

PicoSAM3 przyjmuje wyłącznie `bboxes=`. Przekazanie `points=`, `labels=`,
`masks=`, `text=`, `multimask=True` albo pominięcie ramki, aby segmentować
wszystko, zgłasza czytelny `ValueError`, ponieważ żaden z tych trybów nie
istnieje w modelu źródłowym. `conf` filtruje według przewidywanej jakości maski
(IoU), a nie pewności detekcji, i musi mieścić się między `0.0` a `1.0`. Każda
maska ma identyfikator klasy `0` o nazwie `"object"`. Wywołania `train()`,
`val()` i `track()` zgłaszają `NotImplementedError`. Do promptów punktowych,
tekstowych, maskowych lub segmentujących wszystko należy użyć LibreSAM2 albo
LibreSAM3. Informacje o typach źródeł zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępny jest jeden rozmiar, pico, ze stałym wejściem ROI 96 px. PicoSAM3
wykonuje jeden pełny przebieg CNN dla każdej ramki zamiast jednorazowo kodować
cały obraz.

## Eksport

<export-matrix />

PicoSAM3 jest jedyną rodziną poziomu SAM, którą można eksportować. Udostępnia
surową sieć CNN ROI 96x96 w ONNX, `roi_image -> mask_logits`, bez NMS ani
przetwarzania końcowego maski wbudowanego w graf. Pozostałe rodziny SAM
zgłaszają `NotImplementedError` dla `export()`, ponieważ ich podział na enkoder
i dekoder nie ma jeszcze zdefiniowanego kontraktu eksportu środowiska
uruchomieniowego. Wyeksportowany graf PicoSAM3 nie jest ponownie wczytywany przez
`LibreYOLO()`. Należy uruchomić go bezpośrednio w środowisku takim jak
`onnxruntime`, stosując to samo przetwarzanie wstępne kwadratowego ROI z
dopełnieniem 10%, które pokazano powyżej.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box>

PicoSAM3 jest destylowany z modeli nauczycieli SAM 2.1 i SAM 3. LibreYOLO nie
dołącza ani nie redystrybuuje kodu lub wag żadnego z nauczycieli w tej rodzinie.
Udostępniana jest tylko kompaktowa sieć CNN ucznia i jej przekonwertowany
checkpoint.

</provenance-box>

## Cytowanie

<citation-block />
