---
title: Kwantyzacja
seo_title: Kwantyzacja modelu LibreYOLO w PyTorch
description: >-
  API kwantyzacji LibreYOLO w PyTorch: dziewięć przepisów, kalibracja trzymana z
  dala od danych treningowych, QAT i QAD oraz dwa artefakty do wdrożenia.
lead: >-
  Kwantyzacja w LibreYOLO działa w całości w PyTorch: model.quantize() zastępuje
  moduły Conv2d i Linear modelu ich skwantyzowanymi odpowiednikami i kalibruje
  je. Wynik zachowuje zwykły kontrakt predict, val, train i save, więc
  skwantyzowany model jest oceniany przez te same walidatory co model float.
keywords:
  - kwantyzacja libreyolo
  - kwantyzacja int8 yolo
  - trenowanie z uwzględnieniem kwantyzacji
  - qat qad
  - nvfp4 mxfp4
  - fp8 e4m3
  - dane kalibracyjne kwantyzacja
  - eksport qdq onnx
last_verified: 1.5.0
meta:
  - label: Wywołanie
    value: 'model.quantize(recipe="int8", calib="coco128.yaml")'
    mono: true
  - label: Polecenie
    value: libreyolo quantize --model M.pt --recipe int8 --calib coco128.yaml
    mono: true
  - label: Extra
    value: Brak. Kwantyzacja działa w PyTorch.
  - label: Rodziny
    value: 'yolo9, rfdetr, birefnet, feynobg'
  - label: Przepisy
    value: 'fp16, bf16, fp8, int8, w4a16, w4a8, nvfp4, mxfp4, int2'
    mono: true
  - label: Artefakty do wdrożenia
    value: >-
      export(format="pt") dla spakowanego checkpointu, export(format="onnx") dla
      grafu QDQ INT8
    mono: true
verification: >-
  Odczytane z libreyolo/quant/api.py, libreyolo/models/base/model.py,
  libreyolo/cli/commands/quantize.py oraz docs/quantization.md w gałęzi dev.
  Podane rozmiary checkpointów to wartości zmierzone i zapisane w
  docs/quantization.md.
snippets:
  quantize:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Podmiana struktury plus kalibracja. calib to niewielki zbiór obrazów
        BEZ ETYKIET,

        # czytany tylko w przejściu w przód, aby wyznaczyć zakresy i skale
        aktywacji.

        qmodel = model.quantize(recipe="int8", calib="coco128.yaml",
        samples=128)


        print(qmodel.quant_info())

        qmodel.val(data="coco8.yaml")          # te same walidatory co dla
        modelu float

        qmodel.save("LibreYOLO9s-int8.pt")     # checkpoint niesie manifest
        quant
    - label: CLI
      language: bash
      code: >
        libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib
        coco128.yaml
    - label: Argumenty
      language: python
      code: |
        model.quantize(
            recipe="int8",
            calib="coco128.yaml",      # ścieżka do data.yaml lub nazwa wbudowana; None pomija kalibrację
            samples=128,               # maksymalna liczba obrazów kalibracyjnych
            batch=8,                   # rozmiar batcha przy kalibracji
            algorithm="auto",          # auto i minmax to to samo; percentile to alternatywa
            keep_high_precision=None,  # None używa polityki rodziny
            verbose=True,
        )
  reload:
    - label: Skwantyzowany checkpoint wczytuje się jako skwantyzowany
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Manifest quant odtwarza skwantyzowaną strukturę i skale
        # przed wczytaniem wag.
        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        print(qmodel.quant_info())
  train:
    - label: QAT to zwykłe train() na skwantyzowanym modelu
      language: python
      code: >
        from libreyolo import LibreYOLO


        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")


        # Dostrajanie, a nie przebieg od zera: użyj współczynników uczenia dla
        dostrajania.

        qmodel.train(data="coco8.yaml", epochs=5, lr0=1e-4)
    - label: QAD dodaje istniejące argumenty destylacji
      language: python
      code: |
        qmodel.train(
            data="coco8.yaml",
            epochs=5,
            lr0=1e-4,
            distill_model="LibreYOLO9m.pt",
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train --model LibreYOLO9s-int8.pt --data coco8.yaml --epochs 5
        --lr0 1e-4
  export:
    - label: Spakowany checkpoint PyTorch
      language: python
      code: >
        from libreyolo import LibreYOLO


        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")


        # Zapisuje LibreYOLO9s-int8-final.pt: spakowane wagi niskobitowe i
        skale,

        # usunięte wagi nadrzędne fp32, nieskwantyzowana reszta rzutowana na
        fp16.

        qmodel.export(format="pt")


        # remainder="fp32" zachowuje nieskwantyzowane tensory bez zmian.

        qmodel.export(format="pt", remainder="fp32")
    - label: QDQ INT8 ONNX
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # Pary QuantizeLinear/DequantizeLinear w grafie, niosące własne
        # skale modelu, skalibrowane lub wytrenowane w QAT.
        qmodel.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9s-int8.pt --format onnx
  dequantize:
    - label: Powrót do float z zachowaniem wag wytrenowanych w QAT
      language: python
      code: >
        from libreyolo import LibreYOLO


        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        qmodel.dequantize()


        # Teraz działa dowolny eksporter float, w każdej obsługiwanej przez
        niego precyzji.

        qmodel.export(format="tensorrt", half=True)
source_hash: 4ffb06b87cad017e
---

## Instalacja

Kwantyzacja nie wymaga żadnego dodatku. Podmiana modułów, przebieg kalibracji
i symulowana arytmetyka działają w całości w PyTorch, więc jedynym wymaganiem
jest `pip install libreyolo`. Artefakty do wdrożenia potrzebują tego, czego
potrzebuje ich własny format, czyli w ścieżce ONNX `libreyolo[onnx]`.

## Kwantyzacja

<code-tabs name="quantize" />

`quantize()` przekształca wczytany model w miejscu i zwraca go. Gradienty nie
biorą w tym udziału: podmiana instaluje skwantyzowane moduły, a przebieg
kalibracji działa tylko w przód.

Powstały checkpoint to zwykły checkpoint LibreYOLO z dołączonym manifestem
`quant`, więc wczytuje się ponownie z nienaruszoną strukturą i skalami:

<code-tabs name="reload" />

Checkpointy zapisywane przez moduł trenowania podczas przebiegu QAT również
niosą ten manifest, co oznacza, że `best.pt` z takiego przebiegu sam jest
skwantyzowanym checkpointem.

## Przepisy

Obsługiwane są cztery rodziny: `yolo9`, `rfdetr`, `birefnet` i `feynobg`.

| Przepis | Co robi | Rodziny | Kalibracja |
|---|---|---|---|
| `fp16` | Rzutowanie na połowiczną precyzję z kontraktem wejścia i wyjścia w float32. Tylko inferencja. | wszystkie cztery | brak |
| `bf16` | Rzutowanie na bfloat16, które zachowuje zakres wykładnika z float32. Rozwiązanie na przepełnienie fp16 w modelu typu DETR. Tylko inferencja. | wszystkie cztery | brak |
| `fp8` | Wagi i aktywacje E4M3 w `Conv2d` i `Linear`: skale wag na kanał, kalibrowane skale aktywacji na tensor. | wszystkie cztery | wymagana |
| `int8` | W8A8 w `Conv2d` i `Linear`: symetryczne wagi na kanał, afiniczne aktywacje na tensor. | wszystkie cztery | wymagana, albo `calib=None` dla samych wag |
| `w4a16` | Grupowane symetryczne wagi INT4, grupa 128 wzdłuż `in_features`, aktywacje float, w `Linear`. | rfdetr, birefnet, feynobg | niepotrzebna |
| `w4a8` | Grupowane wagi INT4 plus kalibrowane aktywacje INT8, w `Linear`. | rfdetr, birefnet, feynobg | wymagana |
| `nvfp4` | W4A4 NVFP4 w `Linear`: elementy E2M1, bloki 16-elementowe, skale bloków FP8 E4M3, skala tensora FP32. Dynamiczne skalowanie aktywacji. | rfdetr, birefnet, feynobg | niepotrzebna |
| `mxfp4` | OCP MXFP4 w `Linear`: elementy E2M1, bloki 32-elementowe, skale bloków E8M0 będące potęgami dwójki. Dynamiczne skalowanie aktywacji. | rfdetr, birefnet, feynobg | niepotrzebna |
| `int2` | Wyłącznie do badań: grupowane wagi 2-bitowe, grupa 64, plus aktywacje INT8, w `Linear`. Sama kwantyzacja po trenowaniu jest bezużyteczna, więc wymagane jest QAT lub QAD. | rfdetr | wymagana |

Przepisy poniżej 8 bitów celują w `nn.Linear` i są celowo odrzucane dla
`yolo9`: na obecnym sprzęcie to przyspieszenie dotyczy wyłącznie GEMM, więc
sploty pozostają w wyższej precyzji. YOLO9 korzysta z `int8` lub `fp8`. `int2`
jest odrzucany dla `birefnet` i `feynobg`, ponieważ te rodziny służą wyłącznie
do inferencji, więc naprawcze trenowanie QAT, od którego ten przepis zależy,
jest tam niedostępne.

Wartości domyślne każdej rodziny zostawiają pierwszą warstwę i głowice w float,
a splot DFL w YOLO9 nigdy nie jest kwantyzowany: to stały operator całkowej
wartości oczekiwanej. Gdy jest ku temu powód, można to nadpisać przez
`keep_high_precision=("head.",)`.

## Dane kalibracyjne to nie dane treningowe

`calib=` przyjmuje kilkaset obrazów, nie czyta etykiet i wykonuje wyłącznie
przejście w przód, aby oszacować zakresy aktywacji. `data=` w `train()` i
`val()` to zbiór danych z etykietami, używany do gradientów i metryk. To różne
argumenty o różnym przeznaczeniu, a wartością domyślną dla `calib` jest
`coco128.yaml`.

`algorithm="minmax"` zachowuje bezwzględne ekstrema zaobserwowane we wszystkich
batchach kalibracyjnych i to właśnie wybiera `"auto"`. `"percentile"` używa
średniej z percentyli 0.1 i 99.9 liczonych dla poszczególnych batchy; pomiary
pokazały, że załamuje on dokładność w rodzinie DETR, ponieważ wartości odstające
w aktywacjach transformerów pełnią funkcję nośną. Wrażliwość małych modeli na
INT8 naprawia w rzeczywistości kalibracja na wystarczającej liczbie batchy: przy
domyślnym `coco128` YOLO9-t mieści się w granicach około jednego punktu mAP od
swojego wyniku float. Wybrany algorytm jest zapisywany w manifeście
checkpointu.

## Odzyskiwanie dokładności

<code-tabs name="train" />

Skwantyzowane moduły przechowują nadrzędne wagi fp32 i stosują sztuczną
kwantyzację (fake quantization) z estymatorem straight-through, więc gradienty
docierają do wag nadrzędnych, a istniejące moduły trenowania działają bez zmian:
EMA, AMP, wznawianie z checkpointu i argumenty destylacji dają się łączyć.

QAT to dostrajanie już wytrenowanego modelu. Należy używać współczynników
uczenia właściwych dla dostrajania, a nie wartości domyślnych dla trenowania od
zera, w przeciwnym razie krótki przebieg zniszczy wstępnie wytrenowane wagi
niezależnie od kwantyzacji. Dostępność QAD idzie za obsługą destylacji w danej
rodzinie, co dziś oznacza `yolo9` i `rfdetr`.

Modele skwantyzowane przepisami `fp16` i `bf16` służą wyłącznie do inferencji,
a moduł trenowania odrzuca je, wskazując na `amp=True`.

## Eksport

<code-tabs name="export" />

`format="pt"` krystalizuje model. Spakowane wagi niskobitowe i skale zastępują
wagi nadrzędne, a nieskwantyzowana reszta jest rzutowana na fp16, chyba że
przekazano `remainder="fp32"`. Niezmiennikiem pakowania jest to, że rozpakowanie
odtwarza symulację bit w bit na urządzeniu, na którym przeprowadzono
finalizację, więc sfinalizowany plik osiąga dokładnie ten wynik, który
zwalidowano. Wartości zmierzone: YOLO9-s int8 schodzi z 29.5 MB do 9.6 MB,
RF-DETR-n nvfp4 ze 122 MB do 26 MB. Wczytanie takiego pliku daje model gotowy do
inferencji, a wywołanie na nim `train()` automatycznie odtwarza wagi nadrzędne
ze spakowanych wag.

`format="onnx"` dotyczy modeli `int8` i generuje graf QDQ niosący własne skale
modelu, skalibrowane lub wytrenowane w QAT, które ONNX Runtime i TensorRT
wykonują na prawdziwych jądrach INT8. To inna ścieżka niż
[`export(format="onnx", int8=True)`](/docs/export/onnx) na modelu float, gdzie
ONNX Runtime sam wyznacza skale.

Przepisy rzutujące typ nie potrzebują żadnego eksportera dla modeli
skwantyzowanych:

<code-tabs name="dequantize" />

## Ograniczenia

Skwantyzowana arytmetyka wykonuje się w symulacji, czyli jako sztuczna
kwantyzacja liczona w wyspach float32, nawet pod AMP. Symulacja jest wierna
numerycznie, więc wynik `val()` na dowolnym urządzeniu jest prawdziwym
stwierdzeniem o skwantyzowanej arytmetyce. Nie jest stwierdzeniem o szybkości.

Dwa wyjątki wykonują się natywnie. `fp16` i `bf16` to zwykłe rzutowania typów.
Sfinalizowane moduły `fp8` wykonują swój GEMM bezpośrednio na spakowanych wagach
E4M3 przez `torch._scaled_mm` na sprzęcie klasy Ada, Hopper i Blackwell, używając
tych samych skalibrowanych skal aktywacji co symulacja; ustawienie
`LIBREYOLO_KERNELS=off` przywraca wszędzie dokładnie tę symulowaną ścieżkę.

Zakres wdrożeń jest węższy niż lista przepisów. Tylko `int8` ma tu postać ONNX
nadającą się do wdrożenia; `fp8` i przepisy liniowe poniżej 8 bitów wykonują się
w PyTorch i krystalizują przez `format="pt"`. Żądanie eksportu do ONNX dla nich
kończy się wyjątkiem z tą właśnie wskazówką, podobnie jak żądanie formatu innego
niż ONNX dla modelu `int8`: silniki na dalszych etapach należy budować z grafu
QDQ.

Eksport modelu `int8`, którego aktywacje nigdy nie zostały skalibrowane, zapisuje
ostrzeżenie w logach i daje graf niosący samą kwantyzację wag.
