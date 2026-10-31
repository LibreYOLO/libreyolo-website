---
title: RKNN
seo_title: Eksport do RKNN dla NPU Rockchip
description: >-
  Kompilacja detektora LibreYOLO do artefaktu .rknn firmy Rockchip: SDK
  producenta instalowane samodzielnie, cztery zwalidowane warianty RK3588 i
  parytet w symulatorze.
lead: >-
  RKNN to kompilowany format NPU firmy Rockchip. LibreYOLO eksportuje pośredni
  plik ONNX w opset 19, kompiluje go za pomocą SDK RKNN Toolkit2 i potrafi
  porównać skompilowany graf z ONNX Runtime w symulatorze hosta z Toolkit2, bez
  udziału płytki.
keywords:
  - eksport yolo do rknn
  - rockchip npu
  - rk3588
  - rknn-toolkit2
  - parytet symulatora rknn
  - inferencja orange pi rockchip
last_verified: 1.5.0
meta:
  - label: Flaga
    value: 'export(format="rknn", name="rk3588")'
    mono: true
  - label: Zapisuje
    value: >-
      Jeden plik .rknn, plik sidecar .rknn.metadata.json oraz raport
      .rknn.parity.json, gdy verify=True
  - label: Extra
    value: Brak na PyPI. rknn-toolkit2 to SDK producenta instalowane samodzielnie.
  - label: Ponowne wczytanie
    value: >-
      Nie przez LibreYOLO. Artefakt uruchamia się na płytce w środowisku
      uruchomieniowym Rockchip.
  - label: Kształty
    value: 'Stały kwadrat, batch 1, opset 19. Wszystkie trzy są wymuszane.'
  - label: Precyzja
    value: >-
      Zmiennoprzecinkowa kompilacja producenta. half=True i int8=True są
      odrzucane.
  - label: Zakres
    value: >-
      Cztery warianty detekcji na RK3588: YOLO9-t, YOLO9-E2E-t, PicoDet-s i
      YOLO-NAS-s
verification: >-
  Odczytane z libreyolo/export/rknn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py i docs/rknn.md na gałęzi dev. Zmierzone wartości
  parytetu pochodzą z zapisu walidacji z 2026-08-04 w docs/rknn.md.
snippets:
  install:
    - label: Po stronie LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'SDK producenta, instalowane samodzielnie'
      language: bash
      code: >
        # rknn-toolkit2 to SDK firmy Rockchip na osobnej licencji. LibreYOLO

        # ani go nie dołącza, ani nie instaluje. Tylko Linux x86_64; w systemie

        # Windows należy użyć WSL2 lub kontenera z Linuksem.

        #

        # Toolkit2 2.3.2 wymaga setuptools<81 i zawodzi na ONNX 1.19 lub
        nowszym,

        # w którym usunięto onnx.mapping, wciąż importowane przez jego
        kompilator.

        pip install "setuptools==80.9.0" "onnx==1.18.0"


        # Następnie należy zainstalować pasujący pakiet wheel rknn-toolkit2 z

        # własnego repozytorium wheeli Rockchip i potwierdzić, że się importuje:

        python -c "import rknn.api; print('rknn-toolkit2 ready')"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Zapisuje weights/LibreYOLO9t.rknn i
        weights/LibreYOLO9t.rknn.metadata.json

        path = model.export(format="rknn", name="rk3588", imgsz=640,
        verify=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format rknn --name rk3588 \
          --imgsz 640 --verify
    - label: Argumenty
      language: python
      code: |
        model.export(
            format="rknn",
            name="rk3588",     # platforma docelowa; target= i target_platform= też działają
            imgsz=640,         # musi odpowiadać zarejestrowanemu płótnu wariantu
            batch=1,           # każda inna wartość zgłasza NotImplementedError
            dynamic=False,     # True zgłasza ValueError
            opset=19,          # każda inna wartość zgłasza NotImplementedError
            verify=False,      # True uruchamia symulator PC i uzależnia wynik od parytetu
        )
  parity:
    - label: Parytet bez płytki względem istniejącego artefaktu ONNX
      language: python
      code: |
        import numpy as np
        from libreyolo.export import verify_rknn_simulator_parity

        input_tensor = np.random.default_rng(0).standard_normal(
            (1, 3, 640, 640), dtype=np.float32
        )
        metrics = verify_rknn_simulator_parity(
            "weights/LibreYOLO9t.onnx",
            input_tensor,
            target_platform="rk3588",
            rtol=1e-3,
            atol=1e-4,
            raise_on_failure=False,
        )
        print(metrics)
  support:
    - label: Sprawdzenie jednej rodziny i zadania przed kompilacją
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c659713cc3c8cc9e
---

## Instalacja

Kompilacja wymaga RKNN Toolkit2 od Rockchip, który jest rozprowadzany jako SDK
producenta na własnej licencji Rockchip i nie jest zależnością LibreYOLO. Nie ma
dodatku `libreyolo[rknn]` i nic w tym formacie nie instaluje się jedną linią.

<code-tabs name="install" />

Do kompilacji ani do sprawdzenia parytetu numerycznego płytka nie jest potrzebna.
Płytka RK3588 jest potrzebna do pomiarów opóźnienia, poboru mocy i temperatury, z
których żaden nie został zarejestrowany.

## Eksport

<code-tabs name="export" />

Żądanie jest sprawdzane względem listy dokładnych wariantów modeli, zanim
cokolwiek zostanie skompilowane, a płótno jest sprawdzane tak samo: podanie
`imgsz` innego niż to, przy którym wariant został zarejestrowany, zgłasza błąd,
zamiast po cichu kompilować coś nieprzetestowanego. LibreYOLO zapisuje pośredni
plik ONNX w opset 19, kompiluje go, opcjonalnie symuluje i na koniec usuwa plik
pośredni.

Metadane trafiają do pliku sidecar o nazwie `<model>.rknn.metadata.json`,
ponieważ format RKNN nie ma przenośnego pola metadanych.

`verify=True` uruchamia symulator PC z Toolkit2 w tej samej sesji, w której
artefakt został skompilowany, porównuje każde wyjście z ONNX Runtime na tym samym
wejściu i zapisuje `<model>.rknn.parity.json` z metrykami błędu dla poszczególnych
wyjść. Progami akceptacji są podobieństwo kosinusowe co najmniej 0.9999 i
znormalizowany RMSE co najwyżej 0.02, stosowane do każdego wyjścia, które nie jest
już zbliżone element po elemencie; zmiennoprzecinkowa kompilacja producenta
obniża wewnętrzne tensory do połowicznej precyzji, więc ścisłe `allclose` nie
zachodzi nawet wtedy, gdy zdekodowane ramki są stabilne. Nieudany przebieg
zapisuje `<model>.rknn.failed.parity.json`, odrzuca kandydata i pozostawia
nietkniętym każdy wcześniejszy udany eksport pod tą ścieżką.

Aby porównać artefakt ONNX, który już jest dostępny, bez ponownego eksportu:

<code-tabs name="parity" />

Symulator Toolkit2 uruchamia graf w pamięci wytworzony przez `load_onnx` i
`build`. Nie potrafi ponownie wczytać pliku `.rknn` związanego z konkretnym celem
bez płytki, dlatego `verify=True` wykonuje kompilację, eksport i symulację w
jednej sesji.

## Uruchamianie artefaktu

W `libreyolo/backends` nie ma wpisu dla RKNN, więc `LibreYOLO()` nie wczytuje
pliku `.rknn`. Skompilowany artefakt jest wdrażany na płytkę i wykonywany przez
własne środowisko uruchomieniowe (runtime) Rockchip, a wstępne przetwarzanie,
dekodowanie, NMS i przeskalowanie współrzędnych są tam odpowiedzialnością
aplikacji.

`<model>.rknn.metadata.json` niesie nazwy klas, rozmiar wejścia, zadanie i
platformę docelową, czyli to, czego aplikacja potrzebuje, aby odtworzyć
postprocessing LibreYOLO. Należy dostarczać go razem ze skompilowanym modelem.

Do sprawdzenia po stronie hosta, które nie wymaga płytki, warto zachować artefakt
ONNX o tym samym stałym kształcie i porównać go w symulatorze, jak wyżej.

## Ograniczenia

Kompilują się cztery kombinacje i są to warianty modeli, a nie rodziny:

| Wariant | Zadanie | Płótno | Cel |
|---|---|---:|---|
| YOLO9-t | detect | 640 | RK3588 |
| YOLO9-E2E-t | detect | 640 | RK3588 |
| PicoDet-s | detect | 320 | RK3588 |
| YOLO-NAS-s | detect | 640 | RK3588 |

Wszystko inne jest odrzucane przed kompilacją, z komunikatem, że RKNN w tej
wersji ogranicza się do dokładnych wariantów detekcji przetestowanych w
symulatorze. Wyniki dla innych modeli, ograniczone do samej kompilacji, istnieją,
ale celowo nie są przedstawiane jako wsparcie: w tym samym przebiegu pomiarowym
RF-DETR zostawił dwa węzły `GridSample` dekodera bez loweringu, a D-FINE,
RT-DETR, RT-DETRv2, RT-DETRv4, DEIM, DEIMv2 i EC skompilowały się i zasymulowały
ze zdekodowanymi wyjściami, które były istotnie błędne.

Batch 1, statyczne kształty, opset 19. `half=True` jest odrzucane, ponieważ RKNN
nie udostępnia kontraktu `half` z LibreYOLO, a `int8=True` jest odrzucane, dopóki
nie pojawi się reprezentatywna kalibracja i wyniki dokładności dla zadania.

Inne cele Rockchip są odrzucane: `rk3588` to jedyna zwalidowana platforma.

Pełną siatkę rodzin i zadań zawiera
[macierz eksportu](/docs/reference/export-matrix). Dla jednej kombinacji:

<code-tabs name="support" />
