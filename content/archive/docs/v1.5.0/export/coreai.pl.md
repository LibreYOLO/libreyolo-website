---
title: Core AI
seo_title: Eksport do Apple Core AI z LibreYOLO
description: >-
  Eksport modelu LibreYOLO do zasobu .aimodel dla Apple Core AI: tylko macOS,
  stałe płótno, FP32 i kontrakt kolejności nazwanych wyjść, którego muszą
  przestrzegać konsumenci.
lead: >-
  Core AI to stos inferencji na urządzeniu od Apple. LibreYOLO przechwytuje
  model przez torch.export, obniża jego reprezentację konwerterem Core AI i
  zapisuje zasób .aimodel z metadanymi modelu oraz nazwami wyeksportowanych
  wyjść.
keywords:
  - eksport libreyolo do core ai
  - aimodel
  - coreai-torch
  - torch.export apple
  - inferencja na urządzeniu apple
  - coreai_output_names
last_verified: 1.5.0
meta:
  - label: Flaga
    value: export(format="coreai")
    mono: true
  - label: Zapisuje
    value: Jeden zasób .aimodel z dołączonymi metadanymi
  - label: Extra
    value: 'pip install "libreyolo[coreai]"'
    mono: true
  - label: Wczytywanie z powrotem
    value: >-
      Nie przez LibreYOLO. Konsumenci używają środowiska uruchomieniowego Core
      AI bezpośrednio.
  - label: Kształty
    value: Stałe płótno. dynamic=True zgłasza NotImplementedError.
  - label: Precyzja
    value: Tylko FP32. half=True i int8=True są odrzucane.
  - label: Wymaga
    value: >-
      macOS. Toolchain nigdzie indziej nie konwertuje ani nie uruchamia modeli,
      a coreai-torch przypina torch do 2.11.x.
verification: >-
  Odczytane z libreyolo/export/coreai.py, libreyolo/export/coreai_compat.py,
  libreyolo/export/exporter.py, libreyolo/export/support.py i pyproject.toml w
  gałęzi dev.
snippets:
  install:
    - label: 'Instalacja, na macOS'
      language: bash
      code: |
        # Celowo poza każdym zbiorczym extra: coreai-torch przypina torch
        # do 2.11.x i przeciągnąłby całe środowisko na tę wersję.
        pip install "libreyolo[coreai]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Zapisuje weights/LibreYOLO9t.aimodel
        path = model.export(format="coreai", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreai --imgsz 640
    - label: Argumenty
      language: python
      code: |
        model.export(
            format="coreai",
            imgsz=640,        # int lub (wysokość, szerokość); to jest płótno uruchomienia
            batch=1,
            output_path=None, # None zapisuje weights/<stem>.aimodel
        )

        # dynamic=True zgłasza NotImplementedError.
        # half=True i int8=True są odrzucane podczas walidacji.
  outputs:
    - label: Odczyt kolejności wyjść przed podłączeniem konsumenta
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")

        model.export(format="coreai", imgsz=640)


        # Metadane zasobu zapisują nazwy wyeksportowanych wyjść w kolejności
        grafu,

        # pod kluczem "coreai_output_names". Zwracany przez Core AI słownik
        mapuj

        # po nazwach z tej listy; nigdy nie paruj go pozycyjnie z krotką eager.
  support:
    - label: Sprawdzenie jednej rodziny i zadania przed eksportem
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: a35bfeafac6d6966
---

## Instalacja

Ten format działa tylko na macOS. Zależność `coreai-torch` ma marker
`sys_platform == 'darwin'`, a toolchain nigdzie indziej nie konwertuje ani nie
uruchamia modeli.

<code-tabs name="install" />

Ten extra znajduje się poza każdym zbiorczym extra, w tym `libreyolo[all]`,
ponieważ `coreai-torch` przypina torch do serii 2.11. Zainstaluj go w
środowisku, które chcesz ograniczyć do tej pary.

## Eksport

<code-tabs name="export" />

Przechwytywanie realizuje `torch.export`, czyli prawdziwe przechwycenie grafu z
warunkami guard, a nie pojedynczy zapisany ślad. Jest to ścieżka bardziej
restrykcyjna niż w Core ML: odczyty skalarów po stronie hosta i przepływ
sterowania zależny od danych są odrzucane, zamiast być po cichu zapiekane,
dlatego kilka rodzin jest tu zablokowanych z zapisanym błędem przechwytywania.

Trzy kroki przygotowawcze wykonują się wewnątrz zakresu, który przywraca żywy
model strony wywołującej niezależnie od tego, czy eksport się powiedzie, czy
nie. W rodzinach wywodzących się z Darknet normalizacja batch w trybie
inferencji jest dokładnie zwijana w poprzedzające sploty, ponieważ Core AI 0.4.1
nie zachowuje formuły Darknet z epsilonem po pierwiastku kwadratowym. W
rodzinach siatkowych i kotwicowych kotwice są zamrażane pod stałe płótno. W
RF-DETR embedding pozycyjny jest ponownie zapiekany pod żądane płótno przez
powtórne uruchomienie własnej ścieżki zapiekania modelu, ponieważ konwerter nie
ma obniżania dla `aten._upsample_bicubic2d_aa`.

Na etapie obniżania referencyjna dekompozycja PyTorch dla
`aten.grid_sampler_2d` jest włączana do tabeli dekompozycji, ponieważ konwerter
Core AI nie ma obniżania dla samplera deformable attention używanego przez
rodziny DETR.

Zasoby deklarują minimalną wersję systemu v27, czyli jedyną wartość, jaką
oferuje toolchain. Ogranicza to wdrożenie, a nie konwersję: konwersja i
wykonanie po stronie Pythona działają na wcześniejszych wersjach macOS dzięki
środowisku uruchomieniowemu wewnątrz pakietu wheel, ale wyniki numeryczne różnią
się między wersjami systemu, więc zapisany parytet jest mierzony na macOS 27.

## Uruchamianie artefaktu

W `libreyolo/backends` nie ma wpisu dla Core AI, więc `LibreYOLO()` nie wczytuje
pliku `.aimodel`. Konsumenci używają środowiska uruchomieniowego Core AI
bezpośrednio, a preprocessing, dekodowanie, NMS i przeskalowanie współrzędnych
leżą po ich stronie. Zwalidowany wiersz w macierzy wsparcia to deklaracja, że
wyeksportowany graf liczy te same wartości co referencja, a nie że uruchomi go
`predict`.

Jedyne, czego konsument nie jest w stanie odtworzyć samodzielnie, to kolejność
wyjść:

<code-tabs name="outputs" />

Core AI zwraca nazwany słownik, którego kolejność kluczy nie odpowiada ani
kolejności krotki z przejścia w przód w trybie eager, ani niczemu, co dałoby się
odgadnąć. Właśnie dlatego wyeksportowane nazwy są zapisywane w metadanych zasobu
jako `coreai_output_names`. Mapuj po nazwie.

## Ograniczenia

Stałe płótno, FP32, batch taki, jaki został wyeksportowany. `dynamic=True`
zgłasza `NotImplementedError`, a `half=True` i `int8=True` są odrzucane podczas
walidacji.

Pokrycie po stronie konwersji jest szerokie. Zwalidowane kombinacje obejmują
detekcję w rodzinach YOLO9, YOLOX, YOLO7, czterech detektorach z epoki Darknet,
YOLO-NAS, PicoDet, RTMDet, RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM, DEIMv2,
EC i RF-DETR; cztery rodziny klasyfikacji CNN oraz CLIP i SigLIP2 z zamrożonymi
klasami; Depth Anything V2 i ZipDepth; restaurację NAFNet i Real-ESRGAN;
segmentację semantyczną PIDNet i LingBotVision; a także detekcję punktów FOMO.
Każda ma własny zapisany kontekst, który wypisuje `libreyolo formats`.

Zablokowane, z przyczyną zapisaną dla każdej kombinacji:

| Kombinacja | Dlaczego |
|---|---|
| Segmentacja semantyczna EoMT | Ścisłe przechwytywanie kończy się błędem `GuardOnDataDependentSymNode`: coś w ścieżce masek odczytuje wartość z tensora i rozgałęzia się na jej podstawie |
| Segmentacja semantyczna SegFormer | Ścieżka przechwytywania nie została oceniona, a opublikowane wagi i tak są niekomercyjne niezależnie od formatu |
| Estymacja wzroku L2CS | Sam model obsługuje wyłącznie ONNX, TorchScript, ExecuTorch, TensorRT i OpenVINO, co jest decyzją po stronie modelu |
| Estymacja głębi Depth Anything 3 | Rodzina odrzuca eksport dla każdego formatu |

Z RF-DETR wiąże się jedno zastrzeżenie, które warto przeczytać przed
porównywaniem artefaktów. Parytet jest tu zapisany względem grafu, który
przygotowuje sam eksporter Core AI, a nie względem ONNX, i przy płótnie 640
artefakt ONNX dla RF-DETR rozjeżdża się z tym przygotowanym grafem. Ponowne
zapiekanie w Core AI zachowuje skalowanie z antyaliasingiem wykonywane przez
model w trybie eager, podczas gdy ścieżka ONNX antyaliasing wyłącza. ONNX nie
jest więc poprawną referencją dla tej rodziny przy płótnie innym niż natywne.

Wcześniejszy format Apple opisuje [Core ML](/docs/export/coreml). Pełną siatkę
rodzin i zadań zawiera [macierz eksportu](/docs/reference/export-matrix). Dla
jednej kombinacji:

<code-tabs name="support" />
