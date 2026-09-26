---
title: D-FINE
families:
  - dfine
seo_title: 'D-FINE: dostrajanie, walidacja i eksport na licencji MIT'
description: >-
  Używaj D-FINE w LibreYOLO do detekcji obiektów i segmentacji instancji.
  Instaluj, uruchamiaj predykcję, dostrajaj, waliduj i eksportuj za pomocą kodu
  na licencji MIT.
lead: >-
  Transformer detekcyjny, który przedstawia regresję ramki jako rozkład
  prawdopodobieństwa dla każdej jej krawędzi, udoskonalany między warstwami
  dekodera. LibreYOLO obsługuje go w detekcji i segmentacji instancji.
keywords:
  - D-FINE
  - transformer detekcyjny
  - detekcja obiektów w czasie rzeczywistym
  - segmentacja instancji
  - fine-grained distribution refinement
  - DETR
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDFINEn.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Segmentacja instancji
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sufiks -seg w nazwie pliku wybiera głowicę masek, więc argument
        # zadania nie jest tutaj potrzebny.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDFINEn.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Segmentacja instancji
      language: bash
      code: |
        # Kontynuuje od opublikowanych wag segmentacji wraz z głowicą masek.
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: Segmentacja z wag detekcji
      language: bash
      code: >
        # Wagi detekcji nie zawierają głowicy masek, więc jest to jawny
        transfer:

        # głowica zaczyna bez wytrenowania i staje się użyteczna dopiero po
        trenowaniu.

        # Podanie tutaj task=segment autoryzuje transfer.

        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn.pt data=my-dataset.yaml
    - label: Segmentacja instancji
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # maski
        print(metrics["metrics/mAP50-95(B)"])   # ramki
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDFINEn.pt format=onnx imgsz=640

        libreyolo export model=LibreDFINEn.pt format=tensorrt imgsz=640
        half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie sufiksu pliku, więc
        wyeksportowany

        # artefakt wczytuje się jak każdy checkpoint i zwraca ten sam obiekt
        Results.

        model = LibreYOLO("LibreDFINEn.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 0216631a26185524
---

## Instalacja

D-FINE nie wymaga opcjonalnego dodatku. Wszystkie importowane elementy znajdują
się w instalacji bazowej.

```bash
pip install libreyolo
```

Wyjątkiem jest dostrajanie adapterów z `lora=True`, które wymaga dodatku `lora`.

```bash
pip install "libreyolo[lora]"
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane lokalnie
w pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, dlatego zamiana
detektora wymaga zmiany jednego wiersza. Nazwa pliku z sufiksem `-seg`
automatycznie wybiera zadanie segmentacji, a `result.masks` zawiera wtedy maski
instancji obok ramek. Argumenty `conf` i `max_det` filtrują wybór zapytań.
Argument `iou` jest akceptowany dla zgodności API, lecz nie ma wpływu na wynik,
ponieważ dekoder jest predyktorem zbioru bez etapu NMS. Zobacz stronę
[predykcji](/docs/predict), aby poznać źródła, streaming i obsługę wyników.

## Warianty

Dostępnych jest pięć rozmiarów. Wszystkie działają z tą samą rozdzielczością
wejściową, dlatego tabela rozróżnia je według liczby parametrów i accuracy.

<benchmark-table task="detect" />

<va-embed />

Segmentacja ponownie wykorzystuje backbone, enkoder i dekoder detekcji oraz
dodaje głowicę masek, dlatego checkpoint `-seg` przyjmuje te same argumenty co
odpowiadający mu checkpoint detekcji. Rodzina RT-DETRv4 w LibreYOLO została
zaimplementowana jako podklasa otoki D-FINE. Dziedziczy tę linię dekodera,
a następnie ogranicza listę zadań z powrotem do detekcji, ponieważ nie zawiera
głowicy masek.

## Trenowanie

Dla obu zadań trenowanie rozpoczyna się od opublikowanego checkpointu.

<code-tabs name="train" />

Przy ustawieniach domyślnych moduł trenujący wykonuje 132 epoki z `lr0=2e-4`,
`amp=False`, batchem 16 i early stopping po 50 epokach bez poprawy. Wagi detekcji
są prawidłowym punktem wyjścia do trenowania segmentacji, ale wyłącznie jako
jawny transfer. Głowica masek zaczyna bez wytrenowania i w przeciwnym razie
zwracałaby bezwartościowe maski. Podanie `task=segment` w CLI autoryzuje ten
transfer. Ścieżka Pythona jest bardziej ograniczona: klasę `LibreDFINE` trzeba
utworzyć bezpośrednio z `allow_detect_to_segment_transfer=True`, ponieważ
fabryka `LibreYOLO()` nie przyjmuje takiego argumentu. Bezpośrednie utworzenie
nie pobiera pliku, więc wagi muszą już znajdować się na dysku.

Ustawienie `lora=True` dotyczy detekcji. Trenowanie segmentacji je odrzuca
i wskazuje zamiast niego `freeze='backbone'`, ponieważ głowica masek nie została
przetestowana z adapterami. Na urządzeniach Apple silicon moduł trenujący
przenosi całe uruchomienie na CPU. Przebieg wsteczny skwantowanego mnożenia
macierzy w module Integral powoduje błąd kompilacji Metal. Nie ma to wpływu na
inferencję w MPS.

Zobacz stronę [trenowania](/docs/train), aby poznać zbiory danych, augmentację,
obsługę wielu GPU i loggery.

## Walidacja

Metoda `val()` zwraca słownik indeksowany nazwą metryki i wyświetla wyniki dla
poszczególnych klas, gdy pozostawiono włączone `verbose`.

<code-tabs name="val" />

Dla checkpointu `-seg` zwykły klucz `metrics/mAP50-95` zawiera wynik masek.
To samo uruchomienie raportuje też ramki pod oznaczeniem `(B)` i maski pod
oznaczeniem `(M)`, więc oba wyniki są dostępne po jednym przebiegu.

## Eksport

<export-matrix />

Wyeksportowany artefakt jest ponownie wczytywany przez `LibreYOLO()` na
podstawie sufiksu pliku, dlatego plik `.onnx` lub `.engine` zachowuje się jak
checkpoint i zwraca ten sam obiekt `Results`. Ścieżki OpenVINO, Paddle, MNN
i Core AI eksportują ze stałym obszarem zamiast kształtów dynamicznych. Strona
[Eksport](/docs/export) zawiera argumenty obsługiwane przez każdy format oraz
dodatki wymagane przez niektóre z nich.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box>

Wagi segmentacji mają drugie źródło upstream. Dekoder masek, dopasowywanie masek
i funkcja straty masek pochodzą z ArgoHA/D-FINE-seg, również na licencji
Apache-2.0. Opiekun tego projektu zatwierdził ponowne użycie z podaniem źródła.

</provenance-box>

## Cytowanie

<citation-block />
