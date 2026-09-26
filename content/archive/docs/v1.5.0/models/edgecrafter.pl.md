---
title: EdgeCrafter
families:
  - ec
seo_title: 'EdgeCrafter: wykrywać, pozycjonować i segmentować w LibreYOLO'
description: >-
  Użyj EdgeCrafter w LibreYOLO do wykrywania, określania pozy i segmentacji
  instancji. Instaluj, przewiduj, waliduj i eksportuj za pomocą kodu
  MIT-licensed.
lead: >-
  Kompaktowy transformer wizji do gęstego przewidywania na sprzęcie edge,
  opublikowany jako trzy powiązane modele: ECDet, ECPose i ECSeg. LibreYOLO
  ładuje wszystkie trzy jako jedną rodzinę, przy czym zadanie jest realizowane
  przez checkpoint.
keywords:
  - EdgeCrafter
  - ECDet
  - ECPose
  - ECSeg
  - kompaktowy transformator wizji
  - detekcja obiektów
  - estymacja pozy
  - segmentacja instancji
  - inferencja brzegowe
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreECs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Pozować
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Sufiks -pose w nazwie pliku wybiera głowicę punktów kluczowych, więc
        nie

        # tutaj potrzebny jest argument zadania.

        model = LibreYOLO("LibreECs-pose.pt")

        result = model(SAMPLE_IMAGE, save=True)


        print(result.keypoints.xy)

        print(result.boxes.conf)
    - label: Segmentacja instancji
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
            batch=8,
            lr0=5e-4,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreECs.pt data=my-dataset.yaml epochs=50
        imgsz=640 batch=8 lr0=5e-4
    - label: Pozować
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Wymaga jednoczesnego zestawu punktów kluczowych jednej klasy, którego
        plik data.yaml deklaruje

        # kpt_shape i imgsz w natywnej wielkości checkpointu.

        model = LibreYOLO("LibreECs-pose.pt")

        model.train(
            data="my-pose-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: Segmentacja instancji
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Wymaga etykiet wielokątów i rozmiaru obrazu (imgsz) w natywnym
        rozmiarze checkpointu.

        model = LibreYOLO("LibreECs-seg.pt")

        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            lora=True,
        )
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs.pt data=my-dataset.yaml
    - label: Pozować
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        metrics = model.val(data="my-pose-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: Segmentacja instancji
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # maski
        print(metrics["metrics/mAP50-95(B)"])   # pudełka
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-seg.pt format=onnx imgsz=640
    - label: Użyj wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka kieruje na podstawie rozszerzenia pliku, więc eksportowany
        artefakt się ładuje

        # jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreECs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 39c6975fc16b3ff1
---
## Instalacja

EdgeCrafter nie potrzebuje żadnych dodatkowych opcji. Wszystko, co importuje, znajduje się w instalacji bazowej.

```bash
pip install libreyolo
```

Dostrajanie adaptera z `lora=True` jest wyjątkiem i wymaga dodatkowego `lora`.

```bash
pip install "libreyolo[lora]"
```

## Predykcja

Wagi są pobierane z Hugging Face przy pierwszym użyciu i są przechowywane lokalnie w pamięci podręcznej.

<code-tabs name="predict" />

Zadanie pochodzi z nazwy pliku, więc checkpoint `-pose` lub `-seg` wybiera swoją własną głowicę i nie przyjmuje argumentu zadania. Wszystkie trzy zwracają obiekt `Results`, który zwraca każda rodzina, z `result.keypoints` dodanym dla pozy i `result.masks` dla segmentacji. Poza obejmuje jedną klasę, osobę, z 17 punktami kluczowymi COCO, a ich liczba jest ustalona podczas tworzenia modelu. Nie ma głowicy do obwiedni, więc każda rama pozy zawiera obwiednię własnych punktów kluczowych, a trzeci kanał punktów kluczowych jest stały, zamiast być oceną dla każdego punktu.

Argumenty `conf` i `max_det` filtrują wybór zapytań. Argument `iou` jest
akceptowany dla zgodności API, ale nie ma wpływu na wynik, ponieważ wszystkie
trzy głowice dekodują zbiór zapytań bez etapu NMS. Zobacz stronę
[predykcji](/docs/predict), aby poznać źródła, streaming i obsługę wyników.

## Warianty

Cztery rozmiary. Wszystkie działają przy tej samej rozdzielczości wejściowej, więc tabela rozdziela je według liczby parametrów i dokładności.

<benchmark-table task="detect" />

<va-embed />

Projekt upstream publikuje ECDet, ECPose i ECSeg jako trzy osobne modele, a nie
jeden model z trzema głowicami. Współdzielą ECViT, backbone i hybrydowy enkoder,
a różni je tylko głowica. LibreYOLO łączy je więc w jedną rodzinę, w której
nazwa pliku checkpointu wskazuje zadanie. Litera rozmiaru oznacza ten sam
backbone i enkoder we wszystkich trzech modelach, a metody predykcji, walidacji
i eksportu przyjmują te same argumenty niezależnie od wczytanego wariantu.

## Trenowanie

Wszystkie trzy zadania trenują przez `train()`, który odczytuje zadanie z załadowanego checkpointu i wybiera odpowiedniego trenera.

<code-tabs name="train" />

Dla detekcji i segmentacji sprawdzono zgodność inferencji z projektem upstream
z tolerancją 1e-5, warstwa po warstwie i dla każdego rozmiaru, a także działanie
funkcji straty i pojedynczego kroku trenowania na danych syntetycznych. Zgodnie
z opisem metody `train()` nie sprawdzono zbieżności pełnego dostrajania,
trenowania na wielu GPU, ponownego wczytania najlepszego modelu po wyłączeniu
augmentacji ani mapowania klas Objects365 na COCO. Ścieżka estymacji pozy
korzysta z opublikowanego przepisu DETRPose: dopasowania węgierskiego względem
klas, kosztów L1 i OKS dla punktów kluczowych oraz kontrastowego odszumiania
punktów kluczowych. Jej zbieżności również nie zweryfikowano od początku do
końca.

Pozostawiony sam, trener wykonuje 74 epoki na `lr0=5e-4` z włączoną mieszanką precyzji, zgodnie z recepturą upstream: AdamW, płaski harmonogram kosinusowy, EMA na 0,9999 i wejścia ImageNet-normalized. Poza i segmentacja wymagają `imgsz` w natywnej wielkości checkpointu, ponieważ ich siatka odniesienia do ewaluacji jest tworzona, gdy model jest konstruowany; inna wartość zwiększa się przed rozpoczęciem uruchamiania. Poza wymaga również jednoczęściowego zbioru danych, którego `data.yaml` deklaruje `kpt_shape`, z liczbą punktów kluczowych odpowiadającą głowie.

`lora=True` dotyczy wyłącznie detekcji; pozowanie i segmentacja powodują wywołanie `ValueError` na nim. Na Apple Silicon trener utrzymuje działanie na GPU i wysyła jedną operację do CPU, wsteczny grid-sample wewnątrz deformowalnej uwagi, której PyTorch nie implementuje w Metal.

Zobacz [trenowanie](/docs/train) dotyczące zbiorów danych, augmentacji, multi-GPU i loggerów.

## Walidacja

`val()` zwraca słownik kluczowany nazwą metryki i drukuje wyniki dla każdej klasy, gdy `verbose` jest włączony.

<code-tabs name="val" />

Raporty pozycji kluczowych Pose zawierają metryki OKS pod `metrics/keypoints_*`. Raporty segmentacji zawierają maski pod zwykłym kluczem `metrics/mAP50-95` i powtarzają oba widoki w jednym przebiegu, pudełka pod `(B)` i maski pod `(M)`.

## Eksport

<export-matrix />

Eksportowany artefakt ładuje się z powrotem przez `LibreYOLO()` na podstawie jego sufiksu pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint i zwraca ten sam `Results`. Eksport pozycji i segmentacji odbywa się przy stałym wejściu 640 na 640 zamiast dynamicznych kształtów, a kilka celów detekcji jest również w stałym układzie, w tym OpenVINO, Paddle, MNN, ExecuTorch i Core AI. [Eksport](/docs/export) wymienia argumenty, które akceptuje każdy format, oraz dodatkowe, które dodaje kilka z nich.

<code-tabs name="export" />

## Checkpointy

Każdy opublikowany plik wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
