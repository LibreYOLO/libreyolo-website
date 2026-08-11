---
title: DINOv2
families:
  - dinov2
seo_title: 'DINOv2 w LibreYOLO: semantyczny, klasyfikuj i osadź'
description: >-
  Użyj DINOv2 w LibreYOLO do segmentacji semantycznej, klasyfikacji i osadzania
  całego obrazu na DINOv2-with-Registers backbone. Apache-2.0 na całej długości.
lead: >-
  DINOv2 jest samonadzorowanym transformatorem wizji trenowanym przez Meta AI w
  celu generowania ogólnych cech obrazu bez etykiet. LibreYOLO otacza swoje
  DINOv2-with-Registers backbone do trzech zadań: segmentacji semantycznej,
  klasyfikacji i osadzania całego obrazu.
keywords:
  - DINOv2
  - DINOv2 z rejestrami
  - uczenie samonadzorowane
  - transformator wizji
  - segmentacja semantyczna
  - osadzanie obrazu
  - ekstrakcja cech
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Semantyczny
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.dinov2.model import LibreDINOv2


        # Nie istnieje checkpoint LibreYOLO-hosted dla tej rodziny: this

        # pobiera Apache-2.0 DINOv2-with-Registers-small backbone z

        # Organizacja Hugging Face Meta. Gęsta głowa zaczyna się losowo

        # inicjalizacja aż do momentu, gdy ją wytrenujesz (zobacz Trenuj
        poniżej).

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)

        result = model(SAMPLE_IMAGE)


        mask = result.semantic_mask

        print(mask.data.shape, mask.classes)
    - label: Klasyfikować
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.dinov2.model import LibreDINOv2


        # nb_classes= to liczba klas w twoim zbiorze danych; głowa liniowa
        zaczyna

        # losowo inicjalizowane aż do momentu, gdy je wytrenujesz.

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1, result.probs.top1conf)
    - label: Osadzić
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # Omija każdą głowę zadania: sam backbone wystarczy, więc to
        # nie wymaga dopracowywania, aby być użytecznym.
        model = LibreDINOv2(size="s", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (1, D), L2-normalized
    - label: Osadź partia
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Wygodna nakładka: uruchamia predict() i układa każdy wiersz w jeden
        # (N, D) tensor.
        features = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(features.shape)
  train:
    - label: Semantyczny
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Klasyfikować
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Multi-GPU
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(
            data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4,
            device="0,1",
        )
  val:
    - label: Semantyczny
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Klasyfikować
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
  export:
    - label: Semantyczny
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.export(format="onnx")
    - label: Klasyfikować
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.export(format="onnx")
    - label: Osadzić
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        model.export(format="tflite")
    - label: Użyj wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka kieruje na podstawie rozszerzenia pliku, więc eksportowany
        artefakt się ładuje

        # jak każdy checkpoint i zwraca ten sam obiekt Results. Eksport

        # nazywa plik z zadania, tutaj LibreDINOv2s-sem.onnx.

        model = LibreYOLO("LibreDINOv2s-sem.onnx")

        result = model(SAMPLE_IMAGE)
source_hash: 4256e0a0398e5aaf
---
## Instalacja

LibreDINOv2 jest rejestrowany tylko wtedy, gdy zainstalowano pakiet
`transformers`. Jest to ta sama opcjonalna zależność, której RF-DETR wymaga dla
backbone DINOv2, dlatego potrzebny jest ten sam dodatek.

```bash
pip install "libreyolo[rfdetr]"
```

## Predykcja

LibreYOLO nie publikuje checkpointu LibreDINOv2. Zamiast wczytywać plik, należy
utworzyć otokę bezpośrednio. Domyślne `model_path=None` pobiera przy pierwszym
użyciu z Hugging Face backbone Meta `facebook/dinov2-with-registers-small` na
licencji Apache-2.0. Argument `task=` wybiera wykonywane na nim zadanie.

<code-tabs name="predict" />

`task="semantic"` i `task="classify"` dodają do backbone odpowiednio gęstą lub
liniową głowicę. Głowica jest inicjalizowana losowo i staje się użyteczna dopiero
po wytrenowaniu (zobacz sekcję [Trenowanie](#train)). `task="embed"` pomija
wszystkie głowice i zwraca końcowy, znormalizowany token CLS backbone jako jeden
wiersz reprezentujący cały obraz w `result.embeddings`, dlatego nie wymaga
trenowania. `result.boxes` zawsze ma wartość `None`, ponieważ żadne z trzech
zadań nie tworzy detekcji instancji. Zobacz stronę
[predykcji](/docs/predict), aby poznać źródła, streaming i obsługę wyników.

## Warianty

`size` wybiera szerokość projektora RF-DETR-style nałożoną na backbone, a nie sam backbone: każdy rozmiar korzysta z tego samego kodera DINOv2-S (małego). Segmentacja semantyczna działa na natywnej siatce kwadratowych patchy DINOv2; klasyfikacja i osadzanie działają w mniejszej rozdzielczości klasyfikacyjnej używanej do trenowania liniowej sondy.

## Trenowanie

Zadania `task="semantic"` i `task="classify"` obsługują trenowanie. Zadanie
`task="embed"` nie ma zależnej od klas głowicy do dopasowania, dlatego wywołanie
dla niego `train()` zgłasza `NotImplementedError`.

<code-tabs name="train" />

Główne argumenty kluczowe tutaj to `batch_size` i `lr`, a nie `batch` i `lr0` używane przez większość innych rodzin; `batch` i `lr0` są nadal akceptowane i mapowane na nie, ale podanie obu powoduje błąd konfliktu. `output_dir=` (domyślnie `"runs/train"`) zastępuje `project=`/`name=` jako główny sposób umieszczania biegu, chociaż bezpośrednie podanie `project=`/`name=` nadal działa. Zobacz [trenowanie](/docs/train) dotyczące zbiorów danych, augmentacji, wielu GPU i loggerów.

## Walidacja

`val()` zwraca słownik kluczy `metrics/`: mIoU i dokładność pikseli dla `task="semantic"`, dokładność top-1 i top-5 dla `task="classify"`. `task="embed"` nie ma prawdziwej wartości do oceny i zgłasza `NotImplementedError`, jeśli wywołasz `val()` na nim.

<code-tabs name="val" />

## Eksport

<export-matrix />

Każde zadanie obsługuje inny podzbiór formatów, pokazany powyżej. Wyeksportowany artefakt można ponownie załadować przez `LibreYOLO()` na jego rozszerzeniu pliku, więc plik `.onnx` lub `.engine` działa jak checkpoint i zwraca ten sam `Results`. [Eksport](/docs/export) pokazuje argumenty, które akceptuje każdy format.

<code-tabs name="export" />

## Licencjonowanie

<provenance-box>

Wiersz „Wagi” powyżej podaje obowiązującą licencję Apache-2.0, ale LibreYOLO nie
publikuje ponownie żadnych artefaktów tej rodziny na Hugging Face i nie ma
własnego checkpointu LibreDINOv2. Wywołanie
`LibreDINOv2(model_path=None)` pobiera niezmienione repozytorium Meta
`facebook/dinov2-with-registers-small`.

</provenance-box>

## Cytowanie

<citation-block /> 
