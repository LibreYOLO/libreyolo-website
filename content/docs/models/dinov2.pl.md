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

LibreDINOv2 rejestruje się tylko wtedy, gdy zainstalowany jest `transformers`, ta sama opcjonalna zależność, której RF-DETR potrzebuje dla swojego DINOv2 backbone, więc potrzebuje tego samego dodatkowego elementu.

```bash
pip install "libreyolo[rfdetr]"
```

## Predykcja

LibreYOLO nie publikuje checkpointu LibreDINOv2. Zamiast ładować plik, skonstruuj bezpośrednio opakowanie: `model_path=None` (domyślnie) pobiera Apache-2.0 `facebook/dinov2-with-registers-small` backbone Meta z Hugging Face przy pierwszym użyciu. `task=` wybiera, co na nim działa.

<code-tabs name="predict" />

`task="semantic"` i `task="classify"` dodają gęstą lub liniową głowicę na szczycie backbone; ta głowica jest inicjalizowana losowo i przydatna dopiero po jej wytrenowaniu (zobacz [Train](#train)). `task="embed"` pomija każdą głowicę i zwraca ostateczny znormalizowany token CLS backbone jako jeden wiersz całego obrazu w `result.embeddings`, więc wcale nie wymaga treningu. `result.boxes` to zawsze `None`: żadna z trzech zadań nie produkuje detekcji dla poszczególnych przypadków. Zobacz [predykcja](/docs/predict) dla źródeł, streaming i obsługi wyników.

## Warianty

`size` wybiera szerokość projektora RF-DETR-style nałożoną na backbone, a nie sam backbone: każdy rozmiar korzysta z tego samego kodera DINOv2-S (małego). Segmentacja semantyczna działa na natywnej siatce kwadratowych patchy DINOv2; klasyfikacja i osadzanie działają w mniejszej rozdzielczości klasyfikacyjnej używanej do trenowania liniowej sondy.

## Trenowanie

`task="semantic"` i `task="classify"` oba trenują; `task="embed"` nie ma głowy zależnej od klasy do dopasowania i wywołuje `NotImplementedError`, jeśli wywołasz na nim `train()`.

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

Wiersz "Weights" powyżej podaje licencję, która obowiązuje, Apache-2.0, ale w rzeczywistości nic nie jest ponownie publikowane pod LibreYOLO Hugging Face dla tej rodziny: LibreYOLO nie posiada własnego LibreDINOv2 checkpointu. To, co `LibreDINOv2(model_path=None)` pobiera, to własne repozytorium `facebook/dinov2-with-registers-small` firmy Meta, nienaruszone.

</provenance-box>

## Cytowanie

<citation-block /> 
