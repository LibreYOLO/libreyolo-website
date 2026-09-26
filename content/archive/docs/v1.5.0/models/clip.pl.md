---
title: CLIP
families:
  - clip
seo_title: 'CLIP w LibreYOLO: klasyfikacja zero-shot i embeddingi'
description: >-
  Używaj CLIP w LibreYOLO do klasyfikacji obrazów zero-shot oraz tworzenia
  embeddingów obrazu i tekstu. Bez trenowania: set_classes() określa zestaw
  etykiet w czasie działania.
lead: >-
  CLIP to model z dwiema wieżami, który ocenia obraz względem promptów
  tekstowych zamiast stałego zestawu etykiet. LibreYOLO obsługuje go w
  klasyfikacji zero-shot oraz do tworzenia embeddingów obrazu i tekstu, bez
  etapu trenowania.
keywords:
  - CLIP
  - OpenCLIP
  - klasyfikacja zero-shot
  - embedding obrazu
  - embedding tekstu
  - otwarty słownik
  - LAION-2B
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Bez wywołania set_classes() predykcja CLI używa 1000 nazw klas

        # ImageNet, które model wczytuje domyślnie.

        libreyolo predict model=LibreCLIPb32-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Embedding obrazu i tekstu
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image_embed = model(SAMPLE_IMAGE).embeddings.data

        text_embed = model.embed_text("a photo of a forklift")


        # Oba są znormalizowane normą L2, więc zwykły iloczyn skalarny jest
        podobieństwem cosinusowym.

        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        # data to katalog główny ImageFolder z podziałem train/; nazwy jego
        # folderów stają się promptami klas zero-shot dla tego uruchomienia.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCLIPb32-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreCLIPb32-cls.pt")

        model.set_classes(["a forklift", "an empty aisle", "a spill"])

        model.export(format="onnx")


        # Bieżące etykiety set_classes() i rozdzielczość wejścia zostają
        zapisane

        # w grafie. Po zmianie którejkolwiek z nich należy ponowić eksport.
    - label: CLI
      language: bash
      code: |
        # Tutaj nie ma wywołania set_classes(), więc zapisywanych jest 1000
        # domyślnych klas ImageNet wczytanych przez model.
        libreyolo export model=LibreCLIPb32-cls.pt format=onnx
    - label: Eksport embeddingów
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" śledzi tylko wieżę obrazu; klasy nie są potrzebne.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: ac7cfd75ad6c0fa7
---

## Instalacja

CLIP wymaga własnego dodatku, który instaluje pakiety używane przez dołączony
tokenizator BPE do dokładnego odtworzenia identyfikatorów tokenów.

```bash
pip install "libreyolo[clip]"
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane lokalnie
w pamięci podręcznej.

<code-tabs name="predict" />

`set_classes()` to podstawowa operacja, dzięki której jest to klasyfikator
z otwartym słownikiem. Renderuje każdą etykietę we wszystkich szablonach
promptów, koduje i uśrednia wyniki, a następnie zapisuje powstałą macierz
`[K, D]` w pamięci podręcznej jako głowicę klasyfikatora, dzięki czemu nie jest
ona obliczana ponownie dla każdego obrazu. Metodę można wywołać ponownie, aby
w dowolnej chwili zmienić klasy. Bez tego wywołania LibreCLIP ma już ustawione
1000 nazw klas ImageNet-1k.

Przy `task="embed"` predykcja zwraca po jednym znormalizowanym normą L2 wektorze
obrazu dla każdego wejścia zamiast prawdopodobieństw klas, a `embed_text()`
zwraca znormalizowane wiersze tekstu w tej samej przestrzeni wektorowej. Zwykły
iloczyn skalarny między nimi jest więc podobieństwem cosinusowym. Argument `iou`
nie ma wpływu na żadne z tych zadań, ponieważ nie występuje etap NMS. Zobacz
stronę [predykcji](/docs/predict), aby poznać źródła, streaming i obsługę
wyników.

## Walidacja

Metoda `val()` odczytuje nazwy folderów klas z podziału `train/` ImageFolder,
wywołuje z nimi `set_classes()`, a następnie mierzy accuracy top-1 i top-5
w trybie zero-shot. Accuracy zależy od tego, jak nazwy klas sprawdzają się jako
prompty, a nie od aktualizacji wag, ponieważ model nie jest trenowany. Walidacja
obejmuje wyłącznie `task="classify"`. Dla `task="embed"` nie ma walidatora
zbioru danych.

<code-tabs name="val" />

## Eksport

<export-matrix />

Eksport zapisuje bieżący stan modelu w stałym grafie. W przypadku
`task="classify"` etykiety ustawione ostatnio przez `set_classes()` oraz
rozdzielczość w chwili eksportu zostają zapisane w końcowej warstwie liniowej.
Wyeksportowany graf ONNX lub TensorRT jest więc zwykłym klasyfikatorem obrazów
`[B, K]`, bez wieży tekstu i bez tokenizatora. Po zmianie klas lub rozmiaru należy
wyeksportować model ponownie. Eksport `task="embed"` śledzi tylko wieżę obrazu.
Oba warianty wymagają ONNX opset 14 lub nowszego, który eksporter ustawia
domyślnie.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny. Oba przekonwertowano
z checkpointów OpenCLIP wytrenowanych na LAION-2B (`ViT-B-32` i `ViT-B-16`),
a nie w ramach trenowania na COCO.

<checkpoint-table />

W danych treningowych LAION-2B udokumentowano występowanie materiałów CSAM
(Stanford Internet Observatory, grudzień 2023). Od tego czasu LAION opublikował
Re-LAION, oczyszczone ponowne wydanie. Jeśli te wagi mają być dalej
udostępniane, w miarę dostępności należy preferować checkpointy utworzone
z Re-LAION.

## Licencjonowanie

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
