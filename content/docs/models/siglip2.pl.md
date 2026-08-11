---
title: SigLIP2
families:
  - siglip2
seo_title: 'SigLIP2 w LibreYOLO: klasyfikacja zero-shot i embeddingi'
description: >-
  Używaj SigLIP2 w LibreYOLO do klasyfikacji obrazów zero-shot oraz tworzenia
  embeddingów obrazu i tekstu z wieloetykietową oceną sigmoid. Trenowanie nie
  jest potrzebne.
lead: >-
  SigLIP2 to model dwuwieżowy, który ocenia obraz względem promptów tekstowych
  za pomocą niezależnej funkcji sigmoid dla każdej klasy zamiast wspólnego
  softmax dla stałego zbioru etykiet. LibreYOLO obsługuje go do klasyfikacji
  zero-shot oraz embeddingów obrazu i tekstu, bez etapu trenowania.
keywords:
  - SigLIP2
  - SigLIP 2
  - klasyfikacja zero-shot
  - embedding obrazu
  - embedding tekstu
  - otwarty słownik
  - model wielojęzyczny
  - sigmoid loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Bez wywołania set_classes() predykcja CLI używa 1000 nazw klas

        # ImageNet, które model wczytuje domyślnie.

        libreyolo predict model=LibreSigLIP2b16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Wieloetykietowa ocena sigmoid
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        model.set_classes(["a dog", "a cat", "outdoors"], multi_label=True)

        r = model(SAMPLE_IMAGE)


        # Niezależne prawdopodobieństwa każdej klasy: więcej niż jedna klasa
        albo żadna

        # może jednocześnie uzyskać wysoki wynik. Softmax (domyślny) normalizuje
        je

        # do rozkładu jednoetykietowego, zgodnie z zachowaniem LibreCLIP.

        for i, name in model.names.items():
            print(name, float(r.probs.data[i]))
    - label: Embedding obrazu i tekstu
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")

        image_embed = model(SAMPLE_IMAGE).embeddings.data

        text_embed = model.embed_text("a photo of a forklift")


        # Oba są znormalizowane metodą L2, więc zwykły iloczyn skalarny to
        podobieństwo cosinusowe.

        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSigLIP2b16-cls.pt")


        # data to katalog główny ImageFolder z podziałem train/. Nazwy jego
        folderów

        # stają się promptami klas zero-shot w tym przebiegu.

        metrics = model.val(data="imagenette160")


        print(metrics["metrics/accuracy_top1"])

        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSigLIP2b16-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        model.set_classes(["a forklift", "an empty aisle", "a spill"])

        model.export(format="onnx")


        # Bieżące etykiety set_classes() i rozdzielczość wejściowa są wbudowane

        # w graf. Po zmianie któregokolwiek elementu wykonaj ponowny eksport.
        multi_label

        # musi mieć wartość False (domyślną) podczas eksportu.
    - label: CLI
      language: bash
      code: |
        # Nie wywołano set_classes(), dlatego w grafie zostanie zapisanych 1000
        # domyślnych klas ImageNet wczytywanych przez model.
        libreyolo export model=LibreSigLIP2b16-cls.pt format=onnx
    - label: Eksport embeddingów
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" śledzi tylko wieżę obrazu. Klasy nie są potrzebne.
        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: f992655747fd8819
---

## Instalacja

SigLIP2 wymaga własnego dodatku, który instaluje pakiet SentencePiece używany
przez jego wielojęzyczny tokenizer.

```bash
pip install "libreyolo[siglip2]"
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

`set_classes()` jest mechanizmem tworzącym z tego modelu klasyfikator z otwartym
słownikiem. Umieszcza każdą etykietę we wszystkich szablonach promptów, koduje i
uśrednia wyniki oraz zapisuje powstałą macierz `[K, D]` w pamięci podręcznej jako
głowicę klasyfikatora, dzięki czemu nie jest ona ponownie obliczana dla każdego
obrazu. Ponowne wywołanie pozwala w każdej chwili zmienić klasy. Bez wywołania
LibreSigLIP2 ma już ustawione 1000 nazw klas ImageNet-1k.

SigLIP ocenia każdą klasę niezależnie: `logit = scale * (image . text) + bias`.
Domyślnie ten zbiór logitów nadal przechodzi przez softmax, co daje rozkład
jednoetykietowy zgodny z zachowaniem `top1`/`top5` w LibreCLIP. Przekazanie
`multi_label=True` do `set_classes()` (lub podczas tworzenia) przełącza na
niezależne prawdopodobieństwa sigmoid. Na tym samym obrazie wysoki wynik może
uzyskać więcej niż jedna klasa albo żadna. Tokenizer jest wielojęzycznym modelem
SentencePiece ze słownikiem Gemma, dlatego nazwy klas w językach innych niż
angielski działają tak samo.

Przy `task="embed"` predykcja zwraca jeden znormalizowany metodą L2 wektor obrazu
na wejście zamiast prawdopodobieństw klas, a `embed_text()` zwraca
znormalizowane wiersze tekstu w tej samej przestrzeni wektorowej. Zwykły iloczyn
skalarny jest więc ich podobieństwem cosinusowym. `iou` nie ma wpływu na żadne z
zadań, ponieważ nie ma etapu NMS. Więcej informacji o źródłach, streamingu i
obsłudze wyników zawiera strona [predykcji](/docs/predict).

## Walidacja

`val()` odczytuje nazwy folderów klas z podziału `train/` ImageFolder, wywołuje
dla nich `set_classes()`, a następnie mierzy dokładność zero-shot top-1 i top-5
z oceną softmax. Dokładność zależy od brzmienia nazw klas jako promptów, a nie
od aktualizacji wag, ponieważ nie ma niczego do trenowania. Walidacja obejmuje
wyłącznie `task="classify"`. `task="embed"` nie ma walidatora zbioru danych.

<code-tabs name="val" />

## Eksport

<export-matrix />

Eksport zapisuje bieżący stan modelu w stałym grafie. Dla `task="classify"`
etykiety ustawione ostatnio przez `set_classes()` i rozdzielczość w chwili
eksportu są wbudowane w końcową warstwę liniową z wyuczoną skalą i obciążeniem.
Wyeksportowany graf jest zwykłym klasyfikatorem obrazów `[B, K]` bez wieży
tekstowej ani tokenizera. Po zmianie klas lub rozmiaru trzeba wyeksportować go
ponownie. Eksport w trybie `multi_label=True` nie jest zaimplementowany. Najpierw
należy przywrócić `False`. Eksport `task="embed"` śledzi wyłącznie wieżę obrazu.
Oba wymagają opset ONNX 14 lub nowszego, który eksporter ustawia domyślnie.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny. Oba przekonwertowano z
checkpointów Google na licencji Apache-2.0 `siglip2-base-patch16-256` i
`siglip2-so400m-patch14-384`, a nie z przebiegu trenowania COCO.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
