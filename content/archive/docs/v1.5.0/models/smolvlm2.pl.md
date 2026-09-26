---
title: SmolVLM2
families:
  - smolvlm2
seo_title: 'SmolVLM2 w LibreYOLO: detekcja z otwartym słownikiem'
description: >-
  SmolVLM2 w LibreYOLO: zainstaluj model wizyjno-językowy Hugging Face na
  licencji Apache-2.0, ustaw otwarty słownik, uruchamiaj predykcję lub czat.
lead: >-
  SmolVLM2 to mały model wizyjno-językowy Hugging Face. LibreYOLO udostępnia go
  jako detektor obiektów z otwartym słownikiem oraz zapewnia bezpośredni dostęp
  do swobodnego czatu: podaj listę klas do detekcji albo zadaj pytanie.
keywords:
  - SmolVLM2
  - model wizyjno-językowy
  - detekcja z otwartym słownikiem
  - mały model multimodalny
  - Hugging Face
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")
        model.set_classes(["cat", "dog"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Czat
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")

        # Bezpośredni dostęp pod wygodną warstwą detekcji: dowolne pytanie,
        # nie tylko zapytanie o ramki ograniczające.
        answer = model.chat(SAMPLE_IMAGE, "What is the cat doing?")
        print(answer)
source_hash: b30823b62d6347b5
---

## Instalacja

SmolVLM2 należy do poziomu VLM jako detektor w LibreYOLO, osobnej powierzchni
produktu od rodzin opartych na checkpointach, z własną fabryką. Wymaga dodatku
`vlm`, który instaluje również `num2words`, zależność własnego procesora
SmolVLM2.

```bash
pip install "libreyolo[vlm]"
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Ta rodzina jest wczytywana przez fabrykę `LibreVLM()`, a nie `LibreYOLO()`.
Rodziny VLM nie deklarują loadera checkpointów, dlatego opisany na innych
stronach modeli wybór ścieżki na podstawie rozszerzenia pliku nie ma tu
zastosowania. `set_classes()` ustawia słownik, o którego znalezienie proszony
jest SmolVLM2. Ustawienie jest trwałe i obowiązuje we wszystkich późniejszych
wywołaniach `predict()`/`track()`, dopóki nie zostanie ponownie zmienione.
SmolVLM2 nie wymaga zastąpienia parsera w LibreYOLO. Korzysta z tego samego
szablonu czatu z wyjściem JSON co wspólna wartość domyślna tego poziomu, więc
jego prompt detekcji i format ramek nie zależą od rodziny. Każda detekcja ma ten
sam zastępczy wskaźnik pewności, więc filtrowanie przez `conf` działa na zasadzie
wszystko albo nic. `iou` ma natomiast wpływ na wynik: późniejsza ramka tej samej
klasy jest odrzucana, gdy nakłada się na zachowaną ramkę powyżej progu, ponieważ
generator powtarzający może zwrócić prawie identyczne ramki jednego obiektu.
SmolVLM2 odpowiada także na swobodne pytania przez `chat()`, ten sam bezpośredni
dostęp opisany dla fabryki `LibreVLM`. CLI LibreYOLO nie obsługuje tego poziomu.
Nie istnieje dla niego forma `libreyolo predict model=...`. Więcej informacji o
źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

Rejestr zawiera jeden rozmiar: SmolVLM2-500M-Video-Instruct, wczytywany jako
`LibreVLM("smolvlm2-500m")`. SmolVLM2 jest słabszym detektorem niż specjalnie
zbudowane modele groundingu z tego poziomu. Własny wrapper LibreYOLO opisuje go
jako demonstrację, że nowa rodzina nie wymaga tutaj specjalnego parsera, a nie
jako najlepszą opcję detekcji z otwartym słownikiem.

LibreYOLO nie trenuje, nie waliduje ani nie eksportuje SmolVLM2. Wywołania
`train()`, `val()` i `export()` zgłaszają `NotImplementedError` dla każdej
rodziny na tym poziomie (zobacz poziom obsługi powyżej). Jeśli potrzebny jest
niestandardowy słownik zapisany w wagach, należy dostroić SmolVLM2 w projekcie
źródłowym i wczytać powstałe wagi. Wynik `predict()` trzeba sprawdzić wzrokowo
zamiast uruchamiać walidację w stylu COCO, ponieważ każda detekcja ma ten sam
zastępczy wskaźnik pewności.

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
