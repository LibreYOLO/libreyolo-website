---
title: Florence-2
families:
  - florence2
seo_title: 'Florence-2 w LibreYOLO: detekcja z otwartym słownikiem'
description: >-
  Florence-2 w LibreYOLO: zainstaluj model wizyjny Microsoft na licencji MIT,
  ustaw otwarty słownik i przewiduj ramki.
lead: >-
  Florence-2 to bazowy model wizyjny Microsoft, sterowany tokenem zadania
  zamiast stałą głowicą detekcji. LibreYOLO udostępnia go jako detektor obiektów
  z otwartym słownikiem: listę klas podaje się podczas predykcji.
keywords:
  - Florence-2
  - model wizyjno-językowy
  - detekcja z otwartym słownikiem
  - grounding obrazu
  - Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Wideo
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("florence-2-base")

        model.set_classes(["car", "person", "traffic light"])


        # Dowolne źródło obsługiwane przez bibliotekę: plik, folder, adres URL,
        indeks kamery,

        # strumień RTSP lub lista .streams

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: ad26d9056465d662
---

## Instalacja

Florence-2 należy do poziomu VLM jako detektor w LibreYOLO, osobnej powierzchni
produktu od rodzin opartych na checkpointach, z własną fabryką. Wymaga dodatku
`vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej. LibreYOLO pobiera ponownie opublikowany checkpoint
florence-community zamiast oryginalnego repozytorium `microsoft/Florence-2-*`.
Przyczynę wyjaśnia sekcja Licencja.

<code-tabs name="predict" />

Ta rodzina jest wczytywana przez fabrykę `LibreVLM()`, a nie `LibreYOLO()`.
Rodziny VLM nie deklarują loadera checkpointów, dlatego opisany na innych
stronach modeli wybór ścieżki na podstawie rozszerzenia pliku nie ma tu
zastosowania. `set_classes()` ustawia słownik, o którego znalezienie na obrazie
proszony jest Florence-2. Ustawienie jest trwałe i obowiązuje we wszystkich
późniejszych wywołaniach `predict()`/`track()`, dopóki nie zostanie ponownie
zmienione. Zwracany obiekt `Results` zawiera `boxes` w takim samym kształcie jak
w pozostałych rodzinach, ale każda detekcja ma ten sam zastępczy wskaźnik
pewności. Filtrowanie przez `conf` działa więc na zasadzie wszystko albo nic,
a `iou` nie ma wpływu na wynik. Wrapper Florence-2 tworzy listę detekcji
bezpośrednio z przetworzonego wyjścia tokenu zadania, bez etapu usuwania
duplikatów. `chat()` zgłasza tutaj `NotImplementedError`, ponieważ Florence-2
jest sterowany tokenem zadania `<OPEN_VOCABULARY_DETECTION>`, a nie szablonem
czatu. CLI LibreYOLO nie obsługuje tego poziomu. Nie istnieje dla niego forma
`libreyolo predict model=...`. Więcej informacji o źródłach, streamingu i
obsłudze wyników zawiera strona [predykcji](/docs/predict).

## Warianty

Dostępne są dwa rozmiary, Florence-2-base i Florence-2-large, oba przy 768 px,
wczytywane jako `LibreVLM("florence-2-base")` lub
`LibreVLM("florence-2-large")`. LibreYOLO nie opublikowało benchmarku
porównującego ich dokładność.

LibreYOLO nie trenuje, nie waliduje ani nie eksportuje Florence-2. Wywołania
`train()`, `val()` i `export()` zgłaszają `NotImplementedError` dla każdej
rodziny na tym poziomie (zobacz poziom obsługi powyżej). Jeśli potrzebny jest
niestandardowy słownik zapisany w wagach, należy dostroić Florence-2 w projekcie
źródłowym i wczytać powstałe wagi. Wynik `predict()` trzeba sprawdzić wzrokowo
zamiast uruchamiać walidację w stylu COCO, ponieważ każda detekcja ma ten sam
zastępczy wskaźnik pewności.

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
