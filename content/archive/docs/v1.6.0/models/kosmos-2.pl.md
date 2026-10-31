---
title: Kosmos-2
families:
  - kosmos2
seo_title: 'Kosmos-2 w LibreYOLO: detekcja obiektów z groundingiem'
description: >-
  Kosmos-2 w LibreYOLO: zainstaluj model Microsoft na licencji MIT, ustaw
  otwarty słownik i przewiduj ramki z groundingiem.
lead: >-
  Kosmos-2 to model groundingu Microsoft: tworzy podpis obrazu, a następnie
  lokalizuje ramką każdą frazę rzeczownikową w tym podpisie. LibreYOLO
  udostępnia go jako detektor obiektów z otwartym słownikiem: listę klas podaje
  się podczas predykcji.
keywords:
  - Kosmos-2
  - model wizyjno-językowy
  - grounding obrazu
  - detekcja z otwartym słownikiem
  - Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Wideo
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("kosmos-2")

        model.set_classes(["boat", "person"])


        # Dowolne źródło obsługiwane przez bibliotekę: plik, folder, adres URL,
        indeks kamery,

        # strumień RTSP lub lista .streams

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: 60e0796f34be6d59
---

## Instalacja

Kosmos-2 należy do poziomu VLM jako detektor w LibreYOLO, osobnej powierzchni
produktu od rodzin opartych na checkpointach, z własną fabryką. Wymaga dodatku
`vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej. LibreYOLO bezpośrednio wczytuje własne repozytorium Microsoft
`microsoft/kosmos-2-patch14-224`. W przeciwieństwie do Florence-2 nie jest tu
potrzebne ponowne opublikowanie przez społeczność.

<code-tabs name="predict" />

Ta rodzina jest wczytywana przez fabrykę `LibreVLM()`, a nie `LibreYOLO()`.
Rodziny VLM nie deklarują loadera checkpointów, dlatego opisany na innych
stronach modeli wybór ścieżki na podstawie rozszerzenia pliku nie ma tu
zastosowania. `set_classes()` ustawia słownik, o którego znalezienie proszony
jest Kosmos-2. Ustawienie jest trwałe i obowiązuje we wszystkich późniejszych
wywołaniach `predict()`/`track()`, dopóki nie zostanie ponownie zmienione.
Kosmos-2 wykonuje grounding fraz rzeczownikowych zamiast dokładnego dopasowania
etykiety, dlatego wrapper LibreYOLO przyjmuje częściowe dopasowanie. Klasa o
nazwie `"boat"` pasuje także do wygenerowanej frazy takiej jak „the boats”.
Każda detekcja ma ten sam zastępczy wskaźnik pewności, więc filtrowanie przez
`conf` działa na zasadzie wszystko albo nic, a `iou` nie ma tu wpływu na wynik,
ponieważ wrapper tworzy listę detekcji bezpośrednio z elementów z groundingiem,
bez etapu usuwania duplikatów. `chat()` zgłasza `NotImplementedError`, ponieważ
Kosmos-2 jest sterowany promptem `<grounding>`, a nie szablonem czatu. CLI
LibreYOLO nie obsługuje tego poziomu. Nie istnieje dla niego forma
`libreyolo predict model=...`. Więcej informacji o źródłach, streamingu i
obsłudze wyników zawiera strona [predykcji](/docs/predict).

## Warianty

Dostępny jest jeden rozmiar: `kosmos-2-patch14-224` przy 224 px, wczytywany jako
`LibreVLM("kosmos-2")`. Jest to model z 2023 roku, a własny wrapper LibreYOLO
zaznacza, że jego grounding jest mniej dokładny niż w nowszych detektorach tego
poziomu.

LibreYOLO nie trenuje, nie waliduje ani nie eksportuje Kosmos-2. Wywołania
`train()`, `val()` i `export()` zgłaszają `NotImplementedError` dla każdej
rodziny na tym poziomie (zobacz poziom obsługi powyżej). Jeśli potrzebny jest
niestandardowy słownik zapisany w wagach, należy dostroić Kosmos-2 w projekcie
źródłowym i wczytać powstałe wagi. Wynik `predict()` trzeba sprawdzić wzrokowo
zamiast uruchamiać walidację w stylu COCO, ponieważ każda detekcja ma ten sam
zastępczy wskaźnik pewności.

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
