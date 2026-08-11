---
title: LocateAnything
families:
  - locateanything
seo_title: 'LocateAnything: detekcja i wskazywanie z otwartym słownikiem'
description: >-
  Używaj LocateAnything w LibreYOLO do detekcji i wskazywania z otwartym
  słownikiem. Przewiduj z dowolną etykietą tekstową. Trenowanie, walidacja i
  eksport nie są obsługiwane.
lead: >-
  LocateAnything to model groundingu wizyjno-językowego wydany przez NVIDIA,
  który dekoduje ramki ograniczające i punkty równolegle zamiast po jednym
  tokenie współrzędnych. LibreYOLO udostępnia go jako detektor i wskaźnik z
  otwartym słownikiem: dowolna lista etykiet tekstowych staje się zbiorem klas,
  bez stałej głowicy i bez potrzeby dostrajania.
keywords:
  - LocateAnything
  - NVIDIA
  - model wizyjno-językowy
  - detekcja z otwartym słownikiem
  - detekcja punktów
  - VLM
  - grounding obrazu
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE


        model = LibreLocateAnything(size="3b")


        # Otwarty słownik: działają dowolne słowa, a nie stała głowica klas.
        Ustawienie

        # obowiązuje w każdym późniejszym wywołaniu predict()/track(), dopóki
        nie zostanie zmienione.

        model.set_classes(["person", "bicycle", "dog"])

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Sterowanie promptem punktowym
      language: python
      code: >
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE


        # task="point" zwraca jeden punkt dla każdego dopasowanego obiektu
        zamiast ramki.

        # Zadania we wczytanym modelu przełącza model.set_task("point").

        model = LibreLocateAnything(size="3b", task="point")

        model.set_classes(["the person closest to the camera"])

        result = model(SAMPLE_IMAGE, save=True)


        for pt in result.points:
            print(pt.cls, pt.conf, pt.xy)
    - label: Surowy czat
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # Bezpośredni dostęp pod wygodną warstwą detekcji: swobodne pytania,
        # zliczanie lub dowolny prompt nieobsługiwany przez wrapper ramek.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 378ea758e507a096
---

## Instalacja

LocateAnything wymaga dodatku `vlm`, który instaluje `transformers` oraz pakiety
`decord`, `lmdb` i `peft` importowane podczas wczytywania przez zdalny kod modelu
z Hugging Face.

```bash
pip install "libreyolo[vlm]"
```

## Predykcja

`LibreLocateAnything` jest klasą Pythona, a nie checkpointem `.pt`. Nie jest
wczytywany przez fabrykę `LibreYOLO()`, a CLI `libreyolo` go nie rozpoznaje.
Fabryka `LibreVLM(...)` (`from libreyolo import LibreVLM`) także udostępnia tę
rodzinę przez alias, na przykład `LibreVLM("locate-anything")`. Użyta poniżej
klasa jest tym, co tworzy ta fabryka. Wczytanie pobiera i wykonuje własny zdalny
kod modelu NVIDIA z Hugging Face, dlatego LibreYOLO przypina pobieranie do jednej
stałej rewizji commita zamiast zmiennej gałęzi `main` i jednorazowo wypisuje
informację o licencji przed pierwszym pobraniem.

<code-tabs name="predict" />

`result.boxes` (zadanie `detect`) i `result.points` (zadanie `point`) zawierają
przetworzone wyjście tak jak w każdej innej rodzinie. Pewność jest wartością
zastępczą. LocateAnything nie zwraca wyniku dla poszczególnych ramek, dlatego
każda detekcja otrzymuje tę samą stałą pewność, a `conf=` jedynie odrzuca
wiersze poniżej tej stałej, bez ich klasyfikowania. Pominięcie `set_classes()`
pozostawia domyślne nazwy COCO-80. Więcej informacji o źródłach, streamingu i
obsłudze wyników zawiera strona [predykcji](/docs/predict).

## Warianty

Opublikowano jeden rozmiar, 3b. Dwa zadania współdzielą te same wagi: `detect`
(domyślne) zwraca ramki, a `task="point"` zwraca zamiast nich jeden punkt na
dopasowany obiekt w `result.points`. We wczytanym modelu można je przełączać za
pomocą `model.set_task("point")`. Zestaw testowy LibreYOLO nie wykonał pomiarów
tej rodziny, dlatego nie ma opublikowanych wyników dokładności do porównania.

LibreYOLO udostępnia tę rodzinę wyłącznie do predykcji. `train()`, `val()` i
`export()` zawsze zgłaszają `NotImplementedError`. Model należy dostroić w
projekcie źródłowym i wczytać wynik. Walidacja zbioru danych jest pomijana,
ponieważ zastępcza pewność nadawałaby metryce COCO mAP mylący charakter, a
eksport pozostaje poza zakresem modelu generatywnego bez state dict możliwego do
śledzenia.

## Licencja

<provenance-box>

Licencja NVIDIA zezwala na użycie, powielanie i modyfikację, ale dla podmiotów
innych niż NVIDIA i jej jednostki stowarzyszone ogranicza model oraz wszelkie
pochodne wyłącznie do zastosowań niekomercyjnych, badawczych lub ewaluacyjnych.
Nie ma progu przychodów ani płatnego wyjątku. LocateAnything-3B łączy także dwa
inne licencjonowane komponenty: backbone językowy Qwen2.5-3B-Instruct na
licencji Qwen Research License oraz enkoder wizyjny MoonViT-SO-400M na licencji
MIT. LibreYOLO nie hostuje, nie tworzy kopii lustrzanej ani nie redystrybuuje
żadnego z nich. Przy pierwszym uruchomieniu `LibreLocateAnything` pobiera wagi i
wymagany zdalny kod bezpośrednio z `nvidia/LocateAnything-3B` na Hugging Face,
przypięte do jednego stałego commita.

</provenance-box>

## Cytowanie

<citation-block />
