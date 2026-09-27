---
title: InternVL3
families:
  - internvl3
seo_title: 'InternVL3: detekcja z otwartym słownikiem w LibreYOLO'
description: >-
  Używaj InternVL3 w LibreYOLO do detekcji obiektów z otwartym słownikiem.
  Przewiduj z dowolną etykietą tekstową. Trenowanie, walidacja i eksport nie są
  obsługiwane.
lead: >-
  InternVL3 to natywny, duży model multimodalny wydany przez OpenGVLab, który
  wspólnie uczy się obrazu i języka w jednym etapie trenowania wstępnego.
  LibreYOLO udostępnia go jako detektor obiektów z otwartym słownikiem: dowolna
  lista etykiet tekstowych staje się zbiorem klas, bez stałej głowicy i bez
  potrzeby dostrajania.
keywords:
  - InternVL3
  - InternVL
  - model wizyjno-językowy
  - detekcja z otwartym słownikiem
  - VLM
  - OpenGVLab
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE


        model = LibreInternVL3(size="2b")


        # Otwarty słownik: działają dowolne słowa, a nie stała głowica klas.
        Ustawienie

        # obowiązuje w każdym późniejszym wywołaniu predict()/track(), dopóki
        nie zostanie zmienione.

        model.set_classes(["person", "bicycle", "dog"])

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Surowy czat
      language: python
      code: |
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # Bezpośredni dostęp pod wygodną warstwą detekcji: swobodne pytania,
        # zliczanie lub dowolny prompt nieobsługiwany przez wrapper ramek.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 6305f020d3079d71
---

## Instalacja

InternVL3 wymaga dodatku `vlm`, który instaluje `transformers` dla backbone z
szablonem czatu.

```bash
pip install "libreyolo[vlm]"
```

## Predykcja

`LibreInternVL3` jest klasą Pythona, a nie checkpointem `.pt`. Nie jest
wczytywany przez fabrykę `LibreYOLO()`, a CLI `libreyolo` go nie rozpoznaje.
Fabryka `LibreVLM(...)` (`from libreyolo import LibreVLM`) także udostępnia tę
rodzinę przez alias, na przykład `LibreVLM("internvl3-2b")`. Użyta poniżej klasa
jest tym, co tworzy ta fabryka. Wagi pochodzą z własnych repozytoriów `-hf`
OpenGVLab na Hugging Face, a nie z kopii lustrzanej LibreYOLO. Pierwsze
wywołanie pobiera je i zapisuje lokalnie w pamięci podręcznej, uprzednio
jednorazowo wypisując informację o licencji kontrolowanych wag Qwen.

<code-tabs name="predict" />

`result.boxes` zawiera przetworzone detekcje tak jak w każdej innej rodzinie.
Pewność jest wartością zastępczą. InternVL3 nie zwraca wyniku dla poszczególnych
ramek, dlatego każda detekcja otrzymuje tę samą stałą pewność, a `conf=` jedynie
odrzuca wiersze poniżej tej stałej, bez ich klasyfikowania. `iou` odrzuca prawie
identyczne ramki tej samej klasy powyżej podanego nakładania, będące skutkiem
powtarzania obiektu przez zachłanne dekodowanie. Nie jest to etap NMS osobny dla
każdej klasy. Pominięcie `set_classes()` pozostawia domyślne nazwy COCO-80.
Więcej informacji o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępne są trzy rozmiary, 1b, 2b i 8b, wszystkie jako natywne checkpointy
`-hf` OpenGVLab z backbone LLM Qwen, a nie dwuwieżową architekturą opisaną w
oryginalnej publikacji InternVL. Zestaw testowy LibreYOLO nie wykonał pomiarów
tej rodziny, dlatego nie ma opublikowanych wyników dokładności do ich
porównania. Rozmiar należy dobrać do własnego budżetu obliczeniowego.

LibreYOLO udostępnia tę rodzinę wyłącznie do predykcji. `train()`, `val()` i
`export()` zawsze zgłaszają `NotImplementedError`. Model należy dostroić w
projekcie źródłowym i wczytać wynik. Walidacja zbioru danych jest pomijana,
ponieważ zastępcza pewność nadawałaby metryce COCO mAP mylący charakter, a
eksport pozostaje poza zakresem modelu generatywnego bez state dict możliwego do
śledzenia.

## Licencja

<provenance-box>

Własny kod InternVL3 podlega licencji MIT, jest liberalny i nadaje się do
produktów komercyjnych oraz zamkniętych. Checkpointy `-hf` wczytywane przez tę
rodzinę zawierają backbone LLM Qwen i mają osobną licencję Alibaba Cloud Qwen
License. Pozwala ona bezpłatnie używać, modyfikować i redystrybuować model z
wymaganym oznaczeniem „Built with Qwen” lub „Improved using Qwen” oraz limitem
100 milionów aktywnych użytkowników miesięcznie dla zastosowań komercyjnych.
Powyżej tego limitu wymagane jest zezwolenie Alibaba. LibreYOLO nie hostuje ani
nie redystrybuuje tych wag. Przy pierwszym uruchomieniu `LibreInternVL3` pobiera
odpowiedni rozmiar bezpośrednio z `OpenGVLab/InternVL3-<size>-hf` na Hugging Face
i wyświetla przed pobraniem jednorazową informację o Qwen License.

</provenance-box>

## Cytowanie

<citation-block />
