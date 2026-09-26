---
title: LFM2-VL
families:
  - lfm2vl
seo_title: 'LFM2-VL: detekcja z otwartym słownikiem w LibreYOLO'
description: >-
  Używaj LFM2-VL w LibreYOLO do detekcji obiektów z otwartym słownikiem
  bezpośrednio na urządzeniu. Przewiduj z dowolną etykietą tekstową. Trenowanie,
  walidacja i eksport nie są obsługiwane.
lead: >-
  LFM2-VL to kompaktowy model wizyjno-językowy przeznaczony do działania na
  urządzeniu, wydany przez Liquid AI. LibreYOLO udostępnia go jako detektor
  obiektów z otwartym słownikiem: dowolna lista etykiet tekstowych staje się
  zbiorem klas, bez stałej głowicy i bez potrzeby dostrajania.
keywords:
  - LFM2-VL
  - LFM2
  - Liquid AI
  - model wizyjno-językowy
  - detekcja z otwartym słownikiem
  - VLM
  - VLM na urządzeniu
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE


        model = LibreLFM2VL(size="450m")


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
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # Bezpośredni dostęp pod wygodną warstwą detekcji: swobodne pytania,
        # zliczanie lub dowolny prompt nieobsługiwany przez wrapper ramek.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 40237f0ecc0d2cd5
---

## Instalacja

LFM2-VL wymaga dodatku `vlm`, który instaluje `transformers` dla backbone z
szablonem czatu.

```bash
pip install "libreyolo[vlm]"
```

## Predykcja

`LibreLFM2VL` jest klasą Pythona, a nie checkpointem `.pt`. Nie jest wczytywany
przez fabrykę `LibreYOLO()`, a CLI `libreyolo` go nie rozpoznaje. Fabryka
`LibreVLM(...)` (`from libreyolo import LibreVLM`) także udostępnia tę rodzinę
przez alias, na przykład `LibreVLM("lfm2-vl-450m")`. Użyta poniżej klasa jest
tym, co tworzy ta fabryka. Wagi pochodzą z własnego repozytorium Liquid AI na
Hugging Face, a nie z kopii lustrzanej LibreYOLO. Pierwsze wywołanie pobiera je
i zapisuje lokalnie w pamięci podręcznej, uprzednio jednorazowo wypisując
informację o licencji.

<code-tabs name="predict" />

`result.boxes` zawiera przetworzone detekcje tak jak w każdej innej rodzinie.
Pewność jest wartością zastępczą. LFM2-VL nie zwraca wyniku dla poszczególnych
ramek, dlatego każda detekcja otrzymuje tę samą stałą pewność, a `conf=` jedynie
odrzuca wiersze poniżej tej stałej, bez ich klasyfikowania. `iou` odrzuca prawie
identyczne ramki tej samej klasy powyżej podanego nakładania, będące skutkiem
powtarzania obiektu przez zachłanne dekodowanie. Nie jest to etap NMS osobny dla
każdej klasy. Pominięcie `set_classes()` pozostawia domyślne nazwy COCO-80.
Więcej informacji o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępne są dwa rozmiary, 450m i 1.6b, oba z wydania LFM2.5-VL firmy Liquid AI,
zbudowane do wdrażania na urządzeniu. Zestaw testowy LibreYOLO nie wykonał
pomiarów tej rodziny, dlatego nie ma opublikowanych wyników dokładności do ich
porównania. Rozmiar należy dobrać do własnego budżetu obliczeniowego.

LibreYOLO udostępnia tę rodzinę wyłącznie do predykcji. `train()`, `val()` i
`export()` zawsze zgłaszają `NotImplementedError`. Model należy dostroić w
projekcie źródłowym i wczytać wynik. Walidacja zbioru danych jest pomijana,
ponieważ zastępcza pewność nadawałaby metryce COCO mAP mylący charakter, a
eksport pozostaje poza zakresem modelu generatywnego bez state dict możliwego do
śledzenia.

## Licencja

<provenance-box>

Licencja LFM Open License v1.0 zezwala na użycie komercyjne, powielanie i
modyfikację, ale tylko poniżej progu 10 milionów dolarów rocznego przychodu.
Podmiot prawny osiągający lub przekraczający ten próg nie uzyskuje na mocy tej
umowy żadnej licencji na użycie komercyjne i musi skontaktować się bezpośrednio
z Liquid AI. Kwalifikowane organizacje non-profit są zwolnione z progu w
przypadku zastosowań niekomercyjnych lub badawczych. LibreYOLO nie udostępnia
kodu źródłowego LiquidAI, ponieważ model jest wczytywany przez bibliotekę
`transformers` na licencji Apache-2.0, ani nie hostuje i nie redystrybuuje wag.
Przy pierwszym uruchomieniu `LibreLFM2VL` pobiera odpowiedni rozmiar bezpośrednio
z własnego repozytorium Liquid AI na Hugging Face i wyświetla przed pobraniem
jednorazową informację.

</provenance-box>

## Cytowanie

<citation-block />
