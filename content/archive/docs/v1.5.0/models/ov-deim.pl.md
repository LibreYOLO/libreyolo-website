---
title: OV-DEIM
families:
  - ov_deim
seo_title: 'OV-DEIM w LibreYOLO: detekcja z otwartym słownikiem'
description: >-
  Używaj OV-DEIM w LibreYOLO do detekcji z otwartym słownikiem w stylu DETR i w
  czasie rzeczywistym. Zainstaluj dodatek openvocab i uruchamiaj predykcję ze
  swobodnym słownikiem tekstowym.
lead: >-
  OV-DEIM to detektor obiektów z otwartym słownikiem w stylu DETR, który
  dopasowuje zapytania dekodera do embeddingów tekstowych z dołączonej wieży
  tekstowej MobileCLIP. LibreYOLO udostępnia jego natywny port jako rodzinę
  wyłącznie do predykcji na poziomie detektorów z otwartym słownikiem.
keywords:
  - OV-DEIM
  - DEIMv2
  - detekcja obiektów z otwartym słownikiem
  - detekcja w czasie rzeczywistym
  - detekcja zero-shot
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-s")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Zastąpienie słownika
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-l")
        model.set_classes(["traffic light", "bicycle"])
        first = model.predict(SAMPLE_IMAGE, conf=0.3)

        # Drugie wywołanie set_classes() całkowicie zastępuje słownik i ponownie
        # tworzy jego embeddingi w wieży tekstowej. Pusty wynik jest poprawnym
        # rezultatem, a nie błędem.
        model.set_classes(["giraffe"])
        second = model.predict(SAMPLE_IMAGE, conf=0.5)
        print(second.names, len(second))
source_hash: 0c295f555a9eb303
---

## Instalacja

OV-DEIM jest wczytywany przez poziom detektorów z otwartym słownikiem w
LibreYOLO, który wymaga dodatku `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

W przeciwieństwie do pozostałych rodzin tego poziomu OV-DEIM jest natywnym
portem LibreYOLO, a nie wrapperem `transformers`. Nie istnieje dla niego klasa
modelu `transformers`, ale ten sam dodatek obejmuje pakiety `huggingface_hub`,
`safetensors`, `regex` i `ftfy` wymagane podczas predykcji.

## Predykcja

OV-DEIM nie jest checkpointem wczytywanym przez LibreYOLO za pomocą
`LibreYOLO()`. Korzysta z pokrewnej fabryki `LibreOpenVocab`, która przy
pierwszym użyciu pobiera snapshot Hugging Face i zapisuje go w pamięci
podręcznej w `weights/`.

<code-tabs name="predict" />

`set_classes()` ustawia trwały słownik tekstowy. Ponowne wywołanie całkowicie
zastępuje listę, a pominięcie zachowuje domyślne etykiety COCO-80. Pusty wynik
jest poprawnym rezultatem, a nie błędem. Każde zapytanie dekodera jest oceniane
według podobieństwa cosinusowego względem embeddingów tekstowych z dołączonej
wieży tekstowej MobileCLIP-B(LT). Są one obliczane na bieżąco dla ustawionego
słownika i zachowywane w pamięci podręcznej do jego zmiany, dlatego dowolne
prompty działają bez wcześniej obliczonego pliku embeddingów.

OV-DEIM nie ma progu tokenów tekstowych. Detekcje filtruje wyłącznie `conf`, a
przekazanie `text_threshold` zgłasza błąd. Dopasowanie jest wyborem top-K jeden
do jednego, dlatego nie jest tu uruchamiane tłumienie niemaksymalne. Argument
`iou` jest przyjmowany dla zgodności API, ale powoduje ostrzeżenie i nic nie
robi. Argumenty `imgsz` i `augment=True` są odrzucane. Model ma stałe wejście z
letterboxingiem, a augmentacja podczas testowania pozostaje poza zakresem tego
poziomu. `predict()` dla jednego obrazu zwraca jeden obiekt `Results`, a nie
listę. Katalog, listę obrazów lub `stream=True` ze źródłem wideo należy
przekazać, aby uzyskać kilka wyników. Dla tej rodziny nie ma ścieżki CLI.
`libreyolo predict` wczytuje przez `LibreYOLO()` wyłącznie checkpointy `.pt`,
dlatego rodziny `LibreOpenVocab` uruchamia się z Pythona. Informacje o typach
źródeł i streamingu zawiera strona [predykcji](/docs/predict).

Każde wywołanie `predict()` uruchamia także dołączoną wieżę tekstową
MobileCLIP-B(LT), aby utworzyć embeddingi bieżącego słownika. Sekcja Licencja
opisuje dodane przez nią warunki.

## Warianty

Dostępne są trzy checkpointy, `s`, `m` i `l`. `s` jest domyślnym rozmiarem tego
poziomu, gdy nie podano żadnego. W przeciwieństwie do pozostałych rodzin tego
poziomu OV-DEIM jest natywnym portem, a nie wrapperem `transformers`. LibreYOLO
dołącza moduły detektora na tej samej licencji Apache-2.0 co kod źródłowy i
ponownie wykorzystuje adapter backbone DINOv3 zbudowany już dla rodziny
DEIMv2. Backbone checkpointu `l` jest dostrojonym DINOv3-S, objętym osobną
licencją DINOv3 License firmy Meta. Nie opublikowano jeszcze wyników dokładności
ani opóźnienia tej rodziny.

Trenowanie, walidacja zbioru danych i eksport pozostają poza zakresem tego
poziomu. `train()`, `val()` i `export()` zawsze zgłaszają
`NotImplementedError`. Jest to wrapper przeznaczony wyłącznie do predykcji z
opublikowanym checkpointem.

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box>

OV-DEIM nakłada na każde wywołanie predykcji trzy licencje źródłowe: wagi
detektora podlegają własnej licencji CC BY-NC 4.0 projektu OV-DEIM, internetowa
wieża tekstowa licencji Apple Machine Learning Research Model przeznaczonej
wyłącznie do badań, a w przypadku checkpointu `l` dostrojony backbone DINOv3-S
licencji DINOv3 License firmy Meta. Wszystkie trzy teksty licencji są dołączone
do repozytorium wag LibreYOLO.

</provenance-box>

## Cytowanie

<citation-block />
