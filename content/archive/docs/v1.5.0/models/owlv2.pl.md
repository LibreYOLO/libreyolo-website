---
title: OWLv2
families:
  - owlv2
seo_title: 'OWLv2 w LibreYOLO: detekcja obiektów zero-shot'
description: >-
  Używaj OWLv2 w LibreYOLO do wykrywania dowolnego obiektu opisanego tekstem.
  Zainstaluj dodatek openvocab i uruchamiaj predykcję ze swobodnym słownikiem
  tekstowym.
lead: >-
  OWLv2 to detektor obiektów z otwartym słownikiem opracowany przez Google
  Research, który ocenia regiony obrazu względem embeddingów tekstowych z
  enkodera w stylu CLIP. LibreYOLO udostępnia go jako rodzinę wyłącznie do
  predykcji na poziomie detektorów z otwartym słownikiem.
keywords:
  - OWLv2
  - OWL-ViT
  - detekcja obiektów z otwartym słownikiem
  - detekcja zero-shot
  - detektor sterowany tekstem
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Domyślny słownik
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        # Pominięcie set_classes() zachowuje domyślny słownik COCO-80 tego
        poziomu.

        model = LibreOpenVocab("owlv2-l14")

        result = model.predict(SAMPLE_IMAGE, conf=0.1)

        print(result.names)
source_hash: 2d0ce68af0daabb7
---

## Instalacja

OWLv2 jest wczytywany przez poziom detektorów z otwartym słownikiem w LibreYOLO,
który wymaga dodatku `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Ten dodatek instaluje `transformers` i `timm`, biblioteki Hugging Face używane
przez ten poziom.

## Predykcja

OWLv2 nie jest checkpointem wczytywanym przez LibreYOLO za pomocą
`LibreYOLO()`. Korzysta z pokrewnej fabryki `LibreOpenVocab`, która przy
pierwszym użyciu pobiera snapshot Hugging Face i zapisuje go w pamięci
podręcznej w `weights/`.

<code-tabs name="predict" />

`set_classes()` ustawia trwały słownik tekstowy. Ponowne wywołanie zastępuje
listę, a pominięcie zachowuje domyślne etykiety COCO-80. Każda etykieta jest
umieszczana w stałym szablonie promptu przed przekazaniem do wieży tekstowej,
zgodnie ze sposobem trenowania `Owlv2ForObjectDetection` z `transformers`.

OWLv2 nie ma progu tokenów tekstowych. Detekcje filtruje wyłącznie `conf`, a
przekazanie `text_threshold` zgłasza błąd. Argument `iou` jest przyjmowany dla
zgodności API, ale powoduje ostrzeżenie i nic nie robi, ponieważ nie jest tu
uruchamiane tłumienie niemaksymalne. Argumenty `imgsz` i `augment=True` są
odrzucane. Procesor `transformers` sam kontroluje zmianę rozmiaru, a augmentacja
podczas testowania pozostaje poza zakresem tego poziomu. `predict()` dla jednego
obrazu zwraca jeden obiekt `Results`, a nie listę. Katalog, listę obrazów lub
`stream=True` ze źródłem wideo należy przekazać, aby uzyskać kilka wyników. Dla
tej rodziny nie ma ścieżki CLI. `libreyolo predict` wczytuje przez `LibreYOLO()`
wyłącznie checkpointy `.pt`, dlatego rodziny `LibreOpenVocab` uruchamia się z
Pythona. Informacje o typach źródeł i streamingu zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępne są dwa checkpointy, `b16` (base, rozmiar patcha 16) i `l14` (large,
rozmiar patcha 14). `b16` jest domyślnym rozmiarem tego poziomu, gdy nie podano
żadnego. Oba tworzą kopię lustrzaną oficjalnego wydania Google Research za
pośrednictwem `Owlv2ForObjectDetection` z `transformers`, pobieraną jednorazowo
do hostowanego przez LibreYOLO snapshotu Hugging Face, który zachowuje pliki
źródłowe. Nie opublikowano jeszcze wyników dokładności ani opóźnienia tej
rodziny.

Trenowanie, walidacja zbioru danych i eksport pozostają poza zakresem tego
poziomu. `train()`, `val()` i `export()` zawsze zgłaszają
`NotImplementedError`. Jest to wrapper przeznaczony wyłącznie do predykcji z
opublikowanym checkpointem.

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
