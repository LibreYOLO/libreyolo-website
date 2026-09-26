---
title: API otwartego słownika
seo_title: 'API LibreOpenVocab: aliasy i argumenty'
description: >-
  Fabryka LibreOpenVocab, jej cztery rodziny i wszystkie aliasy, set_classes,
  wartości domyślne conf dla rodzin oraz zasady text_threshold i iou.
lead: >-
  LibreOpenVocab jest fabryką detektorów warunkowanych tekstem. Lista klas jest
  monitem zamiast stałej głowicy, dlatego słownik ustawia się przez set_classes,
  a model zwraca względem niego zwykły obiekt Results detekcji.
keywords:
  - LibreOpenVocab
  - detekcja z otwartym słownikiem
  - Grounding DINO
  - OWLv2
  - OMDet-Turbo
  - OV-DEIM
  - set_classes
last_verified: 1.5.0
verification: >-
  Aliasy odczytano z libreyolo/models/openvocab/__init__.py; repozytoria,
  rozmiary i progi z grounding_dino.py, owlv2.py, omdet_turbo.py i ov_deim.py;
  zasady wywołania z libreyolo/models/openvocab/base.py, wszystkie w wersji
  v1.5.0. Założenia projektowe pochodzą z
  docs/adr/0008-open-vocab-detector-contract.md.
snippets:
  install:
    - label: Bash
      language: bash
      code: |
        pip install 'libreyolo[openvocab]'
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-tiny")
        model.set_classes(["person", "skateboard", "handrail"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
source_hash: 64e4c641c6f8cde0
---

## Instalacja

Ta warstwa wymaga dodatku `openvocab`.

<code-tabs name="install" />

## Fabryka

```python
LibreOpenVocab(model: str = "grounding-dino-tiny", **kwargs) -> LibreOpenVocabDetector
```

`model` jest aliasem, a nie ścieżką. Przed wyszukaniem podkreślenia są zamieniane
na łączniki, dlatego nazwy kwalifikowane rodziną wyświetlane przez spis CLI,
takie jak `omdet_turbo-t` i `grounding_dino-t`, ładują się w podanej postaci.
Nieznany alias powoduje `ValueError` z listą wszystkich znanych aliasów.

Konstruktor przyjmuje `size`, `nb_classes=80`, `names=None`, `device="auto"`,
`task=None` i `text_threshold=None`. Przekazanie `names` odpowiada wywołaniu
`set_classes` bezpośrednio po załadowaniu. Przekazanie `text_threshold` rodzinie,
która go nie obsługuje, powoduje `TypeError`.

<code-tabs name="usage" />

## Rodziny i aliasy

| Rodzina | Aliasy | Rozmiary | Wagi |
|---|---|---|---|
| Grounding DINO | `grounding-dino`, `groundingdino`, `grounding-dino-tiny`, `groundingdino-tiny`, `grounding-dino-t`, `groundingdino-t`, `grounding-dino-base`, `groundingdino-base`, `grounding-dino-b`, `groundingdino-b` | `t`, `b` | `LibreYOLO/LibreGroundingDINOt`, `LibreYOLO/LibreGroundingDINOb` |
| OWLv2 | `owlv2`, `owl-v2`, `owlv2-base`, `owl-v2-base`, `owlv2-b16`, `owl-v2-b16`, `owlv2-large`, `owl-v2-large`, `owlv2-l14`, `owl-v2-l14` | `b16`, `l14` | `LibreYOLO/LibreOWLv2b16`, `LibreYOLO/LibreOWLv2l14` |
| OMDet-Turbo | `omdet-turbo`, `omdet`, `omdetturbo`, `omdet-turbo-tiny`, `omdet-turbo-swin-tiny`, `omdet-turbo-t` | `t` | `LibreYOLO/LibreOMDetTurbot` |
| OV-DEIM | `ov-deim`, `ovdeim`, `ov-deim-s`, `ovdeim-s`, `ov-deim-m`, `ovdeim-m`, `ov-deim-l`, `ovdeim-l` | `s`, `m`, `l` | `LibreYOLO/LibreOVDEIMs`, `LibreYOLO/LibreOVDEIMm`, `LibreYOLO/LibreOVDEIMl` |

Domyślnym aliasem jest `grounding-dino-tiny`.

`LibreGroundingDINO`, `LibreOWLv2` i `LibreOMDetTurbo` są eksportowane na poziomie
pakietu i można je konstruować bezpośrednio z `size=`. OV-DEIM jest dostępny przez
powyższe aliasy fabryki.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreOpenVocabDetector
```

Ustawia słownik dla każdego późniejszego wywołania `predict()` i zwraca model,
co pozwala łączyć wywołania. Lista nie może być pusta, musi zawierać wyłącznie
ciągi, a jej wpisy muszą być unikatowe przy porównaniu bez uwzględniania wielkości
liter. Puste etykiety są odrzucane. Przekazanie samego ciągu powoduje `TypeError`,
ponieważ zostałby rozłożony na klasy jednoznakowe.

Po wywołaniu `model.names` odwzorowuje `0..N-1` na etykiety w podanej kolejności,
a `model.nb_classes` ma wartość `N`.

## Argumenty wywołania

Ta warstwa używa standardowej powierzchni predykcji z trzema różnicami.

`conf` przyjmuje wartość domyślną danej rodziny zamiast wspólnej wartości 0,25:

| Rodzina | Domyślne conf | Tłumienie |
|---|---|---|
| Grounding DINO | 0,25 | |
| OWLv2 | 0,1 | |
| OMDet-Turbo | 0,3 | Własne przetwarzanie końcowe, próg 0,5, respektuje `iou=` |
| OV-DEIM | 0,25 | Dopasowanie jeden do jednego z wyborem top-K, bez tłumienia |

`iou=` ma znaczenie tylko w rodzinie wykonującej tłumienie. OMDet-Turbo przyjmuje
próg jako argument i domyślnie używa 0,5, gdy `iou=` nie jest ustawione. Pozostałe
trzy nie wykonują tłumienia, więc przekazanie im `iou=` powoduje ostrzeżenie i
jest ignorowane.

`text_threshold=` działa tylko w Grounding DINO, gdzie ma domyślnie wartość 0,25.
Można przekazać go podczas konstrukcji jako trwałą wartość albo przy pojedynczym
wywołaniu. Wartości pojedynczego wywołania nie można łączyć ze `stream=True`,
ponieważ wyniki strumieniowe są generowane leniwie. Zamiast tego ustaw ją w
konstruktorze. Każda inna rodzina zgłasza dla niej `TypeError`.

`imgsz=` powoduje `ValueError`: potok przetwarzania wstępnego sam odpowiada za
zmianę rozmiaru w tej warstwie. `augment=True` również powoduje błąd, ponieważ
augmentacja podczas testowania nie należy tutaj do zakresu. Rozmiary wejścia
zapisano tylko jako odniesienie dla rodzin: Grounding DINO 800, OWLv2 960 i 1008,
OMDet-Turbo 640, OV-DEIM 640.

## Nieobsługiwane funkcje

`train()`, `val()`, `track()` i `export()` zgłaszają `NotImplementedError`.
Model należy dostroić w projekcie źródłowym i załadować wynikowe wagi. Zamiast
śledzenia uruchamiaj `predict()` osobno dla każdej klatki. Walidacja wymagałaby
osobnego walidatora, ponieważ wspólny walidator detekcji wywołuje model tensorami
obrazu, podczas gdy ta warstwa wymaga wejść warunkowanych tekstem.

