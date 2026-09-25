---
title: "Alternatywa dla SuperGradients: YOLO-NAS do detekcji obiektów w czasie rzeczywistym"
description: "SuperGradients przestał być rozwijany po przejęciu Deci przez NVIDIA w 2024 roku. LibreYOLO to utrzymywana alternatywa, która obsługuje te same wagi YOLO-NAS: predykcję, trenowanie i eksport w ramach jednego API na licencji MIT."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, supergradients-alternative, yolo-nas, object-detection]
faq:
  - q: "Czy SuperGradients jest nadal utrzymywany?"
    a: "Nie. Ostatnie wydanie, 3.7.1, ukazało się w kwietniu 2024 roku, tuż przed przejęciem Deci przez NVIDIA. Od tego czasu repozytorium nie doczekało się nowych wydań, zgłoszenia pozostają bez odpowiedzi, a strona dokumentacji przestała działać. Projekt nie został formalnie zarchiwizowany, ale nikt się nim nie opiekuje."
  - q: "Jaka jest najlepsza alternatywa dla SuperGradients?"
    a: "Do uruchamiania YOLO-NAS biblioteka LibreYOLO ładuje te same wagi Deci za pomocą utrzymywanego API na licencji MIT, a do tego obsługuje trenowanie, walidację i eksport. Jeśli potrzebne są przepisy trenowania SuperGradients dla innych architektur, stare repozytorium nadal działa po przypięciu jego zależności."
  - q: "Czy można komercyjnie używać YOLO-NAS?"
    a: "Wstępnie wytrenowane wagi Deci są przeznaczone do użytku niekomercyjnego niezależnie od biblioteki, która je ładuje. Sama architektura jest objęta liberalną licencją, więc wytrenowanie modelu YOLO-NAS od zera na własnych danych daje model, który nie wywodzi się z żadnego checkpointu Deci."
  - q: "Czy LibreYOLO zastępuje wszystkie funkcje SuperGradients?"
    a: "Nie wszystkie. LibreYOLO nie eksportuje YOLO-NAS do CoreML, a ścieżka TensorRT dla tej rodziny nie została przetestowana. Przepisy trenowania ze świadomością kwantyzacji w SuperGradients również nie mają bezpośredniego odpowiednika. W tych przypadkach warto zachować stare repozytorium."
---

SuperGradients był biblioteką open source Deci do trenowania i miejscem rozwoju YOLO-NAS, jednego z najdokładniejszych detektorów czasu rzeczywistego, jakie kiedykolwiek wydano. W maju 2024 roku NVIDIA przejęła Deci, a rozwój SuperGradients ucichł. Ostatnie wydanie, 3.7.1, ukazało się 8 kwietnia 2024 roku. Dokumentacja na supergradients.com przestała działać. Około 120 zgłoszeń pozostaje otwartych, a zgłoszenie zatytułowane [„Czy ten projekt umarł?”](https://github.com/Deci-AI/super-gradients/issues/2062) nie doczekało się odpowiedzi od opiekuna projektu. To samo w sobie jest odpowiedzią.

Repozytorium nie zostało zarchiwizowane, więc na pierwszy rzut oka wygląda na aktywne. W praktyce porzucenie projektu daje o sobie znać już przy próbie instalacji: `super-gradients` przypina `torchmetrics==0.8`, co powoduje konflikt z aktualnymi stosami PyTorch, a przy okazji dodaje zależności od hydra, omegaconf, boto3 i tensorboard. Z każdym miesiącem przypięte wersje coraz bardziej kolidują z resztą środowiska, a poprawka nie nadchodzi.

Nie oznacza to, że YOLO-NAS jest gorszym modelem. Duży wariant nadal osiąga 52.2 mAP na COCO przy szybkości czasu rzeczywistego. Model jest w porządku, spłonął tylko jego dom.

<iframe
  src="https://visionanalysis.org/embed/scatter?highlight=yolonas-s%2Cyolonas-m%2Cyolonas-l"
  width="100%"
  height="420"
  style="border:0;border-radius:12px;overflow:hidden"
  loading="lazy"
  title="Dokładność YOLO-NAS a liczba parametrów - visionanalysis.org">
</iframe>

## Uruchamianie tych samych wag w LibreYOLO

LibreYOLO ładuje checkpointy YOLO-NAS bezpośrednio z CDN Deci, korzystając z tego samego API co dla wszystkich pozostałych rodzin modeli. Nie trzeba tworzyć konfiguracji hydra ani rozpakowywać wyników z dodatkowej nakładki predykcyjnej:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

Warianty detekcyjne S, M i L działają, podobnie jak wariant do estymacji pozy: wystarczy podstawić `LibreYOLONASs-pose.pt`, aby otrzymać punkty kluczowe COCO. Ponieważ każda rodzina zwraca ten sam obiekt `Results`, porównanie YOLO-NAS z RF-DETR lub D-FINE na własnych danych wymaga zmiany jednej linii zamiast przechodzenia na drugą bazę kodu.

## Trenowanie i eksport, nie tylko inferencja

SuperGradients był przede wszystkim biblioteką do trenowania, więc prawdziwa alternatywa musi umożliwiać trenowanie. LibreYOLO dostraja YOLO-NAS na własnym zbiorze danych za pomocą tego samego wywołania co w przypadku pozostałych modeli:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

Można też trenować model z losowo zainicjowanymi wagami, co ma znaczenie ze względu na licencję (więcej na ten temat poniżej). Walidacja zwraca metryki mAP dla dowolnego zbioru danych w wymaganym formacie, a eksport obejmuje ONNX, TorchScript, OpenVINO, NCNN i TFLite. To szerszy zakres formatów eksportu, niż oryginalna biblioteka kiedykolwiek oferowała dla YOLO-NAS. Pełne informacje znajdują się na [stronie dokumentacji YOLO-NAS](https://www.libreyolo.com/docs/models/yolo-nas), a krótsza notka wyjaśnia, że [YOLO-NAS jest nadal utrzymywany](/articles/yolo-nas-with-libreyolo).

## Kiedy warto pozostać przy starym repozytorium

Szczera ocena. Są trzy przypadki, w których warto zachować zainstalowane SuperGradients:

**CoreML.** LibreYOLO nie eksportuje YOLO-NAS do CoreML. Jeśli model ma działać na urządzeniach Apple i potrzebny jest plik `.mlpackage`, SuperGradients nadal zapewnia działającą ścieżkę.

**TensorRT.** SuperGradients dokumentował i testował przepływ TensorRT dla YOLO-NAS, łącznie ze wszystkimi osobliwościami rozmiaru batcha. Obsługa TensorRT dla tej rodziny w LibreYOLO nie została przetestowana.

**Przepisy trenowania ze świadomością kwantyzacji.** YOLO-NAS zaprojektowano z myślą o INT8, a SuperGradients udostępniał przepisy QAT, które pozwalały w pełni wykorzystać ten wariant. LibreYOLO eksportuje modele z kwantyzacją INT8 i FP16, ale nie ma odpowiednika tych przepisów trenowania.

Repozytorium nadal działa po przypięciu środowiska z wersjami z 2024 roku. Lepiej jednak nie opierać nowych projektów na fundamencie, którego nikt nie utrzymuje.

## Uwaga na temat wag

Wstępnie wytrenowane wagi YOLO-NAS należą do Deci i zostały wydane na licencji niekomercyjnej. Licencja obowiązuje niezależnie od biblioteki, która ładuje wagi. LibreYOLO nie hostuje ani nie kopiuje tych wag. Plik jest pobierany z publicznego CDN Deci, a przed rozpoczęciem pobierania wyświetlane są warunki Deci. Do badań i zastosowań niekomercyjnych można używać obu bibliotek.

Jeśli potrzebny jest komercyjny model YOLO-NAS, dostępna jest teraz prosta ścieżka: sama architektura jest objęta liberalną licencją, więc trenowanie od zera na własnych danych daje model, który nie wywodzi się z żadnego checkpointu Deci. LibreYOLO obsługuje dokładnie taki scenariusz za pomocą `LibreYOLONAS(None, size="s")`.

## Wypróbuj

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASl.pt")
results = model("image.jpg", save=True)
```

LibreYOLO jest objęte licencją MIT, działa w systemach Linux, Mac i Windows oraz obsługuje GPU, Apple Silicon i zwykły CPU bez zmian w kodzie. Jedno API obejmuje YOLO-NAS, RF-DETR, D-FINE, DEIM, YOLOX, RTMDet i wiele innych modeli, a także detekcję, segmentację, estymację pozy, klasyfikację, estymację głębi i śledzenie.

Dodaj gwiazdkę na GitHubie: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Dokumentacja: [libreyolo.com/docs](https://www.libreyolo.com/docs)
