---
title: Jak uruchomić RTMDet bez MMDetection
description: RTMDet to jeden z najszybszych dokładnych detektorów. Oficjalna instalacja nie działa z żadną wersją PyTorch wydaną w ciągu ostatnich dwóch lat. Oto alternatywa.
date: 2026-06-27
author: Xuban
tags: [LibreYOLO, rtmdet, object-detection, tutorial]
faq:
  - q: "Dlaczego mmcv nie instaluje się z nowszym PyTorch?"
    a: "Najnowsza wersja mmcv, 2.2.0 z kwietnia 2024 roku, udostępnia gotowe pliki binarne tylko dla torch do wersji 2.4 / CUDA 12.1. W każdej nowszej wersji PyTorch trzeba kompilować operacje C++/CUDA mmcv ze źródeł. Zajmuje to od 10 do 30 minut i wymaga zgodnego CUDA toolkit, nvcc oraz kompatybilnego kompilatora C++. Stąd biorą się błędy takie jak brak modułu mmcv._ext."
  - q: "Czy można uruchomić RTMDet bez instalowania MMDetection?"
    a: "Tak. LibreYOLO wczytuje wagi RTMDet przekonwertowane z checkpointów OpenMMLab i zapewnia inferencję równoważną bit po bicie z mmdetection dla tego samego checkpointu. Wystarczy podać nazwę LibreRTMDets.pt, a plik zostanie pobrany automatycznie. Dostępnych jest pięć rozmiarów, od Tiny do X, wszystkie działają przy 640 px."
  - q: "Czy RTMDet można bezpłatnie używać komercyjnie?"
    a: "Tak. MMDetection i RTMDet są objęte licencją Apache 2.0, przekonwertowane wagi podlegają tym samym warunkom Apache 2.0, a kod LibreYOLO jest objęty licencją MIT. Nie ma ograniczeń komercyjnych."
  - q: "Kiedy mmdetection nadal jest właściwym wyborem?"
    a: "Jeśli potrzebne jest trenowanie lub dostrajanie RTMDet z pełnym pipeline augmentacji MMDetection albo jeśli używany jest już rozbudowany ekosystem OpenMMLab z działającym mmcv. Do inferencji i wdrożenia lepszym rozwiązaniem jest LibreYOLO."
---

RTMDet to detektor czasu rzeczywistego od OpenMMLab, który osiąga wysoką dokładność COCO, zachowując szybkość wystarczającą do wdrożenia. Architektura jest przejrzysta. Instalacja już nie.

Uruchomienie RTMDet z oficjalnego źródła wymaga złożenia czterowarstwowego stosu OpenMMLab:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

Zakresy wersji są wąskie. mmdet 3.3.0 wymaga `mmcv < 2.2.0`, ale `mim install mmcv>=2.0.0` bez problemu instaluje wersję 2.2.0, która później kończy się błędem asercji w czasie działania. Wersję trzeba przypiąć ręcznie.

Jest też większy problem. Najnowsza wersja mmcv to 2.2.0, wydana w kwietniu 2024 roku i od tamtej pory nieaktualizowana. Udostępnia gotowe pliki binarne tylko dla **torch do wersji 2.4 / CUDA 12.1**. Świeża instalacja `pip install torch` daje dziś PyTorch 2.12. Oznacza to, że operacje C++/CUDA mmcv trzeba kompilować ze źródeł: zajmuje to od 10 do 30 minut i wymaga zgodnego CUDA toolkit, `nvcc` oraz kompatybilnego kompilatora C++. Z tego wynikają często zgłaszane błędy:

* `ModuleNotFoundError: No module named 'mmcv._ext'` -- skompilowane operacje nie powstały albo są niezgodne,

* `nvcc fatal: Unsupported gpu architecture 'compute_86'` na każdej karcie z serii RTX 30,

* `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'` -- stos wymusza powrót do Pythona 3.8,

* błąd segmentacji przy imporcie z powodu niezgodnej wersji GCC.

Własne FAQ OpenMMLab zaleca instalowanie mmcv przez pip zamiast mim, aby uniknąć problemu z wersjami. Strona Get Started zaleca użycie mim. Oba dokumenty są oficjalne. Są ze sobą sprzeczne.

Po uruchomieniu stosu inferencja nadal wymaga wybrania pliku konfiguracyjnego, którego nazwa koduje przepis trenowania, oraz osobno pobranego checkpointu z nazwą zawierającą skrót, a następnie połączenia ich przez rejestr, którego najpierw trzeba się nauczyć.

LibreYOLO zastępuje to wszystko:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")  # pobierany automatycznie przy pierwszym uruchomieniu
results = model("image.jpg", save=True)
```

Bez `mim`, macierzy zgodności wersji czterech pakietów, kompilowania ze źródeł, plików konfiguracyjnych, szukania checkpointu po skrócie i przypinania Pythona 3.8. Wagi są przekonwertowane z checkpointów OpenMMLab, a inferencja jest równoważna bit po bicie z mmdetection dla tego samego checkpointu.

Dostępnych jest pięć rozmiarów: Tiny, Small, Medium, Large i X. Wszystkie działają przy 640 px.

```python
print(results[0].boxes.xyxy)  # współrzędne xyxy
print(results[0].boxes.conf)  # wskaźniki pewności
```

## Kiedy oryginalne rozwiązanie nadal jest właściwym wyborem

Jeśli potrzebne jest trenowanie lub dostrajanie RTMDet z pełnym pipeline augmentacji MMDetection albo używany jest już rozbudowany ekosystem OpenMMLab z działającym mmcv, warto przy nim pozostać. Do inferencji i wdrożenia lepszym rozwiązaniem jest LibreYOLO.

## Uwaga dotycząca licencji

MMDetection i RTMDet są objęte licencją Apache 2.0. Kod LibreYOLO jest objęty licencją MIT. Wagi podlegają tym samym warunkom Apache 2.0 co upstream. Nie ma ograniczeń komercyjnych.

## Wypróbuj

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDetl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO jest objęte licencją MIT, działa w systemach Linux, Mac i Windows oraz obsługuje GPU, Apple Silicon i zwykły CPU bez zmian w kodzie. Jedno API obejmuje RTMDet, RT-DETR, RF-DETR, D-FINE, YOLOX, YOLO-NAS, segmentację, estymację pozy, estymację głębi i inne zadania.

Dodaj gwiazdkę na GitHubie: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Dokumentacja: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
