---
title: Jak uruchomić Depth Anything V2 w Pythonie za pomocą LibreYOLO
description: Depth Anything V2 generuje najnowocześniejsze mapy głębi z pojedynczego obrazu. Oto jak uruchomić model w dwóch wierszach kodu za pomocą LibreYOLO.
date: 2026-06-26
author: Xuban
tags: [LibreYOLO, depth-anything-v2, depth-estimation, tutorial]
faq:
  - q: "Jak najprościej uruchomić Depth Anything V2?"
    a: "Wystarczą dwa wiersze kodu z LibreYOLO: wczytanie modelu LibreDepthAnythingV2l-depth.pt po nazwie i wywołanie predict z save=True. Wagi są pobierane automatycznie przy pierwszym użyciu, a normalizacja, mapa kolorów i wybór urządzenia są obsługiwane automatycznie na CUDA, Apple Silicon lub zwykłym CPU."
  - q: "Czy można komercyjnie używać Depth Anything V2?"
    a: "Tylko encoder Small jest objęty licencją Apache 2.0. Base, Large i Giant są objęte licencją CC-BY-NC-4.0, więc najmocniejsze checkpointy są przeznaczone do użytku niekomercyjnego. Licencja projektu źródłowego obowiązuje niezależnie od tego, która biblioteka wczytuje wagi."
  - q: "Czy LibreYOLO obsługuje trenowanie Depth Anything V2?"
    a: "Nie. Trenowanie pozostaje w oryginalnym repozytorium; LibreYOLO obsługuje inferencję, wideo i walidację dla zadania estymacji głębi."
---

![Guggenheim w Bilbao obok mapy głębi](/articles/simplest-way-to-run-depth-anything-v2/guggenheim-bilbao-input-vs-depth.jpg)

Depth Anything V2 generuje obecnie jedne z najlepszych dostępnych map głębi z pojedynczego obrazu.

Dla osób znających ekosystem YOLO oficjalne repozytorium nie należy do najłatwiejszych w użyciu. Uzyskanie z niego pojedynczej mapy głębi wymaga kilku ręcznych kroków:

* ręcznego pobrania właściwego checkpointu i umieszczenia go w odpowiednim katalogu,

* dopasowania encodera do jego konfiguracji (`encoder="vitl"`, `features=256`, `out_channels=...`),

* samodzielnego napisania kroku normalizacji i przypisania kolorów,

* obsłużenia wyboru urządzenia, aby model działał na Macu lub CPU, a nie tylko na CUDA,

* poznania API inferencji, które wygląda zupełnie inaczej niż pozostała część stosu.

Żaden z tych kroków nie jest trudny. To po prostu dodatkowa praca, którą można pominąć.

LibreYOLO wczytuje te same wagi Depth Anything V2 i udostępnia zadanie estymacji głębi przez jedno wywołanie `predict()`. Wystarczy podać nazwę modelu, a zostanie pobrany przy pierwszym użyciu:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDepthAnythingV2l-depth.pt")  # auto-downloads on first run
model.predict("image.jpg", save=True)     # writes a colorized depth map to disk
```

To wszystko. Osobom znającym standardowe API YOLO ten schemat wyda się znajomy: wczytanie modelu po nazwie, wywołanie `predict` i zapis wizualizacji przez `save=True`. To dokładnie to samo wywołanie, którego używa się do detekcji obiektów, z tą różnicą, że wynik zawiera mapę głębi zamiast ramek. Mapa kolorów, normalizacja i wybór urządzenia są obsługiwane automatycznie. Model działa tak samo w systemie Linux z CUDA, na Macu z Apple Silicon i na zwykłym CPU. Bez zmian w kodzie.

To samo wywołanie pozwala też pobrać surowe wartości głębi i bezpośrednio na nich pracować. Trenowanie samego Depth Anything V2 pozostaje w oryginalnym repozytorium; LibreYOLO obsługuje inferencję, wideo i walidację.

Dokładnie to samo wywołanie w jednym wierszu, na bardzo różnych scenach:

![Obraz przykładowej sceny parkourowej obok mapy głębi: osoby skaczące z przodu wyraźnie odcinają się od betonowych ścian w tle.](/articles/simplest-way-to-run-depth-anything-v2/parkour-input-vs-depth.jpg)

![Widok z lotu ptaka na zatokę La Concha w Donostii obok mapy głębi: łodzie i linia brzegowa są widoczne na pierwszym planie, a otwarta woda się oddala.](/articles/simplest-way-to-run-depth-anything-v2/donostia-bay-input-vs-depth.jpg)

![Kolumnowy dziedziniec Casa de Juntas de Gernika obok mapy głębi, ukazującej oddalającą się architekturę.](/articles/simplest-way-to-run-depth-anything-v2/gernika-casa-juntas-input-vs-depth.jpg)

![Tłum na nocnym festiwalu na Plaza de la Constitucion w Donostii obok mapy głębi: osoby w rzędach z przodu odcinają się od oświetlonej fasady w tle.](/articles/simplest-way-to-run-depth-anything-v2/donostia-plaza-crowd-input-vs-depth.jpg)

Informacja o wagach: LibreYOLO udostępnia przekonwertowane checkpointy Depth Anything V2 i pobiera je przy pierwszym użyciu, więc nie trzeba niczego pobierać ręcznie. Nadal obowiązuje licencja projektu źródłowego: encoder Small jest objęty licencją Apache-2.0, a Base, Large i Giant licencją CC-BY-NC-4.0 (użytek niekomercyjny), więc najmocniejsze checkpointy są przeznaczone do użytku niekomercyjnego. Przy pracy offline lub samodzielnej konwersji można skorzystać z jednorazowego skryptu konwersji:

```bash
# optional: convert an official checkpoint yourself instead of auto-downloading
python weights/convert_depth_anything_v2_weights.py \
  depth_anything_v2_vitl.pth weights/LibreDepthAnythingV2l-depth.pt
```

## Wypróbuj

```bash
pip install libreyolo
```

LibreYOLO to najbardziej kompletna biblioteka wizji komputerowej, którą można zainstalować przez pip. Jedno znajome API obejmuje coraz więcej najnowocześniejszych modeli: detekcję obiektów (RF-DETR, D-FINE, DEIM), segmentację, estymację pozy, obrócone ramki oraz teraz estymację głębi z Depth Anything V2, a także trenowanie i walidację dla obsługiwanych zadań. Biblioteka działa we wszystkich głównych systemach operacyjnych, na GPU i CPU, a w całości jest objęta licencją MIT, więc można wdrażać ją komercyjnie bez dodatkowych warunków.

Dodaj gwiazdkę na GitHubie: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Dokumentacja: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
