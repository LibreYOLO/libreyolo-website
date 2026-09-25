---
title: "RT-DETR bez Ultralytics: RT-DETRv2 i v4 na Apache-2.0"
description: "Uruchamianie i dostrajanie RT-DETR, RT-DETRv2 i RT-DETRv4 z wagami Apache-2.0 w bibliotece MIT: jedno API do predykcji, trenowania, walidacji i eksportu."
date: 2026-09-24
author: Xuban
tags: [LibreYOLO, rt-detr, rtdetr, rt-detrv4, object-detection, license, tutorial]
faq:
  - q: "Czy RT-DETR można bezpłatnie używać komercyjnie?"
    a: "Tak, jeśli używana jest implementacja na liberalnej licencji. Oryginalny kod RT-DETR i RT-DETRv2 (lyuwenyu/RT-DETR) oraz RT-DETRv4 (RT-DETRs/RT-DETRv4) są objęte licencją Apache-2.0, a LibreYOLO obsługuje wszystkie trzy wersje z wagami Apache-2.0 w bibliotece MIT. Implementacja RT-DETR w pakiecie `ultralytics` jest objęta licencją AGPL-3.0, a alternatywą do użytku z zamkniętym kodem źródłowym jest płatna licencja Enterprise. To ogólne informacje, a nie porada prawna."
  - q: "Na jakiej licencji jest RT-DETRv4?"
    a: "Apache-2.0. Plik LICENSE w repozytorium github.com/RT-DETRs/RT-DETRv4 określa licencję Apache-2.0. RT-DETRv4 korzysta z modelu DINOv3 jako nauczyciela wyłącznie podczas trenowania. Opublikowane wagi ucznia nie zawierają parametrów DINOv3, więc licencja Meta na DINOv3 ich nie obejmuje."
  - q: "Czy można dostrajać RT-DETR bez Ultralytics?"
    a: "Tak. LibreYOLO dostraja checkpointy detekcyjne RT-DETR, RT-DETRv2 i RT-DETRv4 za pomocą `model.train()` lub polecenia `libreyolo train` w wierszu poleceń, zaczynając od opublikowanych wag COCO. Modele z obróconymi ramkami w RT-DETRv2 służą wyłącznie do inferencji."
  - q: "Jakie wagi RT-DETR udostępnia LibreYOLO?"
    a: "RT-DETR w wariantach r18, r34, r50, r50m, r101, l i x; RT-DETRv2 w wariantach r18, r34, r50, r50m i r101; RT-DETRv4 w wariantach s, m, l i x. Wszystkie są modelami do detekcji na COCO w rozdzielczości 640 px. RT-DETRv2 ma też modele z obróconymi ramkami n, s, m, l i x wytrenowane na DOTA v1.0 w rozdzielczości 1024 px. Każdy plik jest objęty licencją Apache-2.0."
---

RT-DETR to transformer do detekcji obiektów w czasie rzeczywistym opracowany przez Baidu. Dekoduje stały zestaw zapytań zamiast gęstej siatki, więc nie ma kroku NMS, którego próg trzeba dostrajać. Po wyszukaniu tego modelu jedną z pierwszych znalezionych implementacji jest ta z pakietu Python `ultralytics`. Rodzi to pytanie o licencję, którego sam model nigdy nie stawiał.

## Licencja RT-DETR zależy od repozytorium

Licencja dotyczy repozytorium, a nie modelu. Istnieje kilka repozytoriów RT-DETR i stosują różne licencje:

| Repozytorium | Zakres | Licencja |
|:---|:---|:---|
| [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | RT-DETR i RT-DETRv2, PyTorch i Paddle | Apache-2.0 |
| [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | RT-DETRv4 | Apache-2.0 |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | RT-DETR w pakiecie `ultralytics` | AGPL-3.0 lub licencja Enterprise |
| [LibreYOLO](https://github.com/LibreYOLO/libreyolo) | RT-DETR, RT-DETRv2 i RT-DETRv4 | kod MIT, wagi Apache-2.0 |

Implementacja Ultralytics jest solidna. Na [stronie RT-DETR](https://docs.ultralytics.com/models/rtdetr/) wymieniono wstępnie wytrenowane modele `rtdetr-l.pt` i `rtdetr-x.pt` oraz trenowanie, walidację, inferencję i eksport. Są one udostępniane w pakiecie na licencji AGPL-3.0. Ultralytics sprzedaje licencję Enterprise zespołom, które nie chcą udostępniać kodu źródłowego całego projektu. Jeśli te warunki nie pasują, architektura jest dostępna na licencji Apache-2.0 bezpośrednio od jej autorów. Artykuł [Wyjaśnienie licencji YOLO](/articles/yolo-licenses-explained) omawia ten sam schemat w rodzinie YOLO, a [przewodnik po licencji komercyjnej](/articles/yolo-commercial-license) wyjaśnia, co AGPL-3.0 oznacza dla produktu o zamkniętym kodzie źródłowym.

## Uruchamianie RT-DETR z LibreYOLO

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreRTDETRr18.pt")  # downloads from Hugging Face on first use
result = model(SAMPLE_IMAGE, save=True)

for box in result.boxes:
    print(box.cls, box.conf, box.xyxy)
```

To samo wywołanie działa z wiersza poleceń:

```bash
libreyolo predict model=LibreRTDETRr18.pt source=image.jpg save=True
```

`conf` i `max_det` filtrują dekodowanie top-k po zapytaniach i klasach. Parametr `iou` jest akceptowany, ale nieużywany, ponieważ nie ma NMS. Zwracany obiekt `Results` jest taki sam dla każdego modelu LibreYOLO, więc zmianę detektora można wprowadzić jedną linią kodu.

## RT-DETR, RT-DETRv2 i RT-DETRv4 w jednym loaderze

Wersja jest częścią nazwy pliku, a `LibreYOLO()` wybiera model na podstawie checkpointu, więc wszystkie trzy wersje ładuje się w ten sam sposób:

* **RT-DETR:** `LibreRTDETR` w wariantach r18, r34, r50, r50m, r101 (backbone ResNet) oraz l, x (HGNetv2).
* **RT-DETRv2:** `LibreRTDETRv2` w wariantach r18, r34, r50, r50m i r101. Zachowuje architekturę wersji 1 i zmienia sposób próbkowania przez deformowalną uwagę.
* **RT-DETRv4:** `LibreRTDETRv4` w wariantach s, m, l i x. Wykorzystuje ponownie architekturę D-FINE, a wagi powstają przez destylację wiedzy z modelu DINOv3 pełniącego rolę nauczyciela do modelu HGNetv2 pełniącego rolę ucznia.

Wszystkie wagi detekcyjne wytrenowano na COCO, a inferencja działa w rozdzielczości 640 px. Dla porównania, pliki README upstream podają 46.5 AP dla RT-DETR-R18 i 53.0 AP dla RT-DETR-L na COCO oraz wynik od 49.8 AP dla RT-DETRv4-S do 57.0 AP dla RT-DETRv4-X. Na [stronie dokumentacji RT-DETR](/docs/models/rt-detr) znajduje się własna tabela benchmarków LibreYOLO dla każdego checkpointu.

RT-DETRv2 obsługuje też obrócone ramki. `LibreRTDETRv2n-obb.pt` do `LibreRTDETRv2x-obb.pt` to oficjalne checkpointy DOTA v1.0, obejmujące 15 klas lotniczych w rozdzielczości 1024 px, przekonwertowane do formatu LibreYOLO:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv2n-obb.pt")
result = model("aerial.png", save=True)
print(result.obb.xywhr)  # (N, 5): cx, cy, w, h, radians
```

[Detekcja z obróconymi ramkami](/docs/tasks/oriented-detection) opisuje format etykiet i metryki.

## Dostrajanie RT-DETR na własnych danych

Trenowanie rozpoczyna się od opublikowanego checkpointu:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRr18.pt")
model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
```

Podczas rzeczywistego trenowania należy wskazać własny plik YAML zbioru danych w parametrze `data`. Każda wersja ma własną domyślną wartość współczynnika uczenia, a wersje 1 i 2 ustawiają współczynnik uczenia backbone na jedną dwudziestą wartości `lr0`, zgodnie z oryginalnym przepisem. Wiele GPU można wskazać w CLI jako `device=0,1`, a dostrajanie adapterami [LoRA](/docs/train/lora) włącza się przez `lora=True` po zainstalowaniu dodatku `lora`. Modele z obróconymi ramkami w RT-DETRv2 służą wyłącznie do inferencji. Informacje o zbiorach danych, augmentacji i loggerach znajdują się w sekcji [trenowania](/docs/train).

## Walidacja i eksport

```python
metrics = model.val(data="coco128.yaml")
print(metrics["metrics/mAP50-95"])

path = model.export(format="onnx")  # needs: pip install "libreyolo[onnx]"
```

`val()` zwraca zwykły słownik. Dostępne formaty eksportu obejmują ONNX, TorchScript, ExecuTorch, TensorRT i OpenVINO. Macierz eksportu na [stronie modelu](/docs/models/rt-detr) pokazuje, które formaty sprawdzono dla każdego zadania. Wyeksportowany plik `.onnx` lub `.engine` można ponownie załadować przez `LibreYOLO()`, a następnie otrzymuje się ten sam obiekt `Results`. Przewodniki dotyczące poszczególnych formatów są dostępne w sekcji [eksportu](/docs/export).

## Kiedy lepiej użyć oryginalnych repozytoriów

Jeśli trzeba odtworzyć wynik z publikacji przy użyciu dokładnych konfiguracji autorów, skorzystać z wersji Paddle albo samodzielnie przeprowadzić destylację wiedzy z modelu DINOv3 pełniącego rolę nauczyciela w RT-DETRv4, należy użyć repozytoriów upstream. LibreYOLO dostraja opublikowany model v4 pełniący rolę ucznia, ale nie uruchamia modelu nauczyciela. Jeśli projekt jest już objęty licencją AGPL-3.0 albo licencją Enterprise Ultralytics, nie ma powodu licencyjnego, aby go przenosić.

## Informacja o licencji

Kod LibreYOLO jest objęty licencją MIT. Każdy plik wag RT-DETR jest objęty licencją Apache-2.0, a każde repozytorium wag na Hugging Face zawiera pliki upstream LICENSE i NOTICE, które zgodnie z Apache-2.0 należy zachować przy redystrybucji wag. Wagi wytrenowane na własnych danych należą do użytkownika. RT-DETRv4 używa DINOv3 wyłącznie jako modelu nauczyciela podczas trenowania, a opublikowane modele ucznia nie zawierają parametrów DINOv3. To ogólne informacje, a nie porada prawna. Przed wydaniem należy sprawdzić aktualne pliki LICENSE.

## Wypróbuj

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv4s.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO jest objęte licencją MIT, działa w systemach Linux, Mac i Windows oraz obsługuje GPU, Apple Silicon i zwykły CPU bez zmian w kodzie.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentacja RT-DETR](/docs/models/rt-detr)
