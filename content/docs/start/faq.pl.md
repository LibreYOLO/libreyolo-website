---
title: FAQ
seo_title: FAQ LibreYOLO
description: >-
  Krótkie odpowiedzi na pytania wspólne dla wszystkich modeli LibreYOLO: sprzęt,
  licencje, wagi, urządzenia, trenowanie, obsługa eksportu i CLI.
lead: >-
  Odpowiedzi na pytania, które nie dotyczą jednej rodziny modeli. Informacje
  właściwe dla rodziny znajdują się na jej stronie.
keywords:
  - LibreYOLO FAQ
  - czy LibreYOLO wymaga GPU
  - licencja LibreYOLO
  - gdzie są wagi LibreYOLO
  - LibreYOLO CLI
  - LibreYOLO offline
last_verified: 1.5.0
source_hash: a729b43a6642f2a0
---

## Od którego modelu zacząć?

Od YOLOv9 w przypadku detektora CNN i RF-DETR w przypadku detektora
transformerowego. Oba należą do poziomu flagowego, co oznacza, że funkcje są
projektowane i walidowane na GPU najpierw względem nich. Zobacz
[YOLOv9](/docs/models/yolov9) i [RF-DETR](/docs/models/rf-detr) albo stronę
[wszystkich modeli](/docs/models), aby poznać pozostałe.

## Czy potrzebne jest GPU?

Nie. Każdy model działa na CPU, a wszystko w
[szybkim starcie](/docs/quickstart) napisano tak, aby tam działało. GPU wpływa
na czas trenowania i inferencji wideo, a nie na to, czy są możliwe.

## Jak LibreYOLO wybiera urządzenie?

Domyślne ustawienie `device="auto"` używa CUDA, gdy PyTorch zgłasza jego
dostępność, następnie Metal Performance Shaders, gdy jest dostępne, a w innym
przypadku CPU. Aby przypiąć urządzenie, przekaż `device` do modelu albo do
`predict`, `train`, `val` i `export`. Akceptowane są `"cpu"`, `"cuda"`,
`"cuda:0"`, `"mps"`, sama liczba całkowita, taka jak `0`, oraz ciąg cyfr.
Dwie ostatnie formy są rozwijane do `cuda:<n>`.

Polecenie `libreyolo checks` wyświetla kompilację Torch, wersje CUDA i cuDNN
oraz każde widoczne GPU. Jeśli polecenie nie pokazuje CUDA, zainstalowany pakiet
PyTorch jest kompilacją dla CPU. Strona [instalacji](/docs/install) opisuje jego
zastąpienie.

## Gdzie trafiają pobrane wagi?

Do katalogu `weights/` względem katalogu roboczego. Odwołanie do modelu bez
składnika katalogu jest rozwiązywane w tym miejscu i pobierane przy pierwszym
użyciu. Odwołanie zawierające katalog jest używane dokładnie w podanej postaci
i nigdy nie jest pobierane. Zobacz [checkpointy i wagi](/docs/weights).

## Czy można działać bez dostępu do sieci?

Tak. Wystarczy raz pobrać checkpointy na połączonej maszynie i skopiować katalog
`weights/`. Później nic nie będzie łączyć się z siecią. Współdzielona ścieżka
tylko do odczytu również działa, ponieważ odwołanie zawierające katalog jest
traktowane dosłownie. Zbiory danych są rozwiązywane w `~/datasets` lub katalogu
wskazanym przez `LIBREYOLO_DATASETS_DIR`.

## Czy LibreYOLO można używać komercyjnie?

Kod jest objęty licencją MIT. Wstępnie wytrenowane wagi są osobną kwestią:
mogą dziedziczyć warunki projektu lub zbioru danych, z którego pochodzą, a te
warunki nie są jednolite nawet w obrębie jednej rodziny. Rozstrzygająca jest
licencja konkretnego repozytorium Hugging Face, a każda strona modelu zawiera
sekcję licencyjną, która ją przytacza. Gdy wagi są ograniczone, LibreYOLO
wyświetla ograniczenie przed rozpoczęciem pobierania.

## Czy można wczytać checkpoint z innego projektu?

Zwykle tak, przekazując jego ścieżkę do `LibreYOLO()`. Rozpoznane układy ze
źródeł nadrzędnych są konwertowane podczas wczytywania z zachowaniem liczby klas
i nazw, a checkpoint LibreYOLO zostaje zapisany obok źródła. Strona
[importowania istniejących wag](/docs/migrate) opisuje rozpoznawane formaty
i przypadki wymagające skryptu konwersji.

## Dlaczego train zgłasza NotImplementedError?

Ponieważ ta rodzina obsługuje tylko inferencję, a wyjątek podaje przyczynę.
Predykcja, walidacja oraz eksport, jeśli jest obsługiwany, działają. W LibreYOLO
nie ma pętli trenowania dla tej architektury. Poziom obsługi w nagłówku strony
modelu informuje o tym przed próbą. Zobacz
[podstawowe pojęcia](/docs/concepts).

## Co zwraca val?

Zwykły słownik, a nie obiekt. Klucze detekcji obejmują `metrics/precision`,
`metrics/recall`, `metrics/mAP50` i `metrics/mAP50-95`. Inne zadania zwracają
odpowiednie dla nich klucze, takie jak `metrics/accuracy_top1` dla klasyfikacji
albo `metrics/PQ`, `metrics/SQ` i `metrics/RQ` dla segmentacji panoptycznej.

## Jak uruchomić model na katalogu, wideo lub kamerze internetowej?

Należy przekazać źródło. Ścieżka pliku oznacza jeden obraz, katalog oznacza
wszystkie zawarte obrazy, ścieżka wideo oznacza wideo, liczba całkowita jest
indeksem kamery internetowej, a adres URL RTSP, RTMP, TCP, UDP lub HLS jest
transmisją na żywo. Plik `.streams` wymienia kilka źródeł jednocześnie. Źródła
na żywo wymagają `stream=True`, co zwraca po jednym `Results` na klatkę zamiast
budować listę. Tej samej flagi warto używać dla długich filmów i dużych
katalogów. Tylko adresy stron YouTube wymagają dodatku `libreyolo[stream]`.

## Jak zachować tylko wybrane klasy?

Przekaż do `predict` parametr `classes` z wymaganymi indeksami klas, na przykład
`classes=[0, 2]`. `conf` ustawia próg pewności, domyślnie `0.25`, a `max_det`
ogranicza liczbę detekcji na obraz, domyślnie do `300`.

## Czy CLI używa flag, czy par key=value?

Klucza i wartości połączonych znakiem równości, dla każdego polecenia:

```bash
libreyolo predict model=yolo9-t source=my-image.jpg save=True
libreyolo train model=yolo9-t data=coco8.yaml epochs=50 imgsz=640
```

`model` przyjmuje ścieżkę lub krótką nazwę w formie `family-size`, opcjonalnie
z sufiksem zadania, a `libreyolo models` wymienia wszystkie prawidłowe wartości.
Polecenia diagnostyczne i inwentarza przyjmują również `--json`, które wyświetla
te same dane jako obiekt do odczytu maszynowego na stdout.

## Czy każdy model można wyeksportować do każdego formatu?

Nie. Pokrycie zależy od rodziny i zadania, nie jest jednolite, a każdy format
ma własny dodatek do zainstalowania. Każda strona modelu zawiera macierz eksportu
rodziny, a [sekcja eksportu](/docs/export) opisuje same formaty.

## Czym różnią się segment, semantic i panoptic?

To trzy oddzielne zadania. `segment` tworzy po jednej masce na wykryty obiekt.
`semantic` przypisuje klasę każdemu pikselowi i nie rozdziela niczego na
instancje. `panoptic` przypisuje każdemu pikselowi dokładnie jedną etykietę,
łącząc policzalne elementy thing z amorficznymi obszarami stuff. Zadania mają
inne dane referencyjne, inne pola wyniku i inne metryki, a rodzina obsługuje te,
które występują na jej liście zadań.

## Jak trenować na własnych klasach?

Utwórz plik YAML zbioru danych z `train`, `val` i `names`. Etykiety znajdują się
obok obrazów w równoległym drzewie `labels/`, po jednym pliku `.txt` na obraz,
ze znormalizowanymi współrzędnymi. `nc` jest opcjonalne i, jeśli występuje, musi
być zgodne z `names`. Najpierw uruchom `libreyolo doctor <data.yaml>`: polecenie
sprawdza problemy w zbiorze danych i kończy się kodem różnym od zera, gdy je
znajdzie, dzięki czemu może służyć jako bramka CI.

## Dlaczego podczas wczytywania pojawia się ostrzeżenie o metadanych?

Ponieważ checkpoint nie zawiera kompletnych metadanych v1.0. Wczytywanie jest
kontynuowane przez ścieżkę zgodności, a ostrzeżenie dokładnie wymienia brakujące
klucze. Uruchom `libreyolo metadata path=<file>`, aby zobaczyć zawartość,
i sprawdź [checkpointy i wagi](/docs/weights), aby poznać wymagania schematu.

## Import przestał działać po aktualizacji. Co się zmieniło?

Dla spójności zmieniono nazwy dwóch klas: `LibreYOLORTDETR` zmieniło się na
`LibreRTDETR`, a `LibreYOLORFDETR` na `LibreRFDETR`. Stare nazwy nadal są
rozwiązywane i emitują `DeprecationWarning` wskazujące nową nazwę, więc
istniejący kod działa podczas aktualizowania.
