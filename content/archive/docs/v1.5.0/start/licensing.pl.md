---
title: Licencjonowanie
seo_title: 'Licencje LibreYOLO: kod i wagi'
description: >-
  Własny kod LibreYOLO jest objęty licencją MIT. Dołączony kod z projektów
  źródłowych i opublikowane checkpointy mają własne licencje, z których część
  wyklucza użycie komercyjne.
lead: >-
  LibreYOLO obejmuje trzy odrębnie licencjonowane elementy: własny kod, kod
  źródłowego projektu dołączony do rodziny modeli oraz wstępnie wytrenowane
  checkpointy. Często nie podlegają one tej samej licencji.
keywords:
  - licencja libreyolo
  - biblioteka wizji komputerowej mit
  - wagi modelu bez użycia komercyjnego
  - licencja checkpointu modelu
  - detekcja obiektów apache-2.0
last_verified: 1.5.0
source_hash: 83536fea4dc4eaec
---

## Własny kod LibreYOLO

Biblioteka jest objęta licencją MIT. Dotyczy to API Pythona, CLI, modułów
trenujących, walidatorów i eksporterów, modułów wczytujących zbiory danych oraz
skryptów konwersji w katalogu `weights/`. Można używać ich w produkcie
komercyjnym lub o zamkniętym kodzie źródłowym. W każdej rozpowszechnianej kopii
należy zachować informację o prawach autorskich i tekst licencji. Na tym kończą
się zobowiązania.

Udzielone prawa obejmują wyłącznie kod. Plik
[`LICENSE`](https://github.com/LibreYOLO/libreyolo/blob/release/LICENSE)
ujmuje to jasno:

> Licencje te są różne i nie wszystkie są liberalne: niektóre opublikowane wagi
> są przeznaczone wyłącznie do użytku niekomercyjnego lub podlegają innym
> ograniczeniom, a niniejsza licencja MIT ich nie obejmuje. Wybór modelu oznacza
> wybór jego licencji.

## Kod źródłowych projektów według rodziny

Większość rodzin to porty opublikowanych prac badawczych, a część bezpośrednio
dołącza kod źródłowego projektu. Dołączony plik zachowuje pierwotny nagłówek
praw autorskich i pierwotną licencję. Licencja MIT ich nie zastępuje, a
LibreYOLO nie udziela ponownie licencji na cudzą pracę. Najczęściej występują
licencje Apache-2.0 i BSD-3-Clause.

Licencja Apache-2.0 obejmuje linię DETR i znaczną część rozwiązań opartych na
transformerach: DETR od Meta AI (FAIR), Deformable DETR od SenseTime, LW-DETR
od Baidu, OV-DEIM autorstwa Leilei Wanga i współautorów, implementację SegFormer
przeniesioną do LibreYOLO z Hugging Face Transformers, PP-OCRv5 od autorów
PaddlePaddle, SwinIR z Computer Vision Lab na ETH Zurich oraz Depth Anything 3
od ByteDance Seed. Obejmuje także klasyfikatory wywodzące się z timm autorstwa
Rossa Wightmana i współtwórców timm, w tym ResNet, DeiT, EfficientNetV2,
MobileNetV4 i Swin. Nazwy ich modułów odpowiadają timm, dzięki czemu tensory
ImageNet są wczytywane bez zmian.

Licencja BSD-3-Clause obejmuje wszystko, co wywodzi się z torchvision: Faster
R-CNN, Mask R-CNN, FCOS, RetinaNet, SSD300, AlexNet, VGG, FCN i DeepLabv3.

Licencja MIT obejmuje mniejszą grupę, w tym NAFNet od Megvii, CenterNet od
Xingyi Zhou oraz YOLOv7 ponownie wydany przez jego autorów, Kin-Yiu Wonga i
Hao-Tang Tsuia, z MultimediaTechLab. Rodziny od YOLOv1 do YOLOv4 odtwarzają
architektury z projektu Darknet autorstwa Josepha Redmona, a w przypadku YOLOv4
również Alexeya Bochkovskiego. Darknet należy do domeny publicznej, więc nie
wiąże się z żadnymi zobowiązaniami.

Jedno dołączone poddrzewo nie jest objęte licencją open source. Rodzina DEIMv2
zawiera kod backbone DINOv3 od Meta Platforms objęty DINOv3 License Agreement,
niestandardową licencją spoza OSI. Rozpowszechnianie tego kodu wymaga dołączenia
kopii umowy. Umowa zabrania także wykorzystania w działaniach podlegających
ITAR, do celów wojskowych lub wojennych, w przemyśle jądrowym, szpiegostwie i
rozwoju broni. Warunki te wiążą wyłącznie to poddrzewo.

Dwa pliki w repozytorium przedstawiają pełny obraz.
[`NOTICE`](https://github.com/LibreYOLO/libreyolo/blob/release/NOTICE) zawiera
listę wszystkich dołączonych poddrzew innych podmiotów wraz ze ścieżką, plikiem
licencji i źródłem projektu.
[`THIRD_PARTY_NOTICES.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/THIRD_PARTY_NOTICES.txt)
zawiera listę projektów, z których wywodzi się LibreYOLO, oraz pełny tekst
każdej licencji.

## Wagi według checkpointu

Pakiet nie zawiera żadnego pliku wstępnie wytrenowanych wag. Opublikowane
checkpointy znajdują się na Hugging Face w ramach [organizacji
LibreYOLO](https://huggingface.co/LibreYOLO), a każde repozytorium ma własny
plik `LICENSE` i informacje o autorstwie odpowiadające projektowi, z którego
pochodzą wagi.

To repozytorium jest rozstrzygającym źródłem warunków. Nie jest nim ta strona,
strona modelu ani podsumowanie w drzewie kodu źródłowego. Informacje o nazwach
plików i miejscu ich pobierania zawiera strona [checkpointy i
wagi](/docs/weights).

Licencje różnią się między rodzinami, a także między plikami w obrębie jednej
rodziny. Oto dwa przykłady drugiego przypadku:

- Checkpointy YOLO9 COCO są objęte licencją MIT. Plik
  `LibreYOLO9P2s-visdrone.pt`, wytrenowany na VisDrone2019-DET, jest objęty
  licencją CC BY-NC-SA 3.0, która wyklucza użycie komercyjne.
- Checkpointy detekcji RF-DETR są objęte licencją Apache-2.0. Checkpointy
  obróconych ramek są objęte licencją CC BY 4.0, ponieważ dostrojono je na
  zbiorze danych Roboflow Universe opublikowanym na licencji CC BY 4.0, a wagi
  zachowują wynikający z tej licencji wymóg podania autorstwa zbioru danych.

Zakres między rodzinami jest jeszcze szerszy, a kilku opublikowanych
checkpointów nie można używać w produkcie komercyjnym:

- SegFormer najwyraźniej pokazuje rozdział tych dwóch warstw. Implementacja to
  port kodu Hugging Face Transformers objęty licencją Apache-2.0. Opublikowane
  checkpointy ADE20K przekonwertowano z wydania NVIDIA objętego NVIDIA Source
  Code License. Pozwala ona na redystrybucję, ale ogranicza użycie do badań lub
  oceny o charakterze niekomercyjnym i przenosi to ograniczenie na prace
  pochodne. Tych checkpointów nie obejmują liberalne warunki LibreYOLO.
- Checkpointy OV-DEIM są objęte licencją CC BY-NC 4.0, co potwierdził autor
  projektu źródłowego. Każda predykcja wczytuje również tekstową część modelu
  MobileCLIP-B(LT) firmy Apple, której licencja ogranicza użycie do badań. Jest
  to warunek bardziej restrykcyjny niż własna licencja checkpointu.
- Kod SenseNova-Vision jest objęty licencją Apache-2.0, a jego wagi licencją CC
  BY-NC 4.0. Moduł wczytujący wyświetla informację o użytku niekomercyjnym przed
  każdym automatycznym pobraniem.

Niektóre rodziny nie mają żadnego checkpointu hostowanego przez LibreYOLO, co
jest wskazane w wierszu Wagi na ich stronach. Dostęp do SAM 3 na Hugging Face
jest ograniczony własną licencją SAM firmy Meta, a plik pobiera się bezpośrednio
od Meta. Zasoby wydań MiDaS są pobierane z oficjalnych adresów URL i weryfikowane
za pomocą skrótu zamiast ponownego hostowania. Odnośnik do Dome-DETR prowadzi do
projektu źródłowego, ponieważ karta modelu nie podaje licencji w metadanych,
podczas gdy jej treść jednocześnie deklaruje Apache-2.0 i ogranicza użycie do
badań akademickich. Te informacje są sprzeczne. Architektury TEED i DexiNed są
objęte licencją MIT, ale checkpointy wydane przez autorów wytrenowano na BIPED,
którego warunki ograniczają użycie do celów niekomercyjnych. Dlatego LibreYOLO
ani ich nie dołącza, ani nie pobiera automatycznie.

Kilka checkpointów torchvision nie ma własnego pliku licencji. LibreYOLO
udostępnia ich kopie na licencji używanej przez projekt, który je wydał. Na
każdej karcie modelu zaznaczono, że podstawa licencyjna jest dorozumiana, a nie
udzielona osobno dla checkpointu, i powtórzono ostrzeżenie torchvision, że
warunki wstępnie wytrenowanego modelu mogą wynikać z danych treningowych.

## Znajdowanie warunków dla jednego modelu

Strona modelu zawiera w nagłówku wiersz **Licencje** w formacie `Kod X, wagi Y`,
który prowadzi do sekcji Licencjonowanie na tej stronie. Sekcja wymienia
oryginalną pracę i jej autorów, licencję projektu źródłowego, źródło projektu,
licencję kodu LibreYOLO, wagi oraz interpretację dozwolonego użycia. Tabela
Checkpointy na tej samej stronie ma kolumnę **Licencja wag** i po jednym wierszu
na każdy opublikowany plik, dzięki czemu rodzina o mieszanych warunkach pokazuje
je osobno dla każdego pliku.

Wszystkie te informacje są renderowane z tych samych danych, względem których
sprawdzana jest biblioteka. Dlatego ta strona nie powtarza ich w tabeli. Ręcznie
napisana macierz licencji staje się nieaktualna w ciągu jednego wydania, a błąd
w tej kwestii jest kosztowny.

W drzewie kodu źródłowego odpowiednikami są `NOTICE` dla dołączonego kodu,
`THIRD_PARTY_NOTICES.txt` dla projektów źródłowych i tekstów ich licencji oraz
[`weights/LICENSE_NOTICE.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/weights/LICENSE_NOTICE.txt)
z podsumowaniem opublikowanych checkpointów dla poszczególnych rodzin.

Następnie należy sprawdzić repozytorium Hugging Face dotyczące dokładnie tego
pliku, który ma zostać pobrany. Jest ono rozstrzygającym źródłem i może zmienić
się bez zmiany strony dokumentacji.

## Użycie komercyjne

Kod rzadko stanowi problem. Licencje MIT, Apache-2.0 i BSD-3-Clause zezwalają na
użycie komercyjne oraz w zamkniętym kodzie źródłowym. Każda z nich wymaga
zachowania tekstu licencji i informacji o autorstwie w rozpowszechnianych
kopiach, Apache-2.0 udziela również licencji patentowej, a żadna nie nakłada
warunków na własny kod aplikacji.

Checkpointy są częstszą przeszkodą dla produktów. Checkpoint przeznaczony do
użytku niekomercyjnego pozostaje niekomercyjny niezależnie od liberalnej
licencji otaczającego kodu, a konwersja pliku nie zmienia obowiązujących go
warunków. Wprost stwierdza to `weights/LICENSE_NOTICE.txt`. Artefakt ONNX lub
TensorRT zbudowany z checkpointu objętego ograniczeniami przejmuje te
ograniczenia.

Jeśli licencja przenosi ograniczenie na prace pochodne, tak jak NVIDIA Source
Code License, dostrajanie również go nie omija. Inaczej jest w przypadku
trenowania tej samej architektury od zera na danych, do których ma się
odpowiednie prawa. Kod ma liberalną licencję, więc samodzielnie wytrenowany
model należy do osoby, która go wytrenowała, a warunki wstępnie wytrenowanego
checkpointu nigdy nie mają zastosowania. Strona SegFormer wyjaśnia to w
odniesieniu do własnych wag. Należy przeczytać wiersz Interpretacja na stronie
każdej rodziny planowanej do wdrożenia.

Kwestię licencji należy rozstrzygnąć podczas wyboru modelu, a nie przed samym
wydaniem produktu. Trzeba też przeczytać warunki dotyczące faktycznie pobranego
pliku, ponieważ obok liberalnie licencjonowanego checkpointu tej samej rodziny
może znajdować się checkpoint objęty ograniczeniami.

## To nie jest porada prawna

Ta strona opisuje obowiązujące licencje. Jest opisem, a nie poradą prawną, i nie
stanowi żadnej gwarancji. Jeśli odpowiedź ma znaczenie komercyjne, należy
samodzielnie przeczytać licencje i zasięgnąć porady prawnej.
