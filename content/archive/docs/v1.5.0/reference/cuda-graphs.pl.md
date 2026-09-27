---
title: Grafy CUDA
seo_title: Macierz obsługi grafów CUDA w LibreYOLO
description: >-
  Które rodziny przechwytują przebieg w przód podczas predykcji oraz przebieg w
  przód i wstecz podczas trenowania, jakie wartości są gwarantowane, gdzie
  przechwytywanie jest dzielone i dlaczego nieobsługiwana rodzina zgłasza błąd.
lead: >-
  Graf CUDA rejestruje jedno wykonanie stałej sekwencji kerneli i odtwarza je
  jako pojedyncze uruchomienie. LibreYOLO przechwytuje inferencję dla 39
  zweryfikowanych rodzin i trenowanie dla 24, zawsze dla konkretnej rodziny,
  zawsze po sprawdzeniu zgodności bitowej i nigdy z cichym powrotem do zwykłego
  wykonania.
keywords:
  - graf CUDA LibreYOLO
  - cuda_graph=True
  - obsługa CUDA graph
  - trenowanie torch CUDA graph
  - capture_error_mode thread_local
  - zgodność bitowa CUDA graph
last_verified: 1.5.0
verification: >-
  Listę rodzin dla inferencji wyprowadzono z macierzy CAPTURABLE w
  tests/e2e/test_cuda_graph_families.py w wersji 1.5.0. Lista rodzin dla
  trenowania, klasy zgodności i czasy pochodzą z docs/training_cuda_graphs.md.
  API oraz NotImplementedError pochodzą z BaseModel._require_cuda_graph_support,
  cuda_graph_scope i capture_graph w libreyolo/models/base/model.py, wraz ze
  zmienną klasową SUPPORTS_CUDA_GRAPH. Podziały w punktach łączenia odczytano z
  nadpisań _get_graph_runner w rodzinach depth_anything3, birefnet, ppocr, sam i
  sensenova oraz z libreyolo/models/base/detr_cuda_graph.py. capture_error_mode
  pochodzi z libreyolo/models/base/cuda_graph.py i
  libreyolo/training/cuda_graph.py. Zachowanie awaryjne trenowania pochodzi z
  libreyolo/training/trainer.py, a flaga --cuda-graph z
  libreyolo/cli/commands/train.py.
meta:
  - label: Rodziny dla inferencji
    value: '39'
  - label: Rodziny dla trenowania
    value: '24'
  - label: Flaga inferencji
    value: predict(cuda_graph=True)
    mono: true
  - label: Flaga trenowania
    value: train(cuda_graph=True)
    mono: true
snippets:
  usage:
    - label: Predykcja
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # True przechwytuje graf przy pierwszym użyciu każdego kształtu wejścia.

        # "auto" czeka na powtórzenie kształtu przed poniesieniem kosztu
        przechwycenia.

        result = model(SAMPLE_IMAGE, cuda_graph=True)
    - label: Trenowanie
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: Trenowanie z CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=my-dataset.yaml \
          epochs=100 --cuda-graph
source_hash: 67c46199939278f2
---

## Co jest przechwytywane

Graf rejestruje stałą sekwencję kerneli oraz adresy pamięci, z których czytają
i do których zapisują. Nie rejestruje wartości, kształtów ani przepływu
sterowania. Odtworzenie wymaga jednego uruchomienia zamiast setek, dlatego zysk
jest największy w małych sieciach przy małych rozmiarach batcha, gdzie krok
jest zdominowany przez narzut uruchamiania, a nie obliczenia.

Te dwa punkty wejścia przechwytują różny zakres pracy.

| | Wewnątrz grafu | Wykonanie eager |
|---|---|---|
| Inferencja | Przebieg sieci w przód, `model._forward(x)` | Przetwarzanie wstępne, NMS, całe przetwarzanie końcowe |
| Trenowanie | Przebieg sieci w przód i wstecz | Funkcja straty, krok optymalizatora, przycinanie gradientów, EMA, harmonogram LR |

Ani NMS, ani funkcja straty detekcji nie nadają się do przechwycenia. Obie
operacje wybierają dane maskami logicznymi, wykonują dopasowanie węgierskie lub
przydzielanie i rozgałęziają się zależnie od wyniku, czyli robią dokładnie to,
czego graf nie może zarejestrować. Pozostawienie ich poza grafem zapewnia
bezpieczeństwo przechwytywania, a nie jest ograniczeniem wymagającym obejścia.

<code-tabs name="usage" />

Podczas predykcji `cuda_graph` przyjmuje trzy wartości. Domyślna to `False`.
`True` przechwytuje graf przy pierwszym napotkaniu każdego kształtu wejścia.
`"auto"` czeka, aż kształt się powtórzy, dzięki czemu jednorazowe zadania
i zadania o zmiennych kształtach nie ponoszą kosztu przechwycenia, którego nie
wykorzystają ponownie. `capture_graph(imgsz=None, batch=1, dtype=None)` przenosi
ten koszt poza pierwsze żądanie, `graph_info()` raportuje przechwycone grafy
i liczbę odtworzeń, a `release_graphs()` je zwalnia.

Podczas trenowania flaga jest zwykłą wartością logiczną, a w CLI ma postać
`--cuda-graph`. Powiązane ustawienia opisano na stronach
[wydajności predykcji](/docs/predict/performance) i
[wydajności trenowania](/docs/train/performance).

## Obsługa inferencji

Obsługa jest określana dla każdej rodziny zmienną klasową
`SUPPORTS_CUDA_GRAPH`. Rodzina jest oznaczana jako obsługiwana dopiero wtedy,
gdy przechwycenie i odtworzenie daje wynik identyczny bitowo z wynikami dwóch
wejść próbnych pochodzących z różnych rozkładów. Wspólna macierz zgodności
obejmuje 39 rodzin w dziewięciu zadaniach.

| Zadanie | Rodziny |
|---|---|
| detect | yolo1, yolo2, yolo3, yolo4, yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, rfdetr, ec |
| segment | dfine, rtmdet, rfdetr, ec |
| pose | ec, yolonas, rfdetr |
| point | fomo |
| classify | resnet, convnext, mobilenetv4, efficientnetv2, clip, dinov2, siglip2 |
| semantic | eomt, dinov2, segformer, pidnet, lingbotvision |
| depth | depth_anything, depth_anything3, zipdepth |
| restore | nafnet, realesrgan, swinir |
| matte | birefnet |

Kilka rodzin występuje przy więcej niż jednym zadaniu, więc macierz wykonuje
więcej wierszy, niż obejmuje odrębnych rodzin. Trzy kolejne rodziny przechwytują
grafy przez właściwe dla siebie ścieżki kodu z osobnymi testami, a nie przez
wspólną macierz, i nie należą do tych 39: PP-OCR, SAM i SenseNova.

Weryfikacja jest bitowa, a nie przybliżona. Wcześniejsza wersja protokołu
oceniała zgodność na podstawie względnej wielkości i błędnie obniżała status
trzech prawidłowo działających rodzin: YOLOX, EfficientNetV2 i YOLOv7. Ich
różnica między wykonaniem eager a grafem wynosi około 1e-7, mimo że dla istotnej
próby wyniki są identyczne bitowo.

## Obsługa trenowania

W tym wydaniu obsługa przechwytywania podczas trenowania wzrosła z dwóch do
24 rodzin w pięciu zadaniach.

| Zadanie | Rodziny |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Wszystkie pozostałe przypadki są trenowane w trybie eager: inne zadania tych
samych rodzin, rodziny spoza listy, przebiegi rozproszone i przebiegi destylacji.
Przechwytywanie jest również pomijane, gdy kształt jest jeszcze nowy, ponieważ
ścieżka trenowania czeka na trzykrotne powtórzenie kształtu wejścia. Oznacza to,
że przy `multi_scale=True` przechwycenie może nigdy nie nastąpić.

## Dwie różne odpowiedzi dla nieobsługiwanej rodziny

Ścieżka inferencji zgłasza wyjątek. Wywołanie `predict(cuda_graph=True)` dla
rodziny, która nie zadeklarowała obsługi, zgłasza `NotImplementedError` z nazwą
rodziny, zamiast wykonać operację w trybie eager i wywołać błędne przekonanie
o uzyskanym przyspieszeniu. Przyczyną jest to, że nieprawidłowe przechwycenie
nie kończy się wyraźnym błędem: odtworzenie przebiegu w przód wykonującego
nieprzechwytywalną operację po cichu zwraca błędne wartości. Dlatego obsługa
musi być jawną deklaracją konkretnej rodziny, a nie próbą z zachowaniem awaryjnym.

Ścieżka trenowania zapisuje komunikat. `train(cuda_graph=True)` można zawsze
bezpiecznie przekazać, a rodzina, zadanie lub konfiguracja, których nie można
przechwycić, zapisuje jeden wiersz i trenuje bez zmian w trybie eager. Jeśli
przechwytywanie nie powiedzie się w trakcie przebiegu, pozostała część również
przechodzi do trybu eager zamiast przerywać pracę. Ta asymetria jest celowa:
wywołanie predykcji można poprawić w miejscu jego użycia, natomiast przebieg
trenowania nie powinien zakończyć się po sześciu godzinach z powodu opcjonalnej
optymalizacji.

## Podział w punktach łączenia

Niektórych rodzin nie można przechwycić w całości, ponieważ jeden etap wykonuje
operację, której graf rzeczywiście nie może zarejestrować. Zamiast rezygnować
z rodziny, przechwytywanie dzieli się w zweryfikowanym punkcie łączenia: część
możliwa do przechwycenia jest odtwarzana, reszta działa w trybie eager, a wynik
połączony jest taki sam jak przy wykonaniu wszystkiego w trybie eager.

| Rodzina | Przechwycone | Tryb eager i przyczyna |
|---|---|---|
| Depth Anything 3 | Sieć | Etap nieba, czyli praca widoczna dla hosta po przebiegu w przód |
| BiRefNet | Enkoder, `forward_enc` | Dekoder, którego `deform_conv2d` po przechwyceniu odtwarza inny wynik |
| PP-OCR | Etap detekcji, `forward_det` | Rozpoznawanie, ponieważ szerokość wyciętych obszarów różni się między wierszami |
| SAM | Enkoder obrazu | Ścieżka podpowiedzi, uruchamiana wiele razy dla każdego kodowania |
| SenseNova | Część wizyjna | Generowanie autoregresyjne z pamięcią podręczną KV rosnącą w każdym kroku |
| Detektory enkoder-dekoder | Backbone i enkoder | Dekoder i kryterium węgierskie |

Podział BiRefNet warto przeczytać dwukrotnie: nieprawidłowe zachowanie
`deform_conv2d` podczas przechwytywania odtwarza się w samodzielnym wywołaniu
poza dowolnym modelem. Odrzucono zastąpienie tej operacji odpowiednikiem
w czystym PyTorch, ponieważ zmieniłoby to również predykcje eager, a wartości
eager są kontraktem.

Przypadek enkoder-dekoder obejmuje D-FINE, DEIM, DEIMv2, RT-DETR, RT-DETRv2,
RT-DETRv4 i EC. Ich dekoder buduje kontrastowe zapytania odszumiające na
podstawie danych referencyjnych (ground truth), a liczba tych zapytań wynika
z największej liczby obiektów referencyjnych w batchu. Z tego powodu liczba
tokenów dekodera zmienia się między batchami. To jedyna rzecz, której graf nie
może tolerować. Backbone wraz z enkoderem zajmuje w tych rodzinach mniej więcej
od jednej piątej do jednej czwartej kroku, dlatego znajdują się na dole tabeli
przyspieszeń.

PP-OCR przechwytuje po jednym grafie dla każdego kształtu wejścia detekcji,
z ograniczeniem wynikającym z pojemności pamięci podręcznej modułu wykonawczego,
i zwraca wynik eager, gdy żaden zakres przechwytywania nie jest aktywny.

## Wartości numeryczne

Większość rodzin daje wyniki identyczne bitowo, a tam, gdzie tak nie jest,
przyczyna zostaje nazwana, a nie pominięta ogólnikiem. W kroku zerowym
trenowania funkcja straty jest identyczna bitowo dla wszystkich 24 rodzin
i żaden bufor BatchNorm się nie różni. Kategorie rozdziela dopiero porównanie
gradientów.

| Klasa | Rodziny | Znaczenie |
|---|---|---|
| Dokładna | Większość z 24 | Każdy gradient jest identyczny bitowo |
| 1 ULP | fomo, lingbotvision | Ostatni bit float32, względnie około 1e-7, wynikający z innej kolejności sumowania |
| Szum eager | Linia DETR | Wynik grafu różni się od eager nie bardziej niż dwa przebiegi eager między sobą |
| Zaokrąglenie zmiennoprzecinkowe | rtmdet | 137 ze 139 gradientów jest identycznych bitowo, dwa różnią się o około 3e-4 |
| Własny strumień RNG | segformer | Głębokość stochastyczna znajduje się w przechwyconym obszarze |

Klasę szumu eager trzeba interpretować poprawnie. W tych rodzinach dwa
przebiegi eager z tym samym ziarnem już dają różne wyniki, więc identyczność
bitowa nie jest poprzeczką, której przebieg grafowy nie pokonał. Nie pokonuje
jej nic. Zjawisko występuje szerzej przy `amp=False`, gdzie zmierzona względna
niedeterministyczność gradientu wag fp32 na poziomie 3.2e-7 narasta: dwa
przebiegi YOLOv9-t w trybie eager z tym samym ziarnem rozchodzą się o 36 procent
po 20 krokach, a wyłączenie TF32 tego nie naprawia.

## Pamięć przypięta

Przechwytywanie działa z `capture_error_mode="thread_local"`. W domyślnym trybie
PyTorch `"global"` wątek pamięci przypiętej DataLoader, przygotowujący kolejny
batch, wywołuje `cudaHostAlloc`. To jednocześnie unieważnia trwające
przechwytywanie i zostaje przez nie uszkodzone, więc przebieg kończy się przy
pobraniu następnego batcha błędem zgłoszonym wewnątrz wątku pamięci przypiętej.
Takie zestawienie zaobserwowano dwukrotnie w rzeczywistym trenowaniu przed
zdiagnozowaniem problemu.

Tryb lokalny dla wątku ogranicza tylko wątek przechwytujący. Wątek przypinania
nigdy nie dotyka przechwytywanego strumienia, więc żadna jego operacja i tak nie
należy do grafu. Trenowanie idzie dalej i tymczasowo podstawia podklasę
`torch.cuda.CUDAGraph`, która wymusza ten tryb, ponieważ
`make_graphed_callables` nie udostępnia odpowiedniego argumentu. Operacja jest
wykonywana pod blokadą, aby dwa równoczesne przechwycenia nie pozostawiły
podstawienia włączonego.

## Opłacalność

Pomiary wykonano na RTX 5070 Ti przy AMP, z jednym procesem na wariant,
odtwarzając jeden rzeczywisty batch, aby wyłączyć dataloader z pomiaru. Podano
najszybszy z 24 kroków po rozgrzewce. Detekcja działała przy 640 px,
a klasyfikacja przy 224 px.

| Rodzina | Batch | Przyspieszenie |
|---|---:|---:|
| FOMO s | 16 | 3.63x |
| MobileNetV4 s | 16 | 2.74x |
| EfficientNetV2 b0 | 16 | 2.44x |
| YOLOv9-t | 8 | 1.99x |
| YOLOv9 e2e | 8 | 1.76x |
| YOLOv9 p2 | 8 | 1.49x |
| Wszystkie pozostałe | różne | od 1.04x do 1.26x |

Zysk całego przebiegu jest mniejszy, ponieważ graf nie może przyspieszyć
dataloadera ani walidacji. Dostrajanie YOLOv9-t przez 20 epok na 406 obrazach
skróciło się z 428.4 s do 367.7 s, co daje przyspieszenie 1.16x od początku do
końca, z identycznym mAP50-95 równym 0.6394 w obu wariantach i identycznymi
funkcjami straty w każdej epoce.

Górną granicę wyznacza udział sieci w kroku. Na tym samym sprzęcie przy 640 px
i batchu 8 wynosi on 84 procent dla YOLOv9-t, ale tylko 26 procent dla RTMDet-t,
który większość kroku spędza w module przydzielania etykiet. Narzut uruchamiania
jest największy w Windows, dlatego zyski w Linux wynoszą około jednej trzeciej
do połowy wartości z tej tabeli, a przebieg ograniczony przez dataloader nie
przynosi żadnej zmiany czasu całkowitego. Szczytowe użycie pamięci zmienia się
od wartości niższej o 5 procent do wyższej o 19 procent.

## Ograniczenia

Graf rejestruje adresy, a nie wartości, więc każda operacja przenosząca
parametry go usuwa. Zmiana urządzenia przez `predict(device=...)`, kwantyzacja
i dekwantyzacja unieważniają wszystkie przechwycone grafy.

Rozmiar batcha ma większe znaczenie niż rodzina: RT-DETR-r18 przyspiesza 1.19x
dla batcha 2 i 1.04x dla batcha 8, ponieważ duży batch jest ograniczony mocą
obliczeniową i ma mniej narzutu uruchamiania do usunięcia.

Zestaw testów zgodności inferencji działał bez zainstalowanego opcjonalnego
pakietu `kernels`, dlatego nie obejmuje bezpieczeństwa przechwytywania przy
aktywnych skompilowanych kernelach Hub. Aby wyłączyć je podczas izolowania
problemu z przechwytywaniem, ustaw `LIBREYOLO_HUB_KERNELS=0`. Zobacz stronę
[kernele](/docs/reference/kernels).
