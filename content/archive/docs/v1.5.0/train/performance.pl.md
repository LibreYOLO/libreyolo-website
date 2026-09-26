---
title: Wydajność trenowania
seo_title: 'Szybsze trenowanie: grafy CUDA, AMP i profiler'
description: >-
  Przyspiesz trenowanie: przechwyć krok w grafach CUDA, wybierz typ danych AMP i
  użyj wbudowanego profilera, aby ustalić, na co rzeczywiście poświęcany jest
  czas.
lead: >-
  Na szybkość kroku trenowania wpływają trzy mechanizmy: mieszana precyzja,
  przechwytywanie przejścia sieci w przód i wstecz w grafie CUDA oraz
  rozwiązania wynikające z rzeczywistego wąskiego gardła wskazanego przez
  profiler.
keywords:
  - grafy cuda trenowanie
  - przyspieszenie trenowania modelu
  - trening mixed precision
  - trenowanie bfloat16
  - profiler pytorch
  - wąskie gardło dataloadera
  - narzut uruchamiania kerneli
  - wykorzystanie gpu podczas trenowania
last_verified: 1.5.0
snippets:
  profile:
    - label: Profilowanie i kontynuowanie trenowania
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Profiluje krótki przedział rzeczywistych kroków, wyświetla werdykt, a
        następnie

        # kontynuuje trenowanie po usunięciu hooków.

        model.train(data="my-dataset.yaml", epochs=100, profile=True)
    - label: 'Tylko pomiar, potem zatrzymanie'
      language: bash
      code: |
        # Ustawia no_aug_epochs=0 i wykonuje tylko tyle epok, aby wypełnić okno.
        libreyolo profile run coco128 --weights LibreYOLO9s.pt --size s
    - label: Analiza szczegółów wyniku
      language: bash
      code: |
        libreyolo profile summary runs/profile/prof/profile.json
        libreyolo profile phases runs/profile/prof/profile.json
        libreyolo profile kernels runs/profile/prof/profile.json --top 10
  graph:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 cuda_graph=true
  amp:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", amp=True, amp_dtype="bfloat16")
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          amp_dtype=bfloat16
source_hash: ee5bb727065b6099
---

## Pomiar przed wprowadzeniem zmian

Trzy opisane niżej mechanizmy rozwiązują różne problemy, a zastosowanie
niewłaściwego niczego nie zmieni. Profiler wskazuje, który problem występuje.

<code-tabs name="profile" />

`profile=True` mierzy przedział rzeczywistych kroków trenowania, domyślnie
odrzuca pięć, a następnie mierzy dwadzieścia, wyświetla raport, zapisuje artefakty
i kontynuuje trenowanie po usunięciu hooków. Gdy jest wyłączone, nic nie
kosztuje, a podczas trenowania rozproszonego jest ignorowane.

Raport kończy się jednym z czterech werdyktów:

| Werdykt | Znaczenie | Mechanizmy |
|---|---|---|
| `dataloader` | GPU czeka na dane wejściowe | więcej `workers`, `cache="ram"` lub `"disk"`, lżejsza augmentacja, większy batch |
| `host / launch` | dane trafiają do GPU zbyt wolno, wiele małych kerneli | większy batch, grafy CUDA, mniej synchronizacji hosta w każdym kroku |
| `compute` | GPU jest w pełni wykorzystany | AMP lub bfloat16 albo pozostawienie bez zmian |
| `memory-pressure` | przeciążenie alokatora, VRAM na granicy pojemności | mniejszy batch; wartości wykorzystania są tu niewiarygodne |

Wartość wykorzystania to czas zajętości kerneli podzielony przez czas kroku bez
synchronizacji. Okno jest celowo podzielone. Pierwsza połowa działa bez
dodatkowej synchronizacji, aby werdykt odzwierciedlał rzeczywiste nakładanie się
operacji, a tylko druga połowa ogranicza każdą fazę synchronizacją w celu
przypisania czasu GPU. Synchronizacja każdej fazy daje procesom roboczym
dataloadera zapas czasu i ukrywa niedobór danych, dlatego wartości udziałów nigdy
nie służą do wyboru werdyktu.

W katalogu uruchomienia pojawiają się cztery pliki: `timeline.html`, który sam
otwiera się w przeglądarce, `profile_trace.json` dla Perfetto lub Nsight,
`profile_summary.json` oraz `profile.json`, samodzielny plik przeznaczony do
kopiowania i przekazywania podpoleceniom `libreyolo profile`.

Warto znać dwie cechy `profile run`. Ustawia `no_aug_epochs=0`, ponieważ profiler
mierzy epokę 0, a krótkie trenowanie z domyślnym `no_aug_epochs` profilowałoby
lżejszy dataloader bez augmentacji zamiast tego rzeczywiście używanego podczas
trenowania. Ponadto `--repeat N` podaje średnią i odchylenie standardowe. Ma to
znaczenie, ponieważ krok ograniczony narzutem uruchamiania jest na tyle zmienny,
że pojedynczy pomiar wprowadza w błąd. Polecenie zapisuje katalogi kolejnych prób
`prof_1`, `prof_2` i tak dalej oraz zbiorczy plik `profile_repeat.json`.

## Mieszana precyzja

`amp=True` jest ustawieniem domyślnym dla większości rodzin i wykonuje przejście
w przód z użyciem mechanizmu autocast CUDA. `amp_dtype` wybiera `float16` albo
`bfloat16`.

<code-tabs name="amp" />

Float16 wymaga dynamicznego skalowania funkcji straty i otrzymuje aktywny skaler
gradientów. Szerszy zakres wykładnika bfloat16 tego nie wymaga, więc jego skaler
jest wyłączony. Cztery rodziny są dostarczane z `amp=False`: D-FINE, DEIM,
YOLO-NAS i FOMO. Ustawienie DEIM jest dziedziczone również przez RT-DETRv4.
D-FINE podaje przyczynę: jego dekoder ogranicza aktywacje do 65504, największej
skończonej wartości float16.

Semantykę argumentu, w tym zachowanie żądania bfloat16 na sprzęcie bez obsługi
bfloat16, opisano w sekcji [Hiperparametry](/docs/train/hyperparameters).

## Grafy CUDA

`cuda_graph=True` przechwytuje przejście sieci w przód i wstecz podczas
trenowania w grafie CUDA, usuwając narzut uruchamiania kerneli w każdym kroku.

<code-tabs name="graph" />

Flagę można zawsze bezpiecznie przekazać. Rodzina, zadanie lub konfiguracja,
których nie można przechwycić, zapisuje jeden komunikat i bez zmian trenuje w
trybie eager.

Przechwytywana jest tylko sieć. Funkcja straty celowo pozostaje w trybie eager,
ponieważ funkcje straty detekcji wybierają elementy za pomocą masek logicznych,
wykonują dopasowanie algorytmem węgierskim i rozgałęziają się zależnie od wyników
przypisania. Graf nie może zarejestrować żadnej z tych operacji. Krok
optymalizatora, przycinanie gradientów, aktualizacja EMA i harmonogram
współczynnika uczenia również pozostają w trybie eager.

Ogranicza to zysk zależnie od tego, jaką część kroku stanowi sieć, a udział ten
znacznie różni się między modelami. W pomiarach na RTX 5070 Ti przy 640 px i
batchu 8 sieć stanowi 84 procent kroku YOLOv9-t, 44 procent kroku YOLOv7-b,
31 procent kroku YOLOX-t i 26 procent kroku RTMDet-t. Dwa ostatnie modele
spędzają większość kroku w swoich algorytmach przypisywania etykiet, więc
przechwytywanie sieci pomaga im najmniej.

### Możliwy zysk

Warunki dla każdej poniższej wartości: RTX 5070 Ti, Windows, AMP, po jednym
procesie na wariant ze wspólnego zapisanego stanu i powtarzanie jednego
rzeczywistego batcha, aby wykluczyć dataloader z pomiaru. Podano najszybszy z 24
kroków po rozgrzewce. Detekcja przy 640 px, klasyfikacja przy 224 px. Rozmiar
batcha jest podany osobno w każdym wierszu.

| Rodzina | Rozmiar | Batch | Eager | Graf | Przyspieszenie |
|---|---|---:|---:|---:|---:|
| FOMO | s | 16 | 7.0 ms | 1.9 ms | 3.63x |
| MobileNetV4 | s | 16 | 14.5 ms | 5.3 ms | 2.74x |
| EfficientNetV2 | b0 | 16 | 29.0 ms | 11.9 ms | 2.44x |
| YOLOv9 | t | 8 | 93.6 ms | 47.0 ms | 1.99x |
| NAFNet | s | 8 | 132.5 ms | 105.5 ms | 1.26x |
| PicoDet | s | 8 | 145.0 ms | 118.7 ms | 1.22x |
| D-FINE | n | 4 | 185.3 ms | 159.2 ms | 1.16x |
| RF-DETR | n | 4 | 276.3 ms | 239.8 ms | 1.15x |
| YOLOX | t | 8 | 102.2 ms | 90.5 ms | 1.13x |
| RTMDet | t | 8 | 149.7 ms | 136.2 ms | 1.10x |
| YOLOv7 | b | 4 | 102.5 ms | 98.0 ms | 1.05x |

Wartości te izolują krok GPU. Pełne dostrajanie wymaga też czasu dataloadera i
walidacji. Dla YOLOv9-t na zbiorze detekcyjnym zawierającym 406 obrazów, przez 20
epok, z batchem 8, rozmiarem 640 px i 4 procesami roboczymi dataloadera na tej
samej maszynie, całkowity czas wyniósł 428.4 s w trybie eager i 367.7 s z grafem.
Oznacza to zysk 1.16x, a mAP50-95 wyniósł 0.6394 w obu wariantach.

Na te wartości wpływają trzy czynniki. Małe batche są ograniczone narzutem
uruchamiania, a duże mocą obliczeniową, dlatego RT-DETR-r18 zyskuje 1.19x przy
batchu 2 i 1.04x przy batchu 8. Narzut uruchamiania jest największy w systemie
Windows, a zyski w systemie Linux wynoszą mniej więcej od jednej trzeciej do
połowy wartości z tabeli. Trenowanie ograniczone dataloaderem nie wykazuje żadnej
zmiany całkowitego czasu, dlatego najpierw należy użyć profilera.

Przechwytywanie działa tak samo przy `amp=False`, ale kernele fp32 pracują
dłużej, więc krok jest w mniejszym stopniu ograniczony narzutem uruchamiania i
większość rodzin zyskuje mniej. Na tym samym sprzęcie MobileNetV4-s przy batchu
16 przechodzi z 2.74x przy AMP do 3.61x przy fp32, natomiast YOLOv9-t przy batchu
8 z 1.99x do 1.69x, a RT-DETR-r18 przy batchu 4 z 1.12x do 0.99x.

### Zastosowanie przechwytywania

| Zadanie | Rodziny |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Wszystkie pozostałe przypadki wracają do trybu eager z jednym komunikatem:
inne zadania w tych rodzinach, rodziny niewymienione na liście, trenowanie
rozproszone i trenowanie z destylacją. Błąd przechwytywania w czasie działania
również przełącza pozostałą część trenowania na tryb eager, zamiast je przerwać.

W detektorach z enkoderem i dekoderem, czyli D-FINE, DEIM, DEIMv2, RT-DETR v1,
v2 i v4 oraz EC, przechwytywane są tylko backbone i enkoder. Dekoder odczytuje
dane referencyjne (ground truth), aby tworzyć zapytania kontrastowego odszumiania,
a liczba tych zapytań zależy od największej liczby obiektów referencyjnych w
batchu. Ich liczba tokenów zmienia się więc między batchami.

### Kształty

Graf jest prawidłowy wyłącznie dla dokładnego kształtu danych wejściowych, z
którym został przechwycony. Trener zlicza kształty batchy i wykonuje
przechwytywanie, gdy dany kształt powtórzy się trzy razy. Batche o każdym innym
kształcie działają w trybie eager. Dotyczy to batchy wieloskalowych i ostatniego,
niepełnego batcha epoki.

To pułapka w przypadku rodzin DETR, które domyślnie zmieniają rozmiar każdego
batcha. Przy `multi_scale=True` podczas krótkiego trenowania jeden kształt może
nigdy nie wystąpić wystarczająco często, aby w ogóle doszło do przechwycenia.
Jeśli celem jest przyspieszenie, przekaż `multi_scale=False`.

YOLOX zmienia obliczenia przechwyconego obszaru w trakcie trenowania, włączając
gałąź regresji L1, gdy mozaika kończy się przy `no_aug_epochs`. Trener unieważnia
wtedy przechwycenie i wykonuje je ponownie, gdy nowy kształt się ustabilizuje.

### Wartości numeryczne i pamięć

Przy AMP większość rodzin odtwarza trajektorię funkcji straty z trybu eager bit
po bicie. FOMO i LingBot-Vision różnią się ostatnim bitem float32 z powodu innej
kolejności sumowania. Detektory z uwagą deformowalną, czyli D-FINE, DEIM, DEIMv2,
RT-DETR, RF-DETR i EC, także nie odtwarzają własnych przebiegów w trybie eager,
ponieważ propagacja wsteczna akumuluje wyniki operacjami atomowymi, a sploty TF32
wybierają kolejność redukcji przy każdym uruchomieniu. Przebieg z grafem mieści
się w zakresie tej zmienności. RTMDet różni się względnie o około 3e-4 dla dwóch
ze 139 gradientów, ponieważ współdzieli sploty głowicy między poziomami piramidy,
a dwie ścieżki propagacji wstecznej sumują trzy składniki w innej kolejności.
SegFormer ma głębokość stochastyczną wewnątrz przechwyconego obszaru, dlatego
powtarzany graf korzysta z własnego strumienia losowego i jest statystycznie
równoważny trybowi eager, a nie identyczny z nim. Menedżer zapisuje o tym jeden
komunikat podczas przechwytywania.

Przy `amp=False` na tym sprzęcie nie można uzyskać identyczności bitowej w żadnym
przypadku, z przechwytywaniem ani bez niego. Dwa identyczne przebiegi YOLOv9-t w
trybie eager, zainicjalizowane tym samym ziarnem, różnią się względnie o 36
procent po 20 krokach, a przebiegi YOLOX-t o 2.6 procent. Wynika to z tego, że dla
niektórych kształtów splotów fp32 cuDNN wybiera niedeterministyczny algorytm
gradientu wag.

Przechwycony graf przypina statyczne bufory danych wejściowych, wyjściowych i
obszaru roboczego, dlatego szczytowe użycie VRAM rośnie mniej więcej o jeden
dodatkowy zestaw aktywacji. W powyższych rodzinach szczytowa alokacja zmieniła
się od -5 do +19 procent. Koszt względny jest największy dla małych modeli
klasyfikacyjnych, których aktywacje są z natury małe. Dla ResNet-18 przy 224 px i
batchu 16 użycie wzrosło z 0.48 GB w trybie eager do 0.57 GB z grafem. Jeśli
powoduje to przekroczenie limitu, zmniejsz batch lub pozostaw flagę wyłączoną.

## Powiązane

- [Hiperparametry](/docs/train/hyperparameters) dotyczące `batch`, `nbs`, `cache`
  i `workers`.
- [Trenowanie na wielu GPU](/docs/train/multi-gpu), gdzie ani grafy CUDA, ani
  profiler nie są dostępne.
- [Grafy CUDA](/docs/reference/cuda-graphs) opisujące połączoną macierz obsługi
  inferencji i trenowania, granice przechwytywanych obszarów oraz gwarancje
  numeryczne.
