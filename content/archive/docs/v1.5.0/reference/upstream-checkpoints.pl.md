---
title: Checkpointy ze źródeł nadrzędnych
seo_title: Wczytywanie checkpointów ze źródeł nadrzędnych w LibreYOLO
description: >-
  Jak automatyczna konwersja zmienia wydany checkpoint ze źródła nadrzędnego w
  checkpoint LibreYOLO v1.0: rozpakowywane układy, rodziny rozpoznające
  poszczególne formaty i granice działania.
lead: >-
  Rodziny LibreYOLO są przeniesione z projektów nadrzędnych, których wydane
  checkpointy są prawie gotowe do wczytania, ale nie zawierają metadanych
  LibreYOLO. Automatyczna konwersja rozpoznaje te pliki, opakowuje je w schemat
  v1.0 i zapisuje wynik obok źródła.
keywords:
  - automatyczna konwersja LibreYOLO
  - wczytywanie checkpointu upstream
  - convert_upstream_state_dict
  - wagi upstream LibreYOLO
  - konwersja checkpointu
last_verified: 1.5.0
verification: >-
  Zachowanie odczytano z libreyolo/models/autoconvert.py i
  BaseModel.convert_upstream_state_dict; mechanizmy rozpoznawania poszczególnych
  rodzin sprawdzono przez odczyt nadpisań convert_upstream_state_dict każdej
  rodziny, wszystko w wersji 1.5.0. Reguły COCO dla RF-DETR pochodzą z
  docs/checkpoint_schema.md.
snippets:
  usage:
    - label: Przekazanie pliku bezpośrednio do fabryki
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Rozpoznany plik ze źródła nadrzędnego jest konwertowany podczas
        wczytywania,

        # a przekonwertowany checkpoint zostaje zapisany obok niego.

        # model = LibreYOLO("yolov9-t-converted.pt")


        # Każdy checkpoint LibreYOLO jest wczytywany bez zmian.

        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.family, model.size, model.task, model.nb_classes)
source_hash: c6022771a2a207a1
---

## Co dzieje się podczas wczytywania

Gdy `LibreYOLO()` napotka plik `.pt`, który nie jest jeszcze kompletnym
checkpointem v1.0, wywołuje automatyczny konwerter, który:

1. rozpakowuje słownik tensorów z typowych układów źródeł nadrzędnych;
2. pyta każdą zarejestrowaną rodzinę, czy rozpoznaje układ, mapując ponownie klucze tam, gdzie nazewnictwo źródła nadrzędnego różni się od natywnego portu;
3. opakowuje zwycięski wynik w checkpoint ze ścisłymi metadanymi v1.0, odczytując rozmiar, zadanie i liczbę klas z samych tensorów, aby poprawnie konwertować dostrojone checkpointy;
4. zapisuje go obok źródła jako `<source>-<Prefix><size>[-task].pt` i zwraca tę ścieżkę, dzięki czemu fabryka wczytuje go zwykłym sposobem.

Kod wywołujący nie musi robić niczego. Plik, do którego nie zgłosi się żadna
rodzina, nie daje wyniku, a fabryka informuje, że nie mogła go wczytać.

<code-tabs name="usage" />

## Rozpakowywane układy

Słownik tensorów jest wyszukiwany w poniższej kolejności pierwszeństwa,
z EMA na początku. Każdy kandydat jest sprawdzany, dopóki nie zostanie
znaleziony taki, który rzeczywiście zawiera tensory. Pusty blok EMA lub blok
zawierający tylko metadane nie przesłania więc prawidłowych wag pod nim.

| Klucz | Uwaga |
|---|---|
| `ema.module` | Typowa otoczka EMA |
| `ema` | Starsze płaskie otoczki EMA przechowujące tensory bezpośrednio |
| `ema_state_dict` | Prefiks `module.` jest usuwany z wpisów |
| `params_ema` | |
| `params` | |
| `ema_net` | |
| `net` | |
| `model` | |
| `state_dict` | |
| Sam plik | Zwykły słownik stanu |

Każdy kandydat jest następnie zawężany do wpisów o wartościach tensorowych
i normalizowany. Początkowy prefiks `module.` lub `_orig_mod.` jest usuwany,
a ze słownika, którego wszystkie klucze zaczynają się od `model.model.`, ten
prefiks również jest usuwany.

## Rodziny i rozpoznawane układy

Rozpoznawanie jest metodą klasową właściwą dla rodziny. Domyślna implementacja
zgłasza układ, którego klucze już odpowiadają natywnemu portowi. Rodzina,
w której nazewnictwo kluczy źródła nadrzędnego się różni, nadpisuje metodę
mapowaniem i nie zwraca niczego dla nierozpoznanych układów.

Rodziny dostarczające mechanizm rozpoznawania z mapowaniem: `centernet`,
`deeplabv3`, `deformable_detr`, `dexined`, `moge2`, `picodet`, `rtdetr`,
`rtdetrv2`, `rtdetrv4`, `rtmdet`, `segformer`, `swin`, `teed`, `yolo7`,
`yolo9`, `yolo9_e2e`, `yolo9_p2`.

Rodziny całkowicie odmawiające automatycznej konwersji: `efficientdet`, `eomt`
i `pidnet` nie zwracają niczego z mechanizmu rozpoznawania, więc ich pliki ze
źródła nadrzędnego przechodzą przez skrypt konwersji. `l2cs` jest wykluczone
z ogólnego mechanizmu rozpoznawania, ponieważ obsługuje tylko inferencję i ma
wagi z ograniczeniami redystrybucji.

RF-DETR zachowuje własny mechanizm rozpoznawania, ponieważ potrzebuje całego
checkpointu, a nie tylko słownika tensorów, aby wykryć rozmiar i ponownie
zmapować klasy COCO. Jest rejestrowany tylko po zainstalowaniu opcjonalnych
zależności.

Każda inna zarejestrowana rodzina korzysta z zachowania domyślnego: zgłasza
plik, gdy jej własny loader już rozpoznaje te klucze.

## Wybór rodziny

Ten sam plik może zostać zgłoszony przez kilka rodzin, dlatego rozstrzyganie
odzwierciedla reguły dyspozytora fabryki.

Zgłoszenie podklasy wygrywa ze zgłoszeniem klasy bazowej. Kolejność rejestracji
odpowiada kolejności tworzenia klas, więc rodzina pochodna rejestruje się po
klasie bazowej, którą uszczegóławia, a jej pozytywne znaczniki nie mogą przegrać
z szerszym przepuszczaniem klasy bazowej.

Następnie decyduje kolejność w rejestrze, ponieważ koduje szczegółowość:
najwcześniejsze zgłoszenie jest najbardziej szczegółowym dopasowaniem.

Jedynym remisem, którego kolejność rejestru nie może rozstrzygnąć, jest DEIM
względem D-FINE, ponieważ ich klucze architektury są identyczne. Tylko w tym
przypadku rozstrzyga nazwa pliku, a plik bez żadnej wskazówki w nazwie jest
odrzucany zamiast zgadywany. Nigdzie indziej nazwa pliku nie jest celowo brana
pod uwagę, dlatego szerokie fałszywie dodatnie zgłoszenie nigdy nie może pokonać
bardziej szczegółowego tylko z powodu nazwy pliku.

## Bezpieczne wczytywanie

Pliki ze źródeł nadrzędnych są wczytywane przez unpickler tylko do wag. Niektóre
checkpointy treningowe ze źródeł nadrzędnych osadzają obiekty bibliotek, które
ten unpickler odrzuca. Obiekty te są metadanymi trenowania, a nie wagami, dlatego
każdy zablokowany element globalny jest ponownie próbowany z obojętną klasą
zastępczą, która spełnia wymagania unpicklera bez wykonywania czegokolwiek.
Przechwycona nazwa jest używana wyłącznie jako etykieta tekstowa. Nigdy nie jest
importowana, obliczana ani wywoływana.

Wrażliwe nazwy modułów są bezwzględnie odrzucane i nigdy nie zastępuje się ich
atrapami: `builtins`, `os`, `sys`, `posix`, `nt` i `subprocess`. Pętla ponownych
prób jest ograniczona do 32 podejść, więc plik przygotowany do wprowadzania
nieograniczonej serii odrębnych elementów globalnych kończy działanie
bezpiecznym błędem zamiast zapętleniem. Do przekonwertowanego checkpointu trafiają
wyłącznie tensory.

## Miejsce zapisu przekonwertowanego pliku

Wynik jest zapisywany obok źródła pod nazwą
`<source>-<Prefix><size>[-task].pt`. Zawsze jest zapisywany ponownie, a nie
używany z wcześniejszego przebiegu. Dzięki temu wielokrotne wczytania tego
samego źródła pozostają aktualne, a zarazem nie dochodzi do kolizji z oficjalnymi
wagami ani innym dostrojeniem tej samej rodziny, rozmiaru i zadania w tym samym
katalogu.

Gdy katalog źródłowy jest tylko do odczytu, konwersja przechodzi do nowego
prywatnego katalogu tymczasowego tworzonego osobno dla każdego wywołania,
a wiersz dziennika podaje używaną ścieżkę. Dopiero gdy to również się nie
powiedzie, konwersja jest porzucana z ostrzeżeniem.

## Istniejące checkpointy LibreYOLO

Plik zawierający znacznik właściwy dla LibreYOLO, `libreyolo_version` lub
`model_family`, należy do zwykłej ścieżki wczytywania i nie jest ponownie
konwertowany. Pominięcie dotyczy tylko zgłoszenia przepuszczającego, czyli
takiego, w którym zbiór kluczy nie uległ zmianie. Zgłoszenie, którego konwersja
zmieniła zbiór kluczy, jest dowodem obcego układu ze źródła nadrzędnego i jest
akceptowane nawet w oznaczonym pliku.

`schema_version` celowo nie jest traktowane jako znacznik, ponieważ inne
narzędzia trenowania i eksportu używają tej ogólnej nazwy. Znacznikami nie są
również `names`, `nc`, `size`, `task` ani `imgsz`, ponieważ dostrojenie ze źródła
nadrzędnego też może je zawierać. Obce dostrojenie zawierające tylko ogólny
klucz `names` nie jest więc oznaczone. Jego zgłoszenie z natywnymi kluczami
konwertuje się normalnie i wyprowadza liczbę klas z głowicy tensora, zamiast
błędnie wczytać model jako 80-klasowy.

## Metadane odczytywane przez konwerter

Nazwy klas są pobierane z klucza najwyższego poziomu `names` albo z
`class_names` wewnątrz bloku `args` lub `hyper_parameters`. Mapa nazw z kluczami
będącymi etykietami zamiast indeksów klas jest bezużyteczna i zostaje zastąpiona
wygenerowanymi wartościami domyślnymi. Lista nazw dłuższa od wykrytej liczby
klas jest przycinana, ponieważ indeksy spoza zakresu nie przeszłyby ścisłej
walidacji i po cichu przerwałyby konwersję.

Pole `args` ze źródła nadrzędnego jest przenoszone jako zwykłe metadane,
z odrzuceniem każdej wartości, która nie jest ciągiem, liczbą, wartością
logiczną, listą ani słownikiem. Dzięki temu nic niebezpiecznego nie trafia do
zapisanego pliku.

## Normalizacja COCO dla RF-DETR

Checkpointy RF-DETR ze źródła nadrzędnego udostępniają głowicę klasyfikacyjną
z 91 wyjściami, czyli 90 klas COCO i tło. Automatyczna konwersja normalizuje
RF-DETR dla COCO do konwencji COCO-80, stosując mapowanie w przetwarzaniu
końcowym.

Checkpoint jest traktowany jako COCO, gdy zawiera dokładnie 80 nazw, deklaruje
liczbę klas równą 80, ma wskazówkę zbioru danych `coco` albo w ogóle nie zawiera
metadanych klas ani zbioru danych. Ten ostatni przypadek jest ważny: zwykły
słownik stanu ze źródła nadrzędnego jest kanonicznym checkpointem wstępnie
wytrenowanym na COCO i jedynym rozpowszechnianym RF-DETR bez metadanych,
z 91 wyjściami.

Rzeczywisty niestandardowy RF-DETR z 90 klasami jest zachowywany jako model
90-klasowy. Rozpoznaje się go na podstawie listy nazw, jawnej liczby klas innej
niż 80 lub wskazówki zbioru danych innego niż COCO, więc awaryjna reguła dla
gołego checkpointu nie zostaje dla niego uruchomiona. Puste pola zastępcze są
ignorowane przy ustalaniu obecności wskazówki zbioru danych.

## Ograniczenia

Automatyczna konwersja rozpoznaje wydane układy ze źródeł nadrzędnych. Nie
przepisuje architektury ani nie umożliwia wczytania modelu, który nie został
przeniesiony. Gdy żadna rodzina nie zgłasza pliku, właściwym rozwiązaniem jest
skrypt konwersji, a nie argument fabryki. Repozytorium dostarcza
`weights/convert_*.py` dla wymagających go rodzin, w tym EoMT, PIDNet
i EfficientDet.

Konwersja nie wymyśla też metadanych, których nie może odczytać. Rozmiar,
zadanie i liczba klas pochodzą z tensorów. Nazwy pochodzą z pliku, gdy są
obecne, a w przeciwnym razie są generowane jako `class_i`.
