---
title: Import istniejących wag
seo_title: Wczytywanie wag z projektów źródłowych w LibreYOLO
description: >-
  Wskaż LibreYOLO checkpoint z projektu źródłowego. Automatyczna konwersja
  opakuje go ponownie podczas wczytywania, zachowując liczbę i nazwy klas.
lead: >-
  LibreYOLO przenosi rodziny modeli z projektów źródłowych, dlatego wydane przez
  nie checkpointy są niemal gotowe do wczytania. Brakuje im tylko metadanych.
  Automatyczna konwersja uzupełnia je podczas wczytywania.
keywords:
  - libreyolo konwersja wag
  - wczytywanie checkpointu z innego projektu
  - migracja libreyolo
  - konwersja pth do libreyolo
  - automatyczna konwersja checkpointu
last_verified: 1.5.0
meta:
  - label: Punkt wejścia
    value: LibreYOLO("path/to/upstream.pth")
    mono: true
  - label: Zapisywany obok źródła jako
    value: '<source>-<Prefix><size>[-task].pt'
    mono: true
  - label: Skryptowe konwertery
    value: weights/ in the repository
    mono: true
snippets:
  convert:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Zastąp tę wartość ścieżką do posiadanego checkpointu. Rozpoznany układ

        # projektu źródłowego zostanie przekonwertowany w locie, zapisany obok

        # źródła, a następnie wczytany.

        model = LibreYOLO("path/to/upstream-checkpoint.pth")


        # Liczba i nazwy klas pochodzą z tensorów oraz własnych metadanych
        pliku,

        # dlatego dostrojony model zachowuje własny zestaw etykiet zamiast COCO.

        print(model.family, model.size, model.task, model.nb_classes)

        print(model.names)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=path/to/upstream-checkpoint.pth \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Sprawdzenie wyniku
      language: bash
      code: |
        # Przekonwertowany plik spełnia ten sam schemat co plik opublikowany.
        libreyolo metadata path=path/to/upstream-checkpoint-LibreYOLO9t.pt
source_hash: bf9d7c7d168fd2c0
---

Ta strona dotyczy checkpointów z innych projektów. W przypadku przenoszenia
własnego kodu ze starszej wersji LibreYOLO zobacz [aktualizację do
1.5.0](/docs/upgrade).

## Co się dzieje podczas wczytywania obcego pliku

Funkcja `LibreYOLO()` najpierw wczytuje każdy plik wag przez ograniczoną ścieżkę
obsługującą tylko wagi. Jeśli wynik zawiera kompletne metadane LibreYOLO, jest
używany bezpośrednio. W przeciwnym razie plik trafia do automatycznego konwertera
przed podjęciem jakiejkolwiek innej próby. Jeśli ograniczone wczytywanie od razu
zakończy się niepowodzeniem, co zdarza się, gdy checkpoint zawiera obiekt innej
firmy zapisany przez pickle, automatyczny konwerter używa modułu wczytującego,
który neutralizuje takie obiekty.

Automatyczna konwersja wykonuje cztery czynności. Rozpakowuje słownik tensorów z
układu używanego przez projekt źródłowy. Następnie pyta każdą zarejestrowaną
rodzinę, czy rozpoznaje otrzymane klucze, i mapuje nazwy, jeśli nazewnictwo
projektu źródłowego różni się od portu LibreYOLO. Opakowuje zwycięski wynik w
checkpoint zgodny ze schematem metadanych v1.0, odczytując rozmiar, zadanie i
liczbę klas z samych tensorów. Na koniec zapisuje wynik obok pliku źródłowego i
go wczytuje.

<code-tabs name="convert" />

Konwersja nie odbywa się po cichu. Informacja o przekonwertowanym pliku jest
zapisywana w logu wraz z rodziną, nazwą źródłową, nazwą wyjściową i wynikową
liczbą klas. Dzięki temu log uruchomienia dokładnie wskazuje, co wczytano.

## Rozpakowywane układy

Checkpointy projektów źródłowych zagnieżdżają wagi w kilku typowych miejscach,
a konwerter sprawdza je po kolei, aż znajdzie tensory: blok EMA w `ema.module`
lub płaski `ema`, `ema_state_dict` po usunięciu prefiksu `module.`, następnie
`params_ema`, `params`, `ema_net`, `net`, `model`, `state_dict`, a na końcu sam
obiekt. Sprawdzanie kilku miejsc zamiast tylko pierwszego sprawia, że blok `ema`
zawierający jedynie liczniki nie zasłania rzeczywistych wag znajdujących się
niżej.

Usuwane są również prefiksy opakowań: `module.` z trenowania rozproszonego,
`_orig_mod.` ze skompilowanego modelu oraz zagnieżdżenie `model.model.` dodawane
przez niektóre redystrybucje.

## Co jest odczytywane i skąd

Rozmiar, zadanie i liczba klas pochodzą z tensorów, a nie z nazwy pliku. Dlatego
dostrojony checkpoint jest konwertowany z własną liczbą klas zamiast z wartością
domyślną architektury. Nazwy klas pobiera się z własnych metadanych checkpointu,
jeśli są dostępne, albo z bloku `args` lub `hyper_parameters`, jeśli znajdują się
tam. Lista jest przycinana do wykrytej liczby klas, aby dostrojony model, który
zachował bazowy zestaw etykiet, nie zawierał indeksów nieobecnych już w jego
głowicy.

Zadania gęste są obsługiwane jawnie, bez przypisywania im sztucznych etykiet.
Checkpoint głębi otrzymuje jedną klasę o nazwie `depth`, a checkpoint odtwarzania
jedną klasę o nazwie `image`. Checkpoint estymacji pozy musi dostarczyć liczbę
punktów kluczowych z tensorów albo z rodziny. Jeśli żadna z tych możliwości nie
przyniesie wyniku, konwersja zostaje odrzucona zamiast utworzenia niekompletnego
pliku.

RF-DETR ma własny mechanizm rozpoznawania, ponieważ wykrywanie rozmiaru wymaga
całego checkpointu, a jego głowica ma 91 wyjść, podczas gdy LibreYOLO stosuje
konwencję 80 klas COCO. Checkpoint jest normalizowany do 80 klas, jeśli zawiera
dokładnie 80 nazw, deklaruje liczbę klas równą 80, wskazuje COCO jako swój zbiór
danych albo w ogóle nie zawiera metadanych klas lub zbioru danych. Prawdziwy
model z 90 klasami, rozpoznany na podstawie nazw, jawnej liczby innej niż 80 lub
wskazania zbioru danych innego niż COCO, zostaje zachowany bez zmian.

## Miejsce zapisu przekonwertowanego pliku

Wynik jest zapisywany obok źródła i otrzymuje nazwę utworzoną na jego podstawie:

```text
<source stem>-<FilenamePrefix><size>[-<task suffix>].pt
```

Mały detektor YOLOv9 zapisany jako `upstream-checkpoint.pth` staje się więc
plikiem `upstream-checkpoint-LibreYOLO9t.pt`. Nazwa wywodzi się ze źródła, a nie
z rodziny, dzięki czemu dwa dostrojone modele tej samej rodziny i tego samego
rozmiaru w jednym katalogu nie nadpisują się nawzajem ani nie kolidują z
oficjalnym checkpointem. Plik jest zapisywany ponownie przy każdym wczytaniu,
więc nigdy nie pozostaje nieaktualny względem źródła. Jeśli katalog jest tylko
do odczytu, przekonwertowany plik trafia do nowego prywatnego katalogu
tymczasowego, a log wskazuje jego lokalizację.

Od tej chwili jest to zwykły checkpoint LibreYOLO. Wczytuje się przez ścieżkę
metadanych, a `libreyolo metadata` zgłasza go jako prawidłowy.

## Przypadki wymagające ręcznej obsługi

Dwie rodziny znajdują się poza ogólnym mechanizmem rozpoznawania. Rodzina
estymacji spojrzenia jest całkowicie wykluczona: służy tylko do inferencji, a
warunki redystrybucji jej wydanych wag zawierają ograniczenia. RF-DETR jest
wykluczony, ponieważ obsługuje go opisany wyżej dedykowany mechanizm
rozpoznawania.

Surowe checkpointy PIDNet z projektu źródłowego są odrzucane wraz z błędem
wskazującym plik `weights/convert_pidnet_weights.py`. Skrypt ten zapisuje
metadane semantyczne Cityscapes wymagane przez checkpoint.

D-FINE i DEIM mają te same klucze architektury, więc same tensory nie pozwalają
ich rozróżnić. Gdy obie rodziny zgłoszą dopasowanie pliku, a w procesie nie
uczestniczy rodzina pokrewna z wyróżniającym znacznikiem, rozstrzyga nazwa pliku.
Nazwa w formacie `dfine_hgnetv2_n_coco.pth` lub `deim_hgnetv2_n_coco.pth`
rozwiązuje niejednoznaczność, natomiast plik o nieinformacyjnej nazwie jest
odrzucany wraz z wyjaśnieniem zamiast zgadywania. Bezpośrednie utworzenie
instancji `LibreDFINE` lub `LibreDEIM` również rozwiązuje problem.

Gdy kilka rodzin zasadnie rozpoznaje jeden plik, podklasa ma pierwszeństwo przed
klasą bazową, którą rozszerza, a o pozostałych przypadkach decyduje kolejność w
rejestrze, ponieważ odzwierciedla ona szczegółowość kontroli każdej rodziny.
Nazwa pliku jest sprawdzana tylko przy remisie D-FINE i DEIM, więc nigdy nie
może nadać szerokiemu dopasowaniu pierwszeństwa przed precyzyjnym.

## Konwertery skryptowe

Repozytorium zawiera w katalogu `weights/` skrypty konwersji dla poszczególnych
rodzin oraz współdzielone funkcje pomocnicze do powtarzających się operacji.
Stanowią one ścieżkę dla plików odrzucanych przez mechanizm czasu działania,
pozwalają utworzyć checkpoint przed wczytaniem i obsługują rodziny, których
metadane trzeba podać zamiast wywnioskować z tensorów.

Skrypty te są częścią repozytorium, a nie zainstalowanego pakietu, dlatego ich
użycie wymaga sklonowania projektu:

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
python weights/convert_pidnet_weights.py --help
```

Każdy skrypt zapisuje checkpoint zgodny ze schematem v1.0, czyli z tym samym
wymaganiem, które spełnia automatyczna konwersja i opublikowane wagi. Zawartość
tego schematu opisuje strona [checkpointy i wagi](/docs/weights).
