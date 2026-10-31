---
title: Segmentacja sterowana promptem
seo_title: Segmentacja sterowana promptem w LibreYOLO
description: >-
  Zamieniaj punkt, ramkę lub pojęcie tekstowe w maskę obiektu w LibreYOLO.
  Wczytuj SAM, SAM 2, SAM 3, EdgeTAM, MobileSAM lub PicoSAM3 przez LibreSAM.
lead: >-
  Segmentacja sterowana promptem zamienia kliknięcie w maskę: wskazuje się
  obiekt lub rysuje wokół niego ramkę, a model zwraca jego obrys. W LibreYOLO
  nie jest to osobny klucz zadania, lecz warstwa modeli wczytywana przez fabrykę
  LibreSAM, której wyniki są zwykłymi obiektami Results segmentacji.
keywords:
  - segmentacja sterowana promptem
  - segmentacja interaktywna
  - segment anything python
  - prompt punktowy
  - prompt ramki
  - SAM python
  - maska z kliknięcia
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompty punktowe i ramkowe
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # Punkt ma postać [x, y] w pikselach; etykiety to 1 dla dodatnich i 0
        dla ujemnych.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # wielokąty

        print(result.boxes.xyxy)    # ciasne ramki wyprowadzone z masek


        # Prompt ramkowy zwraca jedną maskę na ramkę.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: 'Jedno kodowanie, wiele promptów'
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # set_image uruchamia ciężki enkoder obrazu raz i zapisuje wynik w
        pamięci podręcznej.

        model.set_image(SAMPLE_IMAGE)

        first = model.predict(points=[640, 420], labels=[1])

        second = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
    - label: Segmentowanie wszystkiego
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # Brak promptu oznacza siatkę punktów na całym obrazie. Domyślna

        # siatka 32 na bok to około 1024 przebiegi dekodera, co jest wolne na
        CPU.

        result = model.predict(SAMPLE_IMAGE, points_per_side=8)

        print(len(result.masks))
    - label: Maski niejednoznaczności
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # Jeden punkt może oznaczać rękaw, koszulę lub osobę. multimask=True
        # zwraca wszystkie trzy maski całości lub części zamiast najlepszej.
        result = model.predict(
            SAMPLE_IMAGE, points=[640, 420], labels=[1], multimask=True
        )
        print(len(result.masks))
source_hash: bb70ff24e6c0a767
---

## Definicja

Segmentacja sterowana promptem przyjmuje obraz wraz z promptem przestrzennym i
zwraca maskę wskazanego obiektu. Nic nie jest klasyfikowane. Nie ma listy klas,
a `result.boxes` zawiera ciasne ramki wyprowadzone z masek, a nie samodzielne
detekcje. Pole `result.masks` zawiera dane masek, natomiast `result.masks.xy` ich
wielokąty.

Interfejsem jest prompt. `points` zawiera współrzędne pikselowe `[x, y]`, po
jednym zestawie na obiekt, a `labels` oznacza każdy punkt jako dodatni (1,
uwzględnij) lub ujemny (0, wyklucz). `bboxes` ma postać `[x1, y1, x2, y2]`, z
jedną maską na ramkę. Punkty i ramki można łączyć. Są wtedy parowane według
obiektów i muszą mieć tę samą długość. Pominięcie wszystkich promptów uruchamia
ścieżkę segmentowania wszystkiego, czyli siatkę punktów na obrazie.

Pojedynczy punkt jest z założenia niejednoznaczny. Kliknięcie rękawa może
oznaczać rękaw, koszulę albo osobę, dlatego `multimask=True` zwraca dla każdego
promptu wszystkie trzy maski całości lub części zamiast jednej najlepszej.
`conf` filtruje według przewidywanego przez model IoU, czyli wyniku jakości
maski, a nie pewności detekcji.

LibreYOLO nie ma klucza zadania `promptable`. Warstwa rejestruje się jako
`segment`, czyli pod tym samym kluczem co segmentacja instancji. Odróżnia ją
postać wywołania, dlatego ma własną fabrykę `LibreSAM()`, równorzędną z
`LibreYOLO()`, `LibreOpenVocab()` i `LibreVLM()`. Pojedyncza sygnatura
`predict(image)` nie może wyrazić pętli, do której zbudowano te modele.
`set_image()` uruchamia enkoder obrazu raz i zapisuje embeddingi w pamięci
podręcznej, każde późniejsze wywołanie `predict()` z `source=None` ponosi tylko
koszt dekodowania promptu, a `reset_image()` czyści pamięć podręczną. Enkoder
obrazu jest głównym źródłem kosztu i działa raz na obraz, więc drugi prompt na
tym samym obrazie całkowicie go pomija.

## Modele

Sześć rodzin wczytuje się przez alias za pomocą `LibreSAM`.

[SAM](/docs/models/sam) jest domyślny i występuje w rozmiarach `base`, `large`
oraz `huge`, zapisywanych też jako `b`, `l` i `h`.

[SAM 2](/docs/models/sam-2) występuje jako `sam2-tiny`, `sam2-small`,
`sam2-base-plus` i `sam2-large`. LibreYOLO obsługuje jego ścieżkę obrazu.

[SAM 3](/docs/models/sam-3), dostępny jako `sam3`, jest jedyną rodziną, która
przyjmuje prompt pojęcia tekstowego. `text="yellow school bus"` zwraca każdą
pasującą instancję. Przekazanie `text=` do dowolnej innej rodziny zgłasza błąd z
komunikatem wskazującym SAM 3. Wagi pochodzą od Meta i są udostępniane na
niestandardowej licencji SAM License zamiast licencji MIT biblioteki LibreYOLO,
a repozytorium jest zabezpieczone. Przed pierwszym pobraniem zaakceptuj warunki
na stronie modelu i uwierzytelnij się przez `hf auth login`. Przed wdrożeniem
należy przeczytać stronę [SAM 3](/docs/models/sam-3).

[EdgeTAM](/docs/models/edgetam), dostępny jako `edgetam`, jest wariantem SAM 2
na urządzenia. LibreYOLO obsługuje jego ścieżkę obrazu.

[MobileSAM](/docs/models/mobilesam), dostępny jako `mobilesam`, zastępuje enkoder
ViT-H z SAM destylowanym enkoderem TinyViT.

[PicoSAM3](/docs/models/picosam3), dostępny jako `picosam3`, jest kompaktową
siecią CNN do obszarów wskazywanych promptem ramkowym na czujnikach brzegowych.
Prompty ramkowe stanowią tutaj cały kontrakt. Punkty, tekst, maska, multimask i
segmentowanie wszystkiego zgłaszają błąd z komunikatem wskazującym SAM 2 lub
SAM 3.

Dodatek tej warstwy obejmuje cztery rodziny wczytywane przez `transformers`:

```bash
pip install "libreyolo[sam]"
```

MobileSAM i PicoSAM3 są natywnymi portami LibreYOLO i do działania nie wymagają
instalacji `transformers`.

## Predykcja

<code-tabs name="predict" />

`source` i `set_image()` są alternatywami, a nie sekwencją. Przekaż obraz do
`predict()`, aby wykonać jednorazowe wywołanie, albo najpierw wywołaj
`set_image()`, a następnie `predict(source=None)` dla każdego promptu.
Przekazanie `device=` do `predict()` przenosi model dla tego i wszystkich
kolejnych wywołań oraz unieważnia wszystkie embeddingi w pamięci podręcznej.

Segmentowanie wszystkiego jest kosztownym trybem. `points_per_side` ma domyślną
wartość 32, co odpowiada około 1024 przebiegom dekodera na obrazie. Dla pracy
interaktywnej na CPU należy ją zmniejszyć. Jeśli `conf` nie jest ustawione, w
tym trybie stosowany jest próg siatki danej rodziny, natomiast w ścieżce z
promptem zachowywane są wszystkie maski. Przekazanie `conf=0.0` wyłącza
filtrowanie w obu trybach, a `max_det` ogranicza liczbę zwracanych masek.

Prompty masek nie są obsługiwane w tej wersji, a `masks=` zgłasza błąd zamiast
być ignorowane. `track()` również zgłasza błąd w całej warstwie. Są to
segmentatory obrazów, dlatego należy uruchamiać `predict()` dla każdej klatki.
Informacje o źródłach i obsłudze wyników znajdują się w sekcji
[predykcja](/docs/predict).

## Trenowanie

Żadna rodzina w tej warstwie nie jest trenowana wewnątrz LibreYOLO. `train()`
zgłasza błąd. Należy dostroić model w projekcie nadrzędnym i wczytać wynikowe
wagi.

## Walidacja

Dla tej warstwy nie ma walidatora, a `val()` zgłasza błąd. Maska sterowana
promptem nie ma stałego zestawu klas do porównania, więc zwykłe metryki detekcji
i segmentacji nie mają klucza odniesienia. Ocena takiej maski polega na
porównaniu jej z samodzielnie dostarczoną maską referencyjną dla istotnych
promptów.

## Eksport

Eksport nie wchodzi w zakres całej warstwy, a `export()` zgłasza błąd, z jednym
wyjątkiem. [PicoSAM3](/docs/models/picosam3) eksportuje surową sieć CNN obszaru
96x96 do ONNX jako `roi_image -> mask_logits`. Wycinanie ramki i ponowne
skalowanie maski do współrzędnych obrazu pozostają w Pythonie. Każda inna
rodzina działa przez `predict()` w środowisku PyTorch. Formaty dostępne w innych
częściach biblioteki opisano w sekcji [eksport](/docs/export).

