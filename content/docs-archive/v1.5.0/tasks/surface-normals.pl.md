---
title: Normalne powierzchni
seo_title: Estymacja normalnych powierzchni w LibreYOLO
description: >-
  Przewiduj gęste pole normalnych powierzchni z jednego obrazu w LibreYOLO.
  Poznaj konwencję układu kamery, waliduj błąd kątowy i eksportuj model.
lead: >-
  Estymacja normalnych powierzchni przewiduje kierunek zwrócenia każdej
  widocznej powierzchni. LibreYOLO udostępnia ją jako zadanie normal, które
  zwraca gęste pole wektorów jednostkowych na obszarze roboczym oryginalnego
  obrazu.
keywords:
  - estymacja normalnych powierzchni python
  - mapa normalnych z obrazu
  - geometria monokularna
  - metryka błędu kątowego
  - dense normal prediction
last_verified: 1.5.0
snippets:
  predict:
    - label: Predykcja pola normalnych
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE, save=True)


        normals = result.normal_map

        print(normals.data.shape)      # wektory jednostkowe float32 (H, W, 3)

        normals.assert_normalized()    # zgłasza błąd, jeśli piksel nie ma
        długości jednostkowej
    - label: Odczytywanie jednego piksela
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE)


        # Układ kamery OpenCV: +x w prawo, +y w dół, +z w głąb sceny.
        Powierzchnia

        # zwrócona do kamery ma wartość zbliżoną do (0, 0, -1).

        field = result.normals.data

        h, w = field.shape[:2]

        print(field[h // 2, w // 2])
    - label: Zapisywanie wizualizacji
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE)


        # plot() renderuje pole; jest zdefiniowane dla wyników normalnych i
        krawędzi.

        result.plot().save("normals.png")
  val:
    - label: Walidacja i odczytywanie kluczy metryk
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])     # stopnie
        print(metrics["metrics/median_angular_error"])   # stopnie
        print(metrics["metrics/within_11_25"])           # procent pikseli
        print(metrics["metrics/within_22_5"], metrics["metrics/within_30"])
  export:
    - label: Eksport
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
    - label: Uruchamianie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie sufiksu pliku, więc
        wyeksportowany artefakt

        # wczytuje się jak dowolny checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreMoGe2s-normal.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.normal_map.data.shape)
source_hash: d26d26d894b436ff
---

## Definicja

Zadanie `normal` przewiduje z jednego obrazu RGB trójskładnikowy wektor
jednostkowy dla każdego piksela, czyli kierunek zwrócenia powierzchni w tym
punkcie. W przeciwieństwie do głębi dane wyjściowe nie mają dowolnej skali,
dlatego można bezpośrednio porównywać dwie predykcje bez wyrównywania.

Predykcja wypełnia `result.normal_map`, czyli strukturę `NormalMap` zawierającą
tablicę float32 `(H, W, 3)` na obszarze roboczym oryginalnego obrazu, dostępną
również jako `result.normals`. Wektory używają układu kamery OpenCV przyjętego w
LibreYOLO, gdzie `+x` wskazuje w prawo, `+y` w dół, a `+z` w głąb sceny. Są
skierowane do kamery, więc powierzchnia równoległa do płaszczyzny obrazu ma
wartość `(0, 0, -1)`. Metoda `.assert_normalized()` sprawdza, czy każdy piksel
jest skończony i ma długość jednostkową w granicach tolerancji. Pole
`result.boxes` pozostaje puste, więc `conf`, `iou` i `max_det` nie mają wpływu,
a `Results.plot()` obsługuje to zadanie.

## Modele

Zadanie `normal` obsługują dwie rodziny.

[MoGe-2](/docs/models/moge-2) jest rodziną wyspecjalizowaną. To jednoetapowy
model geometrii monokularnej w trzech rozmiarach enkodera. LibreYOLO nie kopiuje
tych checkpointów do własnej organizacji. Wczytanie pobiera odpowiedni rozmiar
z oficjalnych repozytoriów w przypiętej rewizji i weryfikuje go względem
zarejestrowanego skrótu SHA-256.

[LibreMODUS](/docs/models/libremodus) generuje normalne jako jeden z celów modelu
przetwarzającego dowolne dane wejściowe na dowolne dane wyjściowe i może przyjąć
mapę głębi zamiast obrazu RGB. Wymaga dodatku `modus` oraz własnego
uwierzytelnionego konta Hugging Face. Nie oferuje ani `val()`, ani `export()`,
dlatego nie uczestniczy w poniższych sekcjach walidacji i eksportu.

## Predykcja

Przy pierwszym użyciu wagi MoGe-2 są pobierane i zapisywane lokalnie w pamięci
podręcznej.

<code-tabs name="predict" />

`imgsz` musi być podzielne przez rozmiar patcha enkodera ViT, co LibreYOLO
sprawdza przed rozpoczęciem przebiegu. Predykcja listy obrazów wykonuje po jednym
przebiegu na obraz. To zadanie nie ma szybkiej ścieżki połączonego batcha.
Informacje o źródłach, streamingu i obsłudze wyników znajdują się w sekcji
[predykcja](/docs/predict).

## Format zbioru danych

Walidacja normalnych paruje każdy obraz z trójkanałowym, 16-bitowym plikiem PNG o
tej samej nazwie bazowej i rozdzielczości oraz z opcjonalną maską poprawności.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  normals/
    val/room.png
  masks/
    val/room.png
```

```yaml
path: dataset
train: images/train
val: images/val
normals_dir: normals
masks_dir: masks
nc: 1
names: {0: normal}
```

Docelowy plik PNG ma dokładnie trzy kanały typu `uint16`, zapisane jako RGB.
Dekodowanie ma postać `n = png / 65535 * 2 - 1`, po czym każdy wektor jest
ponownie normalizowany. Zdekodowane wektory używają tego samego układu kamery
OpenCV co predykcje. Piksel maski jest uznawany za poprawny, gdy ma wartość
niezerową. Bez pliku maski każdy skończony, niezerowy zdekodowany wektor jest
poprawny. Niepoprawne i dopełnione piksele docelowe są wewnętrznie przechowywane
jako `(0, 0, 0)` i nigdy nie wpływają na metrykę. Pełny kontrakt opisano w sekcji
[formaty zbiorów danych](/docs/reference/dataset-formats).

## Trenowanie

Żadna rodzina normalnych nie ma implementacji trenowania. `train()` zgłasza
`NotImplementedError` w obu przypadkach. Strona MoGe-2 wskazuje przypięte
oficjalne checkpointy do predykcji, walidacji i eksportu.

## Walidacja

`val()` mierzy kąt między każdym przewidzianym wektorem a jego wektorem danych
referencyjnych (ground truth) dla pikseli oznaczonych przez zbiór danych jako
poprawne.

<code-tabs name="val" />

`metrics/mean_angular_error` i `metrics/median_angular_error` określają ten kąt w
stopniach, a niższa wartość jest lepsza. `metrics/within_11_25`,
`metrics/within_22_5` i `metrics/within_30` to odsetek poprawnych pikseli, których
błąd kątowy mieści się odpowiednio w 11.25, 22.5 i 30 stopniach, więc wyższa
wartość jest lepsza. Należy zwrócić uwagę na jednostkę: te trzy wartości są
procentami, a nie ułamkami. `fitness` to `metrics/within_11_25` podzielone przez
100, co umieszcza wybór najlepszego checkpointu na tej samej skali `[0, 1]` co
w każdym innym zadaniu.

## Eksport

Wyeksportowany model normalnych jest ponownie wczytywany przez `LibreYOLO()` na
podstawie sufiksu pliku, więc plik `.onnx` zachowuje się jak checkpoint i zwraca
ten sam obiekt `Results`.

<code-tabs name="export" />

Eksport normalnych używa kontraktu środowiska uruchomieniowego o stałej
rozdzielczości i batchu 1. `dynamic` oraz `batch` inny niż 1 są odrzucane, a
`imgsz` musi być podzielne przez rozmiar patcha enkodera. Zakres poszczególnych
formatów znajduje się na stronie [MoGe-2](/docs/models/moge-2) i w
[pełnej macierzy eksportu](/docs/reference/export-matrix).
[Eksport](/docs/export) wymienia argumenty akceptowane przez każdy format.

