---
title: Siatka ciała
seo_title: Rekonstrukcja siatki ciała w LibreYOLO
description: >-
  Odtwarzaj parametryczną siatkę 3D ciała dla każdej osoby w LibreYOLO.
  Przewiduj na podstawie ramek osób lub detektora i odczytuj wierzchołki, stawy
  oraz translację kamery.
lead: >-
  Rekonstrukcja siatki ciała przekształca pojedynczy obraz i zestaw ramek osób w
  parametryczne ciało 3D dla każdej osoby: parametry kształtu i pozy, ustawione
  wierzchołki, stawy 3D oraz translację kamery umieszczającą je przed
  obiektywem.
keywords:
  - rekonstrukcja siatki człowieka python
  - siatka ciała 3d
  - pozycja ciała 3d
  - SAM 3D Body
  - MHR
  - parametryczny model ciała
  - libreyolo mesh
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # Ta rodzina nie jest zarejestrowana w funkcji fabrykującej LibreYOLO(),

        # dlatego tworzy się ją bezpośrednio. model_path=None uruchamia
        pobieranie

        # z Hugging Face z ograniczonym dostępem, a ciąg znaków jest traktowany
        jako

        # istniejący lokalny checkpoint i nigdy nie jest pobierany. Inferencja
        wymaga CUDA.

        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])


        meshes = result.meshes

        print(meshes.body_model)      # parametryzacja używana przez te tensory

        print(meshes.vertices.shape)  # (N, V, 3), układ kamery, metry

        print(meshes.joints3d.shape)  # (N, J, 3)

        print(meshes.joints2d.shape)  # (N, J, 2), piksele na obrazie źródłowym
    - label: Z detektorem osób
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # person_detector przyjmuje utworzony detektor LibreYOLO, zwykły obiekt

        # wywoływalny albo instancję PersonDetector. Nie ma skrótu w postaci
        nazwy.

        detector = LibreYOLO("LibreYOLO9s.pt")

        model = LibreSAM3DBody(None, size="d3", device="cuda")


        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 31c5b44171cbcd0e
---

## Definicja

Rekonstrukcja siatki ciała zwraca dla każdego obrazu strukturę `Meshes`, której
wiersze są wyrównane z `result.boxes`. Wiersz `i` opisuje osobę w ramce `i`,
zgodnie z tym samym kontraktem, którego zadanie estymacji pozy używa dla punktów
kluczowych.

Wszystkie wartości są wyrażone w układzie kamery oryginalnego obrazu. `transl`
jest wartością metryczną w metrach, a +z wskazuje kierunek od kamery. `vertices`
i `joints3d` są wartościami metrycznymi i uwzględniają już `transl`, więc nie
wymagają dalszej kompozycji. `joints2d` podaje piksele na płótnie oryginalnego
obrazu, a nie na wycinku widzianym przez sieć. `faces` przechowuje topologię
siatki jeden raz dla całego obrazu zamiast dla każdego wiersza, ponieważ każda
osoba korzysta z tej samej topologii. W tej wersji nie ma układu świata ani
grawitacji i żadne pole nie zastępuje ich niejawnie.

Układy parametrów różnią się między modelami ciała, dlatego żaden kształt nie
jest stały. `body_model` podaje nazwę parametryzacji, a liczby są odczytywane z
tensorów. W przypadku `"mhr"`, czyli Momentum Human Rig, obroty są kątami Eulera
w radianach zamiast reprezentacją oś-kąt, `body_pose` jest płaskim wektorem
parametrów dla poszczególnych stawów zamiast jednej trójki na staw, a `betas`
stanowią współczynniki blendshape tożsamości. Skala szkieletu, pozycja dłoni i
wyraz twarzy znajdują się w `extras`.

Kanoniczny klucz zadania to `mesh`. Wartości `body-mesh`, `hmr` i
`human-mesh-recovery` są do niego normalizowane.

## Modele

[SAM 3D Body](/docs/models/sam-3d-body) jest jedyną rodziną obsługującą to
zadanie i stanowi opakowanie zamiast portu. Pakiet `sam-3d-body` firmy Meta jest
publikowany na SAM License, od której własny kod LibreYOLO nie może się wywodzić,
dlatego żadna jego część nie jest dołączona. Dwa backbone korzystają z tego
samego modelu ciała MHR: `d3` na enkoderze DINOv3 ViT-H/16+ oraz `h` na
oryginalnym ViT-H.

Przed pierwszą predykcją trzeba spełnić trzy wymagania. Żadne z nich nie jest
opcjonalne.

Pakiet projektu źródłowego instaluje się samodzielnie, a nie przez LibreYOLO:

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

Wskaż bibliotece sklonowany katalog przez `sam_3d_body_path=` lub zmienną
środowiskową `SAM_3D_BODY_PATH`. Jeśli ta rodzina nigdy nie jest tworzona, import
nie zostanie uruchomiony.

Dostęp do kopii checkpointu jest ograniczony. Zaakceptuj licencję na stronie
modelu Hugging Face i uwierzytelnij się poleceniem `hf auth login`. W przeciwnym
razie pierwsze pobranie zakończy się niepowodzeniem. Sam model ciała MHR jest
osobnym wydaniem Apache-2.0, pobieranym z własnej publicznej lokalizacji i
zapisywanym w lokalnej pamięci podręcznej.

Inferencja wymaga urządzenia CUDA. Estymator projektu źródłowego przenosi batch
na GPU bez sprawdzania, dlatego nie ma zapasowej ścieżki CPU, a `device="cpu"`
zgłasza wyjątek.

## Predykcja

<code-tabs name="predict" />

Osoby trafiają do modelu na jeden z dwóch sposobów. `person_boxes` przekazuje
posiadane ramki i działa tylko dla pojedynczego obrazu. Stały zestaw ramek nie
może podążać za osobami w kolejnych klatkach wideo, więc użycie go ze źródłem
wideo zgłasza wyjątek zamiast po cichu ponownie używać ramek z pierwszej klatki.
`person_detector` przyjmuje utworzony detektor LibreYOLO, obiekt wywoływalny lub
`PersonDetector` i stanowi ścieżkę dla wideo. `focal_length` podaje znaną wartość
wewnętrzną kamery. Jeśli nie zostanie ustawiona, model używa własnej estymacji,
którą zwraca `meshes.focal_length`.

Ta rodzina nie jest podłączona do funkcji fabrykującej `LibreYOLO()` ani do
polecenia CLI `libreyolo predict`. Jedynym punktem wejścia jest
`LibreSAM3DBody`. Informacje o źródłach, streamingu i obsłudze wyników zawiera
strona [predykcji](/docs/predict).

## Trenowanie

Żadna rodzina w tym zadaniu nie jest trenowana w LibreYOLO.
`LibreSAM3DBody.train()` zgłasza wyjątek. Trenowanie należy przeprowadzić w
projekcie źródłowym, a wynikowy checkpoint wczytać tutaj.

## Walidacja

Nie ma walidatora siatek, a `val()` zgłasza wyjątek. Typowe benchmarki są
przeznaczone wyłącznie do celów badawczych, więc żaden nie jest dołączony ani
nie może zostać automatycznie pobrany.

Same metryki są dostępne jako `libreyolo.validation.mesh_metrics` i służą do
oceny względem posiadanego zbioru danych. Funkcja przyjmuje przewidywane i
docelowe stawy, opcjonalnie przewidywane i docelowe wierzchołki, a zwraca słownik
z kluczami identycznymi jak w walidatorze:

`metrics/mpjpe` jest średnim błędem położenia stawu po wyrównaniu stawu
głównego. Ocenia więc pozę, pomijając miejsce osoby w scenie.
`metrics/pa_mpjpe` jest tą samą wartością po pełnym wyrównaniu Prokrustesa, czyli
obrocie, jednolitym skalowaniu i translacji. Usuwa to błąd globalnej orientacji
i rozmiaru ciała, pozostawiając pozę artykułowaną. `metrics/pve` jest średnim
błędem wierzchołka na powierzchni siatki po wyrównaniu względem centroidu
wierzchołków. W przeciwieństwie do metryk stawów uwzględnia kształt ciała i
pojawia się tylko wtedy, gdy dostarczono obie tablice wierzchołków. We wszystkich
trzech przypadkach mniejsza wartość jest lepsza. Przyjmuje się, że dane wejściowe
są metryczne i podane w metrach, a `scale_to_mm` przelicza wyniki na milimetry
stosowane w literaturze.

## Eksport

Eksport siatki nie jest zaimplementowany. LibreYOLO nie zdefiniowało kontraktu
metadanych wyeksportowanego grafu dla tego zadania, w tym sposobu przenoszenia
układu parametrów MHR poza PyTorch. Dlatego `export()` zgłasza wyjątek zamiast
emitować graf, którego danych wyjściowych nie dałoby się zinterpretować.
