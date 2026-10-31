---
title: Dome-DETR
families:
  - domedetr
seo_title: 'Dome-DETR: wykrywanie małych obiektów w LibreYOLO'
description: >-
  Dome-DETR do detekcji małych obiektów, trenowania i walidacji. Kopie wstępnie
  wytrenowanych checkpointów służą tylko badaniom akademickim.
lead: >-
  Specjalista od bardzo małych obiektów zbudowany na D-FINE: głowica gęstości
  decyduje, gdzie znajdują się obiekty, uwaga enkodera jest ograniczona do
  okien, które je zawierają, a liczba zapytań jest ustalana na podstawie tej
  gęstości zamiast być stała. LibreYOLO obsługuje to dla detekcji.
keywords:
  - Dome-DETR
  - wykrywanie małych obiektów
  - wykrywanie małych obiektów
  - obrazowanie lotnicze
  - wykrywanie dronów
  - teledetekcja
  - VisDrone
  - AI-TOD
  - DETR
  - zapytania adaptacyjne do gęstości
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Wstępnie wytrenowane wagi są ograniczone do badań akademickich.
        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt", device="cpu")
        print(model(SAMPLE_IMAGE).boxes)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        model.train(data="my-dataset.yaml", epochs=160, imgsz=800, batch=4,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 imgsz=800 batch=4 lr0=2e-4
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml
source_hash: 8482301790a9b8d9
---

## Instalacja

Dome-DETR nie potrzebuje żadnych dodatkowych opcji. Wszystko, co importuje, znajduje się w instalacji bazowej.

```bash
pip install libreyolo
```

## Predykcja

Sześć przekonwertowanych checkpointów pobiera się automatycznie z kopii w repozytoriach LibreYOLO. Warunki projektu źródłowego ograniczają ich użycie do badań akademickich.

<code-tabs name="predict" />

Zwrócony obiekt `Results` jest tym, który zwraca każda rodzina, więc zamiana na inny detektor to zmiana jednej linii. `conf` i `max_det` filtrują wybór zapytania; `iou` jest akceptowany dla zgodności z API, ale nie ma efektu, ponieważ dekoder jest przewidującym zestaw, który nie ma kroku NMS. Zobacz [predykcja](/docs/predict) dla źródeł, streaming i obsługi wyników.

Dwie funkcje są wyłączone dla tej rodziny. Przechwytywanie grafu CUDA jest wyłączone, ponieważ liczba zapytań PAQI zależy od danych, a przekaz w przód zmienia kształt w zależności od obrazu, co dokładnie jest tym, czego przechwytywanie grafu nie może obsłużyć. Augmentacja w czasie testu działa przy jednym ustalonym kwadratowym rozmiarze, więc żądanie wieloskalowe TTA nie ma efektu.

## Warianty

Trzy rozmiary, s, m i l, wszystkie w wymiarach 800 na 800. Rozmiar wybiera backbone, a zbiór danych, z którego pochodzą wagi, wybiera głębokość dekodera i budżet zapytań, więc sam kod rozmiaru nie identyfikuje wykresu. Wagi AI-TOD-V2 wybierają między 300 a 1500 zapytań na obraz, wagi VisDrone między 250 a 500, a duży model uruchamia cztery warstwy dekodera na AI-TOD-V2 w porównaniu do sześciu na VisDrone.

Dome-DETR to D-FINE z trzema dodatkami. DeFE przewiduje mapę gęstości. MWAS używa tej mapy, aby ograniczyć uwagę enkodera do okien, które faktycznie zawierają obiekty, zamiast zwracać uwagę wszędzie. PAQI dobiera rozmiar zestawu zapytań z tej samej gęstości zamiast dekodować stałe 300. Zysk koncentruje się tam, gdzie obiekty są najmniejsze, i zawęża się w miarę ich wzrostu: własna ablacja upstream przesuwa AP na bardzo małych obiektach z 14,0 do 17,8, podczas gdy AP na średnich obiektach przesuwa się tylko z 45,4 do 46,4. Traktuj to jako uzupełnienie do [D-FINE](/docs/models/d-fine) dla obrazów lotniczych, z dronów i zdalnego wykrywania, a nie jako jego zamiennik.

Dla tej rodziny nie zapisano wyników benchmarków Vision Analysis.

## Trenowanie

Dome-DETR jest możliwy do trenowania. Trening uruchamia pełny cel upstream: straty D-FINE plus nadzór nad gęstością i liczbą DeFE, z zapytaniami wypełniającymi wyłączonymi z terminów klasyfikacji oraz maskami uwagi do odszumiania na obraz, tak aby wypełnienie jednego obrazu nie przenikało do innego.

<code-tabs name="train" />

Konfiguracja dziedziczy przepis D-FINE i zmienia to, czego wymaga MWAS. `imgsz` wynosi 800, `lr0` to `2e-4`, grupa parametrów backbone jest skalowana przez `backbone_lr_mult=0.1`, a `multi_scale` jest wyłączone, ponieważ okna MWAS potrzebują, aby wejście pozostało podzielne przez krok 8. `batch` domyślnie wynosi 4, a nie 16 jak w D-FINE: PAQI dopełnia każdą partię do najszerszego elementu, tak aby pamięć śledziła najbardziej zajętą grafikę w partii, a nie przeciętną.

Jedno szczere zastrzeżenie dotyczące dokładności. Pociągi w górę trenują przez 160 epok na `MultiStepLR(milestones=[80, 120], gamma=0.8)`, podczas gdy te domyślne ustawienia uruchamiają harmonogram flat-cosine D-FINE przez te same 160 epok. Ten harmonogram nie został tutaj odtworzony, a liczby z AP z artykułu również nie zostały odtworzone, więc należy je traktować jako wyniki autorów upstream, a nie jako obietnicę, że ta receptura je osiąga. Dostarcz harmonogram upstream, jeśli celem jest dopasowanie się do artykułu.

Zobacz [trenowanie](/docs/train) dotyczące zbiorów danych, augmentacji, multi-GPU i loggerów.

## Walidacja

`val()` zwraca słownik kluczowany nazwą metryki i drukuje wyniki dla każdej klasy, gdy `verbose` jest włączony.

<code-tabs name="val" />

Walidacja odbywa się na twoim własnym zbiorze danych w formacie, na którym trenowałeś. Bramka walidacyjna COCO biblioteki nie ma tutaj zastosowania, ponieważ nie istnieje żaden checkpoint COCO dla tej rodziny, względem którego można by dokonać pomiaru.

## Eksport

Eksport nie jest obsługiwany dla żadnego formatu, a jego żądanie powoduje błąd zamiast wygenerowania pliku.

Powodem jest PAQI. Decyduje on o liczbie zapytań na obraz, na podstawie propozycji przefiltrowanych pod względem gęstości oraz zachłannej pętli tłumienia adaptacyjnego do gęstości, więc długość wyjścia dekodera jest cechą danych wejściowych, a nie samego grafu. Śledzenie wprasowuje w to liczbę, którą wygenerował obraz śledzenia, co powoduje artefakt, który cicho zwraca błędne wyniki dla każdego innego obrazu. Formulacja statyczna musiałaby rozwijać to tłumienie dla wszystkich 250 do 1500 kandydatów, a sprowadzenie do stałego top-k usunęłoby dokładnie ten malutki obiekt recall, dla którego rodzina istnieje. Jeśli potrzebujesz eksportowalnego transformera detekcji, [D-FINE](/docs/models/d-fine) jest tym, do którego warto sięgnąć.

## Checkpointy

<checkpoint-table />

## Licencjonowanie

<provenance-box>

Sześć kopii zachowuje ograniczenie projektu źródłowego do badań akademickich. Kod podlega osobnej licencji. Repozytorium źródłowe ma licencję Apache-2.0, port LibreYOLO ma licencję MIT, a wagi wytrenowane samodzielnie na własnych danych należą do ich autora.

</provenance-box>
