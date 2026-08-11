---
title: Dome-DETR
families:
  - domedetr
seo_title: 'Dome-DETR: wykrywanie małych obiektów w LibreYOLO'
description: >-
  Użyj Dome-DETR w LibreYOLO do wykrywania małych obiektów na zdjęciach
  lotniczych i z dronów. Konwertuj wagi upstream, przewiduj, dopracowuj i
  waliduj w ramach kodu MIT-licensed.
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
last_verified: 1.5.0
snippets:
  predict:
    - label: 'Konwertuj, a następnie przewiduj'
      language: bash
      code: >
        # Żadne wagi Dome-DETR nie są hostowane przez LibreYOLO, więc checkpoint
        to

        # pobrano z repozytorium upstream i przetworzono raz.

        hf download RicePasteM/Dome-DETR --include 'best_ckpts_dome_2026/*' \
          --local-dir dome-ckpts

        python weights/convert_domedetr_weights.py \
          dome-ckpts/best_ckpts_dome_2026/dome-s-visdrone_converted.pth \
          LibreDOMEDETRs-visdrone.pt --size s --variant visdrone
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Lokalna ścieżka, nie sama nazwa: nic nie jest pobierane dla tej
        rodziny.

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        result = model("drone-frame.jpg", save=True)


        for box in result.boxes:
            print(result.names[int(box.cls)], box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDOMEDETRs-visdrone.pt
        source=drone-frame.jpg save=True
    - label: Nazwy klas
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Nie ma checkpointu COCO, więc klasy pochodzą z zbioru danych
        # Wagi były trenowane i są odczytywane z metadanych checkpointu.
        aitod = LibreYOLO("LibreDOMEDETRs-aitod.pt")
        print(aitod.model.names)     # 9 klas AI-TOD-V2

        visdrone = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        print(visdrone.model.names)  # 12 klas VisDrone
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
source_hash: 381f01d769e7c420
---
## Instalacja

Dome-DETR nie potrzebuje żadnych dodatkowych opcji. Wszystko, co importuje, znajduje się w instalacji bazowej.

```bash
pip install libreyolo
```

## Predykcja

Nie ma nic do automatycznego pobrania. LibreYOLO nie udostępnia tych wag, więc przebieg jest następujący: pobierz checkpoint upstream, przekonwertuj go raz, a następnie załaduj przekonwertowany plik za pomocą ścieżki. [Licencja](#licensing) wyjaśnia dlaczego.

<code-tabs name="predict" />

Zwrócony obiekt `Results` jest tym, który zwraca każda rodzina, więc zamiana na inny detektor to zmiana jednej linii. `conf` i `max_det` filtrują wybór zapytania; `iou` jest akceptowany dla zgodności z API, ale nie ma efektu, ponieważ dekoder jest przewidującym zestaw, który nie ma kroku NMS. Zobacz [predykcja](/docs/predict) dla źródeł, streaming i obsługi wyników.

Dwie funkcje są wyłączone dla tej rodziny. Przechwytywanie grafu CUDA jest wyłączone, ponieważ liczba zapytań PAQI zależy od danych, a przekaz w przód zmienia kształt w zależności od obrazu, co dokładnie jest tym, czego przechwytywanie grafu nie może obsłużyć. Augmentacja w czasie testu działa przy jednym ustalonym kwadratowym rozmiarze, więc żądanie wieloskalowe TTA nie ma efektu.

## Warianty

Trzy rozmiary, s, m i l, wszystkie w wymiarach 800 na 800. Rozmiar wybiera backbone, a zbiór danych, z którego pochodzą wagi, wybiera głębokość dekodera i budżet zapytań, więc sam kod rozmiaru nie identyfikuje wykresu. Wagi AI-TOD-V2 wybierają między 300 a 1500 zapytań na obraz, wagi VisDrone między 250 a 500, a duży model uruchamia cztery warstwy dekodera na AI-TOD-V2 w porównaniu do sześciu na VisDrone.

Dome-DETR to D-FINE z trzema dodatkami. DeFE przewiduje mapę gęstości. MWAS używa tej mapy, aby ograniczyć uwagę enkodera do okien, które faktycznie zawierają obiekty, zamiast zwracać uwagę wszędzie. PAQI dobiera rozmiar zestawu zapytań z tej samej gęstości zamiast dekodować stałe 300. Zysk koncentruje się tam, gdzie obiekty są najmniejsze, i zawęża się w miarę ich wzrostu: własna ablacja upstream przesuwa AP na bardzo małych obiektach z 14,0 do 17,8, podczas gdy AP na średnich obiektach przesuwa się tylko z 45,4 do 46,4. Traktuj to jako uzupełnienie do [D-FINE](/docs/models/d-fine) dla obrazów lotniczych, z dronów i zdalnego wykrywania, a nie jako jego zamiennik.

LibreYOLO nie publikuje żadnych wierszy referencyjnych dla tej rodziny, ponieważ nie publikuje żadnych checkpointów do porównania.

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

Nie ma żadnych do wymienienia. LibreYOLO nie publikuje żadnych wag Dome-DETR, a żadna nazwa w formie `LibreDOMEDETR<size>-<dataset>.pt` nie prowadzi do pobrania.

Upstream publikuje sześć checkpointów, s, m i l dla każdego z dwóch zbiorów danych: AI-TOD-V2 z 9 klasami i VisDrone z 12. Nie ma checkpointu COCO, więc kanoniczna nazwa pliku zawsze zawiera sufiks zbioru danych, a nazwy klas znajdują się w metadanych checkpointu, zamiast pochodzić z stałej rodziny. Próba użycia samego `LibreDOMEDETRs.pt` powoduje natychmiastowy błąd z komunikatem podającym dwie rzeczywiste nazwy plików i polecenie konwersji, zamiast próby pobrania, która skutkowałaby błędem 404.

`weights/convert_domedetr_weights.py` wykonuje konwersję. Odbudowuje graf LibreYOLO, ładuje do niego tensory upstream i odmawia zapisania czegokolwiek, jeśli brakuje choćby jednego klucza, jest on nieoczekiwany lub ma niewłaściwy kształt, więc przekonwertowany plik jest albo dokładnie taki sam, albo nie istnieje. Wskaż go na upstream `.pth` i przekaż rozmiar oraz wariant:

```bash
python weights/convert_domedetr_weights.py \
    dome-ckpts/best_ckpts_dome_2026/aitod-s-best.pth \
    LibreDOMEDETRs-aitod.pt --size s --variant aitod
```

Pod kątem wierności numerycznej, `weights/parity_domedetr.py` porównuje ten port z implementacją upstream we wszystkich sześciu punktach kontrolnych i raportuje `max_abs_diff == 0.0` zarówno na `pred_logits`, jak i `pred_boxes`, po uprzednim sprawdzeniu bit po bicie maski okna MWAS, a osobno porównuje każdy termin straty z kryterium upstream. Bądźmy jasni: jest to ręczny skrypt, który wymaga upstream checkout i opublikowanych checkpointów na dysku, uruchamiany ręcznie. Nie jest częścią ciągłej integracji i żaden job CI go nie odtwarza.

## Licencjonowanie

<provenance-box>

Wagi są powodem, dla którego ta rodzina nie jest lustrzana. Karta modelu upstream nie zawiera pola licencji w swojej metadanej, a jej opis stwierdza, że projekt to Apache-2.0, jednocześnie ograniczając materiały wyłącznie do celów badań akademickich. Te dwa odczyty się nie zgadzają, a ten bardziej restrykcyjny nie jest przyznaniem prawa do redystrybucji, więc LibreYOLO linkuje repozytorium upstream zamiast kopiować pliki, oczekując na wyjaśnienie. Ta sama logika obowiązuje w przypadku [YOLO-NAS](/docs/models/yolo-nas) tutaj.

Kod to osobne pytanie i jaśniejsze. Repozytorium upstream to Apache-2.0, port LibreYOLO to MIT, a wagi, które trenujesz samodzielnie na swoich danych, należą do ciebie.

</provenance-box>

## Cytowanie

Dome-DETR został opublikowany w ACM Multimedia 2025 jako „Dome-DETR: DETR z manipulacją cechowej kwerendy zorientowanej na gęstość dla efektywnego wykrywania małych obiektów”. Preprint jest dostępny na [arxiv.org/abs/2505.05741](https://arxiv.org/abs/2505.05741). Autorzy nie publikują żadnego bloku BibTeX w swoim repozytorium, więc żaden z nich nie jest tu odtworzony, a raczej złożony ręcznie.

<citation-block />
