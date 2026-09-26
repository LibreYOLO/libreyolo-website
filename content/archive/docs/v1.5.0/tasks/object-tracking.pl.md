---
title: Śledzenie obiektów
seo_title: Śledzenie obiektów w LibreYOLO
description: >-
  Śledź obiekty między klatkami wideo w LibreYOLO za pomocą ByteTrack, BoT-SORT,
  OC-SORT lub Deep OC-SORT, korzystając z dowolnego modelu detekcji, segmentacji
  lub estymacji pozy.
lead: >-
  Śledzenie przypisuje każdej detekcji stabilną tożsamość między klatkami wideo.
  LibreYOLO nie traktuje go jako zadania z własnymi wagami: jest to tryb
  predykcji, model.track(), który uruchamia wybrany tracker na wynikach modelu
  detekcji, segmentacji lub estymacji pozy z poszczególnych klatek.
keywords:
  - śledzenie obiektów python
  - multi object tracking
  - bytetrack
  - botsort
  - ocsort
  - deep ocsort
  - track id
  - śledzenie reid
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # track() jest generatorem: jeden obiekt Results na każdą przetworzoną
        klatkę.

        for result in model.track("video.mp4"):
            print(result.track_id)        # tensor liczb całkowitych (N,), wyrównany z ramkami
            print(result.boxes.xyxy)
    - label: Wybór trackera
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # "bytetrack" (domyślnie), "botsort", "ocsort" lub "deepocsort".
        for result in model.track("video.mp4", tracker="botsort"):
            print(result.track_id)
    - label: Zapis wideo z adnotacjami
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Bez output_path plik trafia do runs/track/<video_stem>.mp4.
        for result in model.track("video.mp4", save=True, vid_stride=2):
            pass
    - label: Dostrajanie trackera
      language: python
      code: >
        from libreyolo import BoTSortConfig, LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Typ konfiguracji wybiera tracker, dlatego tracker= jest tutaj zbędny.

        config = BoTSortConfig(track_buffer=60, frame_rate=25, enable_cmc=False)

        for result in model.track("video.mp4", tracker_config=config):
            print(result.track_id)

        # Można też przekazać te same pola jako argumenty nazwane i pozwolić
        track() utworzyć konfigurację.

        for result in model.track("video.mp4", tracker="botsort",
        track_buffer=60):
            print(result.track_id)
source_hash: f1fa7dcf60597d6b
---

## Definicja

Śledzenie nie jest jednym z kluczy zadań LibreYOLO i nie ma checkpointu
śledzenia do pobrania. Jest to metoda modelu, `model.track(source)`, która
uruchamia detekcję na każdej klatce i wiąże wyniki w czasie. Metoda jest
generatorem: zwraca jeden obiekt `Results` na każdą przetworzoną klatkę, z polem
`result.track_id` ustawionym na tensor liczb całkowitych `(N,)`, wyrównany z
`result.boxes`. Te same identyfikatory znajdują się także w `result.boxes.id`.

Zwracane są tylko potwierdzone, aktualnie śledzone obiekty. Ścieżka utracona
przez algorytm asocjacji pozostaje aktywna przez skonfigurowaną liczbę klatek,
zanim zostanie usunięta. Określa ją `track_buffer` dla ByteTrack i BoT-SORT oraz
`max_age` dla obu wariantów OC-SORT, dzięki czemu obiekt ponownie znaleziony w
tym przedziale zachowuje pierwotny identyfikator.

Ponieważ asocjacja odbywa się po detekcji, pozostałe dane klatki zostają
zachowane: śledzony obiekt `Results` jest wynikiem detekcji `Results` ograniczonym
do dopasowanych wierszy, więc wraz z ramkami przekazywane są maski i punkty
kluczowe.

## Modele

Na przebieg śledzenia składają się dwa niezależne wybory: model generujący ramki
dla każdej klatki oraz tracker, który je łączy.

Każdy natywny model LibreYOLO do detekcji, segmentacji lub estymacji pozy
udostępnia `track()`, więc detektor wybiera się tak samo jak zwykle. Pełna lista
znajduje się w [indeksie modeli](/docs/models), a zacząć można od
[YOLO9](/docs/models/yolov9), [RF-DETR](/docs/models/rf-detr),
[D-FINE](/docs/models/d-fine) lub [RTMDet](/docs/models/rtmdet). Zadania, których
wyniki nie mają ramki do powiązania, odrzucają wywołanie zamiast zwracać
bezsensowne identyfikatory. Dotyczy to klasyfikacji, obróconych ramek, punktów,
głębi, normalnych powierzchni, krawędzi, segmentacji semantycznej i panoptycznej,
rekonstrukcji, OCR oraz siatki ciała. Wszystkie zgłaszają błąd z `track()`.

Dwie warstwy modeli LibreYOLO również nie obsługują tej metody. Modele wczytane
przez `LibreSAM` są segmentatorami obrazów, a modele wczytane przez
`LibreOpenVocab` są detektorami pojedynczych klatek. Oba zgłaszają błąd z
`track()` i zamiast tego używa się ich z `predict()` dla każdej klatki.

Śledzenie działa na natywnych modelach PyTorch. Wyeksportowany artefakt wczytany
przez `LibreYOLO("model.onnx")` zwraca obiekt backendu środowiska
uruchomieniowego, który udostępnia `predict()`, ale nie `track()`.

Biblioteka zawiera cztery trackery wybierane argumentem `tracker`:

`"bytetrack"` jest domyślny. Korzysta wyłącznie z ruchu, filtru Kalmana i
trójetapowej asocjacji. Najpierw dopasowuje detekcje o wysokiej pewności,
następnie w drugim przebiegu daje detekcjom o niskiej pewności szansę na
dopasowanie do istniejącej ścieżki przed ich odrzuceniem, a na końcu dopasowuje
niepotwierdzone ścieżki. Konfiguruje się go przez `TrackConfig`.

`"botsort"` zachowuje trójetapowy cykl życia ByteTrack, lecz używa stanu Kalmana
w postaci środek-szerokość-wysokość i przed dopasowaniem kompensuje przewidywane
ścieżki o ruch kamery. Jest to wariant BoT-SORT korzystający wyłącznie z ruchu,
bez modelu wyglądu. Konfiguruje się go przez `BoTSortConfig`, który dodaje
`enable_cmc`, `cmc_method` i `cmc_downscale`.

`"ocsort"` także korzysta wyłącznie z ruchu i dodaje do kosztu asocjacji składnik
kierunku prędkości, drugi przebieg asocjacji względem ostatniej rzeczywistej
obserwacji każdej ścieżki oraz wygładzanie stanu Kalmana wzdłuż wirtualnej
trajektorii po ponownym znalezieniu ścieżki. Konfiguruje się go przez
`OCSortConfig`.

`"deepocsort"` rozszerza OC-SORT o wygląd. Każda ścieżka przechowuje ważoną
pewnością średnią ruchomą embeddingów ponownej identyfikacji, a składnik
podobieństwa cosinusowego dołącza do kosztu asocjacji. Dzięki temu tożsamości
przetrwają długie zasłonięcia i przecinanie się obiektów. Wymaga to jednego
przejścia małej sieci embeddingów na klatkę, a jej wagi OSNet są pobierane przy
pierwszym użyciu. Konfiguruje się go przez `DeepOCSortConfig`.

## Predykcja

<code-tabs name="predict" />

`track_conf` ustawia próg pierwszego etapu asocjacji: `track_high_thresh` dla
ByteTrack i BoT-SORT oraz `det_thresh` dla OC-SORT i Deep OC-SORT. Nie jest to
`conf` z `predict()`. W przypadku ByteTrack, BoT-SORT i OC-SORT detektor działa
wewnętrznie z niższym progiem, aby słabe detekcje pozostały dostępne w przebiegu
odzyskiwania. Deep OC-SORT uruchamia detektor bezpośrednio z `det_thresh`. Dla
ByteTrack i BoT-SORT wartość `track_conf` musi być równa lub wyższa niż
`track_low_thresh`, którego wartość domyślna to 0.1.

Ustawienia trackera można przekazać na dwa sposoby. Instancję konfiguracji można
przekazać do `tracker_config=`, a jej typ wybierze tracker, przez co `tracker=`
będzie zbędny. Można też przekazać pola jako argumenty nazwane i pozwolić
`track()` utworzyć konfigurację dla wskazanego trackera. Nieznane klucze
powodują ostrzeżenie zamiast cichego zastosowania. W obu przypadkach
`track_conf` jest ignorowane po jawnym ustawieniu odpowiedniego klucza.

Pozostałe argumenty odpowiadają predykcji: `iou`, `imgsz`, `classes`, `max_det`,
`vid_stride`, `show` oraz `save` z `output_path`. Źródłem jest ścieżka do pliku
wideo. Obsługę wyników opisano w sekcji [predykcja](/docs/predict).

## Trenowanie

Trackery nie są trenowane. Trzy z czterech to modele czysto ruchowe, całkowicie
pozbawione wyuczonych parametrów, a sieć wyglądu Deep OC-SORT jest opublikowanym
checkpointem ponownej identyfikacji, pobieranym przy pierwszym użyciu. Poprawa
jakości śledzenia wymaga poprawy detektora lub dostrojenia opisanych wyżej
progów asocjacji.

