---
title: Embeddingi
seo_title: Embeddingi obrazów i regionów w LibreYOLO
description: >-
  Zadanie embed zwraca znormalizowane metodą L2 wektory float32 dla całego
  obrazu, każdego wykrytego regionu lub tekstu. Dodawaj referencje do galerii,
  dopasowuj podobieństwem cosinusowym i wyszukuj z Pythona albo CLI.
lead: >-
  Jedno zadanie obejmuje każdy wektor tworzony przez LibreYOLO. embed zwraca
  wiersze float32 o długości jednostkowej, których iloczyn skalarny jest miarą
  podobieństwa, niezależnie od tego, czy wiersz opisuje cały obraz, pojedynczą
  wykrytą twarz czy wiersz tekstu. Ta sama klasa Gallery dopasowuje je
  wszystkie.
keywords:
  - embeddingi obrazów python
  - embedding normalizacja l2
  - wyszukiwanie podobieństwo cosinusowe
  - libreyolo embed
  - wyszukiwanie obrazem
  - galeria embeddingów
  - clip embeddings
  - dinov2 embeddings
  - reid embeddings
last_verified: 1.5.0
verification: >-
  Klucz zadania i aliasy odczytano z libreyolo/tasks.py. Struktury wyników
  pochodzą z klas Embeddings i Identities w libreyolo/utils/results.py. API
  Gallery pochodzi z libreyolo/utils/gallery.py. Funkcje embed i
  _postprocess_embeddings pochodzą z libreyolo/models/base/model.py. Obsługiwane
  rodziny znaleziono przez wyszukanie embed w SUPPORTED_TASKS w plikach
  libreyolo/models/**/model.py. Powierzchnia CLI pochodzi z
  libreyolo/cli/__init__.py, libreyolo/cli/commands/special.py i
  libreyolo/cli/commands/predict.py. Założenia projektowe pochodzą z
  docs/adr/0015-embed-generalization.md.
meta:
  - label: Klucz zadania
    value: embed
    mono: true
  - label: Aliasy
    value: 'face-recognition, reid, face'
    mono: true
  - label: Struktury wyników
    value: 'Embeddings, Identities'
    mono: true
  - label: Typ wiersza
    value: 'float32, długość jednostkowa'
snippets:
  predict:
    - label: Cały obraz
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Domyślnym zadaniem CLIP jest classify, dlatego jawnie wybierz wektor.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)  # (1, 512), jeden wiersz na obraz
        print(result.boxes)                  # None: niczego nie zlokalizowano
    - label: Każdy region
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        # Wiersz i opisuje region w ramce i.
        print(result.boxes.xyxy.shape)       # (N, 4)
        print(result.embeddings.data.shape)  # (N, 512)
    - label: Wiele obrazów jednocześnie
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Każdy wiersz z każdego wyniku połączony w jeden tensor.
        vectors = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(vectors.shape)  # (3, 384)
    - label: Tekst
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        # Tekst jest metodą, nigdy źródłem predykcji. Ciąg przekazany do
        # model(...) nadal oznacza ścieżkę lub adres URL.
        text = model.embed_text(["a photo of a cat", "a photo of a dog"])
        print(text.shape)  # (2, 512)
  similarity:
    - label: Porównanie dwóch zestawów wierszy
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")


        query = model.embed("query.jpg")          # (1, 512)

        pool = model.embed(["a.jpg", "b.jpg"])    # (2, 512)


        # Wiersze mają długość jednostkową, więc podobieństwo cosinusowe jest
        iloczynem skalarnym.

        scores = model("query.jpg").embeddings.similarity(pool)

        print(scores.shape)  # (1, 2)
    - label: Obraz względem tekstu
      language: python
      code: |
        import torch

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image = model.embed("photo.jpg")                       # (1, 512)
        text = model.embed_text(["a cat", "a dog", "a car"])   # (3, 512)

        print(torch.matmul(image, text.T))
  gallery:
    - label: Rejestracja i identyfikacja
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("refs.npz")

        result = model("group.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # poniżej progu name ma wartość None
    - label: Wyszukiwanie top-k
      language: python
      code: |
        from libreyolo import Gallery
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        gallery = Gallery.load("refs.npz", model=model)

        result = model("query.jpg")
        matches = gallery.match(result.embeddings, top_k=5, threshold=0.4)
        print(matches[0])   # [(name, score), ...] dla pierwszego wiersza
    - label: Rejestracja istniejącego wektora
      language: python
      code: |
        from libreyolo import Gallery

        gallery = Gallery()
        gallery.enroll_embedding("ada", vector)  # normalizowany przy dodawaniu
        print(gallery.identities, gallery.dim, len(gallery))
  cli:
    - label: Rejestracja drzewa folderów
      language: bash
      code: >
        # source/<identity>/*.jpg. Istniejąca galeria jest rozszerzana w
        miejscu.

        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=refs.npz
    - label: Identyfikacja podczas predykcji
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=group.jpg \
          gallery=refs.npz gallery_threshold=0.45
    - label: Porównanie dwóch obrazów
      language: bash
      code: >
        libreyolo compare model=librefacerec-l.onnx \
          source=a.jpg source2=b.jpg threshold=0.4

        # verify jest tym samym poleceniem dostępnym pod drugą nazwą.

        libreyolo verify model=librefacerec-l.onnx source=a.jpg source2=b.jpg
        --json
source_hash: ffbaad5599035bc7
---

## Definicja

`embed` przekształca obraz, region obrazu lub ciąg znaków w wiersz float32 o
stałej szerokości i długości równej jeden. Ponieważ każdy wiersz jest wektorem
jednostkowym, dwa wektory porównuje się iloczynem skalarnym, a dwa ich zestawy
jednym mnożeniem macierzy. Żaden inny element zadania nie zależy od modelu.
Wyszukiwanie, wykrywanie duplikatów, ponowna identyfikacja i rozpoznawanie twarzy
to te same działania arytmetyczne wykonywane na innych wierszach.

Wektor jest wynikiem. Nie ma listy klas, dlatego nazwa jest przypisywana później
przez porównanie z dostarczonymi referencjami, a nie na podstawie elementu,
którego przewidywanie wytrenowano w sieci.

### Trzy kształty

| Kształt | `Results.embeddings` | `Results.boxes` | Źródło |
|---|---|---|---|
| Cały obraz | `(1, D)` | `None` | Przekazanie obrazu do rodziny obsługującej cały obraz |
| Region | `(N, D)` | `(N, 4)`, wiersze wyrównane | Rodziny wykonujące najpierw lokalizację, na przykład rozpoznawanie twarzy |
| Tekst | nie jest obiektem `Results` | | `model.embed_text(texts)`, zwraca `(M, D)` |

Wynik całego obrazu pozostaje dwuwymiarowy nawet w przypadku jednego obrazu.
Kształt `(D,)` nie jest dozwolony, więc kod korzystający z wyniku nie musi
osobno obsługiwać przypadku pojedynczego wiersza. Tekst zwraca zwykły tensor
zamiast `Results`, ponieważ ciąg znaków nie jest źródłem obrazu. Przekazanie go
do `model(...)` nadal oznacza ścieżkę lub adres URL, a biblioteka nigdy nie
zgaduje, że ciąg znaków jest prozą.

Kanoniczny klucz zadania to `embed`. Wartości `embedding`, `embeddings`,
`face-recognition`, `facial-recognition`, `recognition`, `face`, `faceid` i
`reid` są do niego normalizowane, więc `task="reid"` i `task="embed"` wybierają
dokładnie to samo zadanie.

## Modele

Zadanie obsługują cztery rodziny. Wyraźnie dzielą się według tego, czy najpierw
lokalizują jakiś element.

| Rodzina | Kształt | Wymiar | Obsługuje również |
|---|---|---|---|
| [LibreFaceRec](/docs/models/librefacerec) | Region, jeden wiersz na wykrytą twarz | 512 | Nic; `embed` jest jedynym zadaniem |
| [CLIP](/docs/models/clip) | Cały obraz, ze sparowaną częścią tekstową | 512 dla `b32` i `b16`, 768 dla `l14` | `classify`, które pozostaje zadaniem domyślnym |
| [SigLIP 2](/docs/models/siglip2) | Cały obraz, ze sparowaną częścią tekstową | 768 dla `b16`, 1152 dla `so400m` | `classify`, które pozostaje zadaniem domyślnym |
| [DINOv2](/docs/models/dinov2) | Cały obraz, tylko obraz | 384 | `semantic`, `classify` |

CLIP i SigLIP 2 zachowują `classify` jako zadanie domyślne, dlatego trzeba
jawnie wybrać `task="embed"`. Ich istniejący checkpoint `-cls` jest wspólnym
artefaktem z dwiema częściami. Nie publikuje się zduplikowanego checkpointu
`-embed` dla identycznych wag.

Funkcja `embed_text` istnieje wyłącznie w CLIP i SigLIP 2, czyli dwóch rodzinach
z częścią tekstową. DINOv2 jej nie ma. Embedding DINOv2 pomija głowice
semantyczne i klasyfikacyjne oraz odczytuje końcowy znormalizowany token CLS przy
224 pikselach. Warianty `n`, `s`, `m` i `l` korzystają z tego samego enkodera
DINOv2-S, dlatego wszystkie zwracają `D = 384`.

Backbone wyłącznie klasyfikacyjne dodane w tym wydaniu, [ViT](/docs/models/vit),
[Swin](/docs/models/swin) i [DeiT](/docs/models/deit), deklarują tylko `classify`
i nie obsługują tego zadania.

<code-tabs name="predict" />

`model.embed(source, **kwargs)` jest skrótem dla batcha. Uruchamia `predict` i
łączy każdy wiersz z każdego wyniku w jeden tensor float32 CPU o kształcie
`(N_total, D)`, zgłaszając wyjątek, jeśli wiersze mają różne wymiary. Rodzina,
której obsługiwane zadania nie obejmują `embed`, zgłasza `NotImplementedError`.

## Struktury wyników

`result.embeddings` jest strukturą `Embeddings`. Jej `data` ma zawsze kształt
`(N, D)`, typ float32 i jest już znormalizowane metodą L2 przez ścieżkę
inferencji. Dane wejściowe, które nie są dwuwymiarowe, powodują zgłoszenie
wyjątku zamiast cichej zmiany kształtu.

| Element | Znaczenie |
|---|---|
| `.data` | Macierz `(N, D)` |
| `.dim` | `D` |
| `.normalized` | Te same wiersze ponownie normalizowane zapobiegawczo |
| `.similarity(other)` | `(N, M)` względem innego zestawu lub `(N,)` względem pojedynczego wektora `(D,)` |
| `.verify(i, j, threshold=0.4)` | Czy wiersze `i` i `j` przedstawiają ten sam obiekt |

`result.identities` jest strukturą `Identities`, obecną tylko po przekazaniu
galerii. Jest zwykłym kontenerem, a nie tensorem, więc przenoszenie obiektu
`Results` między urządzeniami pozostawia ją bez zmian.

| Element | Znaczenie |
|---|---|
| `.name` | Lista nazw, z `None` tam, gdzie żaden wynik nie przekroczył progu |
| `.score` | Najlepszy wskaźnik cosinusowy float32 o kształcie `(N,)`, zachowywany nawet przy nazwie `None` |
| `.data` | Lista krotek `(name, score)` |

<code-tabs name="similarity" />

Wektory są domyślnie pomijane w `summary()` i `to_json()`, ponieważ wiersz z
512 wartościami float zajmuje około dwóch kilobajtów na obiekt. Każdy wiersz
zamiast tego zgłasza `embedding_dim`, a po użyciu galerii także `identity` i
`identity_score`. Aby uwzględnić liczby, przekaż `summary(embeddings=True)`.

## Galerie

`Gallery` jest nazwanym zestawem wierszy referencyjnych. Przechowuje każdą
referencję osobno zamiast je uśredniać, dlatego nazwa jest oceniana na podstawie
jednej najlepiej dopasowanej referencji, a dodanie złego zdjęcia nie przesuwa
centroidu tożsamości.

<code-tabs name="gallery" />

`Gallery(model)` wiąże galerię z wagami, które będą tworzyć jej wektory.
`enroll(name, sources, select="best")` uruchamia predykcję na każdym źródle i
zachowuje wiersz o największej pewności z każdego wyniku. `select="all"`
zachowuje zamiast tego każdy wiersz, co jest właściwym wyborem, gdy obraz
referencyjny faktycznie zawiera kilka obiektów. `enroll_embedding(name, vector)`
pomija inferencję i bezpośrednio przyjmuje wektor, normalizując go i odrzucając
wiersz złożony wyłącznie z zer.

`FaceGallery` jest trwałym aliasem tej samej klasy, a archiwa utworzone przez
starsze wydania obsługujące wyłącznie twarze nadal można wczytać.

### Dopasowanie i progi

Dopasowanie jest gęstym mnożeniem macierzy względem każdej zapisanej referencji,
redukowanym do jednego wskaźnika na nazwę przez wybranie maksimum. Nie ma
indeksu przybliżonego, co zachowuje dokładne wyniki i nakłada praktyczne
ograniczenie na rozmiar galerii.

Dwa punkty wejścia różnią się działaniem poniżej progu. Funkcja `match()` zwraca
`[(name, score), ...]` dla każdego wiersza i odrzuca wszystko poniżej progu,
dlatego wiersz bez dopasowania otrzymuje pustą listę. Funkcja `identify()` zwraca
strukturę `Identities`, która zawsze zachowuje najlepszy wskaźnik i ustawia
nazwę na `None`, gdy znajduje się on poniżej progu. Żadna z nich nigdy nie
podstawia najbliższej nazwy niespełniającej progu.

Domyślny próg wynosi wszędzie `0.4`. Jest to wartość cosinusowa, a nie
prawdopodobieństwo. Właściwy punkt pracy jest cechą danych i tolerancji na
fałszywe dopasowania, dlatego zamiast przyjmować wartość domyślną należy
przetestować zakres progów na oznaczonych parach. Polecenie `libreyolo enroll`
i argument predykcji `gallery=` używają tej samej wartości.

### Trwały zapis

Funkcja `save(path)` zapisuje skompresowany plik `.npz` zawierający wektory,
nazwy i blok metadanych z wersją formatu, wymiarem embeddingu oraz odciskiem wag,
które utworzyły wiersze. `Gallery.load(path, model=...)` sprawdza oba elementy
przed jakimkolwiek porównaniem, dlatego wskazanie galerii innemu modelowi
zgłasza wyjątek zamiast po cichu porównywać wektory z dwóch niezwiązanych
przestrzeni. Nie można zapisać pustej galerii.

## Wiersz poleceń

| Polecenie | Cel |
|---|---|
| `libreyolo enroll` | Przejście przez drzewo z jednym folderem na tożsamość oraz zapisanie lub rozszerzenie galerii `.npz` |
| `libreyolo compare` | Utworzenie embeddingu głównego obiektu na dwóch obrazach i zgłoszenie podobieństwa cosinusowego |
| `libreyolo verify` | To samo polecenie dostępne pod drugą nazwą |
| `libreyolo predict gallery=...` | Dołączenie tożsamości do zwykłego uruchomienia predykcji |

<code-tabs name="cli" />

Każde polecenie LibreYOLO przyjmuje zarówno postać `key=value`, jak i
`--key value`, dlatego `gallery=refs.npz` oraz `--gallery refs.npz` są tym samym
argumentem.

Polecenie `enroll` przyjmuje `model`, `source` i `gallery`, a opcjonalnie także
`face-detector`, `device`, `--json` i `--quiet`. Odczytuje po jednym folderze na
tożsamość. Nazwa folderu jest tożsamością, a każdy obraz w jego wnętrzu wnosi
referencje:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

Obraz, który niczego nie zwraca, jest pomijany z komunikatem na stderr zamiast
przerwania uruchomienia, a podsumowanie podaje liczbę referencji zapisanych dla
każdej nazwy. Istniejący plik galerii jest rozszerzany w miejscu, dzięki czemu
tożsamości można dodawać z czasem.

`compare` i `verify` są jedną funkcją zarejestrowaną dwukrotnie. Przyjmują
`model`, `source`, `source2` oraz opcjonalny `threshold`, po czym wyświetlają
podobieństwo cosinusowe, wynik zgodności lub różnicy oraz próg, który dał ten
wynik. Opcja `--json` wyświetla te same trzy pola jako obiekt.

W poleceniu `predict` argument `gallery` wskazuje zapisany plik `.npz`, a
`gallery_threshold` nadpisuje domyślną wartość `0.4`. Przekazanie galerii do
modelu, którego zadaniem nie jest `embed`, powoduje błąd zamiast cichego braku
działania. Brakujący plik galerii powoduje wyświetlenie sugestii użycia
polecenia `libreyolo enroll`, które może go utworzyć.

## Twarze

Rozpoznawanie twarzy jest regionalnym kształtem tego zadania i jedyną
dostarczaną implementacją tego kształtu. Dodaje przed głowicą embeddingu etap
detekcji i wyrównania, a także metodę `verify()`, argument własnych ramek,
opublikowane wyniki dokładności i wskazówki kalibracji progu. Wszystko to
znajduje się na stronie [rozpoznawania
twarzy](/docs/tasks/face-recognition), która jest odpowiednim przewodnikiem dla
twarzy. Cała zawartość tej strony ma do niej zastosowanie bez zmian.

## Trenowanie, walidacja i eksport

Żaden element tego zadania nie jest trenowany w LibreYOLO. Głowica embeddingu
twarzy jest artefaktem ONNX, którego funkcje `train()`, `val()` i `export()`
zgłaszają wyjątek. Głowicę należy wytrenować w projekcie źródłowym i wczytać
plik ze ścieżki. CLIP, SigLIP 2 i DINOv2 obsługują trenowanie i eksport przez
zadania klasyfikacji i segmentacji, a nie przez `embed`.

Nie ma walidatora wyszukiwania. Dokładność weryfikacji należy mierzyć na
oznaczonych parach przez przeszukiwanie wartości `threshold`, a dokładność
identyfikacji przez dodanie referencji do galerii i odczytywanie
`identities.name` oraz `identities.score` na odłożonych obrazach. Nazwę `None`
należy liczyć jako odrzucenie.
