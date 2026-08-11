---
title: Embeddings
seo_title: Bild- und Region-Embeddings in LibreYOLO
description: >-
  Der embed-Task liefert L2-normalisierte float32-Vektoren für ein ganzes Bild,
  für jede erkannte Region oder für Text. Registriere eine Gallery, matche über
  Kosinusähnlichkeit und suche aus Python oder über die CLI.
lead: >-
  Ein Task deckt jeden Vektor ab, den LibreYOLO erzeugt. embed liefert
  float32-Zeilen der Länge eins, deren Skalarprodukt ein Ähnlichkeitswert ist,
  egal ob die Zeile ein ganzes Bild, ein einzelnes erkanntes Gesicht oder eine
  Textzeile beschreibt, und dieselbe Gallery matcht sie alle.
keywords:
  - bild embeddings python
  - l2 normalisiertes embedding
  - kosinusähnlichkeit suche
  - libreyolo embed task
  - bildersuche nach ähnlichkeit
  - gallery registrieren
  - clip embeddings
  - dinov2 embeddings
  - reid embeddings
last_verified: 1.5.0
verification: >-
  Task-Key und Aliase gelesen aus libreyolo/tasks.py. Result-Payloads aus den
  Klassen Embeddings und Identities in libreyolo/utils/results.py. Gallery-API
  aus libreyolo/utils/gallery.py. embed und _postprocess_embeddings aus
  libreyolo/models/base/model.py. Unterstützte Familien ermittelt durch die
  Suche nach embed in SUPPORTED_TASKS über libreyolo/models/**/model.py.
  CLI-Oberfläche aus libreyolo/cli/__init__.py,
  libreyolo/cli/commands/special.py und libreyolo/cli/commands/predict.py.
  Entwurfsabsicht aus docs/adr/0015-embed-generalization.md.
meta:
  - label: Task-Key
    value: embed
    mono: true
  - label: Aliase
    value: 'face-recognition, reid, face'
    mono: true
  - label: Result-Payloads
    value: 'Embeddings, Identities'
    mono: true
  - label: dtype der Zeilen
    value: 'float32, Einheitslänge'
snippets:
  predict:
    - label: Ganzes Bild
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # CLIP nutzt standardmäßig classify, fordere den Vektor explizit an.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)  # (1, 512), eine Zeile pro Bild
        print(result.boxes)                  # None: nichts wurde lokalisiert
    - label: Pro Region
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        # Zeile i beschreibt die Region in Box i.
        print(result.boxes.xyxy.shape)       # (N, 4)
        print(result.embeddings.data.shape)  # (N, 512)
    - label: Viele Bilder auf einmal
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Jede Zeile jedes Ergebnisses, zu einem Tensor verkettet.
        vectors = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(vectors.shape)  # (3, 384)
    - label: Text
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        # Text ist eine Methode, nie eine Vorhersagequelle. Ein String an
        # model(...) ist weiterhin ein Pfad oder eine URL.
        text = model.embed_text(["a photo of a cat", "a photo of a dog"])
        print(text.shape)  # (2, 512)
  similarity:
    - label: Zwei Zeilenmengen vergleichen
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        query = model.embed("query.jpg")          # (1, 512)
        pool = model.embed(["a.jpg", "b.jpg"])    # (2, 512)

        # Zeilen haben Länge eins, Kosinus ist also ein Skalarprodukt.
        scores = model("query.jpg").embeddings.similarity(pool)
        print(scores.shape)  # (1, 2)
    - label: Bild gegen Text
      language: python
      code: |
        import torch

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image = model.embed("photo.jpg")                       # (1, 512)
        text = model.embed_text(["a cat", "a dog", "a car"])   # (3, 512)

        print(torch.matmul(image, text.T))
  gallery:
    - label: Registrieren und identifizieren
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
            print(name, score)   # Name ist None unter dem Schwellenwert
    - label: Top-k-Suche
      language: python
      code: |
        from libreyolo import Gallery
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        gallery = Gallery.load("refs.npz", model=model)

        result = model("query.jpg")
        matches = gallery.match(result.embeddings, top_k=5, threshold=0.4)
        print(matches[0])   # [(name, score), ...] für die erste Zeile
    - label: Einen vorhandenen Vektor registrieren
      language: python
      code: |
        from libreyolo import Gallery

        gallery = Gallery()
        gallery.enroll_embedding("ada", vector)  # beim Eintragen normalisiert
        print(gallery.identities, gallery.dim, len(gallery))
  cli:
    - label: Einen Ordnerbaum registrieren
      language: bash
      code: >
        # source/<identity>/*.jpg. Eine vorhandene Gallery wird ergänzt.

        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=refs.npz
    - label: Beim Vorhersagen identifizieren
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=group.jpg \
          gallery=refs.npz gallery_threshold=0.45
    - label: Zwei Bilder vergleichen
      language: bash
      code: >
        libreyolo compare model=librefacerec-l.onnx \
          source=a.jpg source2=b.jpg threshold=0.4

        # verify ist derselbe Befehl unter einem zweiten Namen.

        libreyolo verify model=librefacerec-l.onnx source=a.jpg source2=b.jpg
        --json
source_hash: ffbaad5599035bc7
---

## Definition

`embed` macht aus einem Bild, einem Bildausschnitt oder einer Zeichenkette eine
float32-Zeile fester Breite, deren Länge eins ist. Weil jede Zeile ein
Einheitsvektor ist, ist der Vergleich zweier Zeilen ein Skalarprodukt und der
Vergleich zweier Mengen davon eine einzige Matrixmultiplikation. Sonst ist
nichts an diesem Task modellspezifisch: Retrieval, Duplikaterkennung,
Re-Identifikation und Gesichtserkennung sind alle dieselbe Arithmetik über
unterschiedlichen Zeilen.

Der Vektor ist die Ausgabe. Es gibt keine Klassenliste, ein Name wird also erst
später angehängt, indem gegen Referenzen verglichen wird, die du bereitstellst,
und nicht durch irgendetwas, das das Netz vorherzusagen gelernt hat.

### Drei Formen

| Form | `Results.embeddings` | `Results.boxes` | Erzeugt durch |
|---|---|---|---|
| Ganzes Bild | `(1, D)` | `None` | Ein Bild an eine Ganzbild-Familie übergeben |
| Region | `(N, D)` | `(N, 4)`, zeilenweise ausgerichtet | Familien, die zuerst lokalisieren, etwa die Gesichtserkennung |
| Text | überhaupt kein `Results` | | `model.embed_text(texts)`, liefert `(M, D)` |

Ein Ganzbild-Ergebnis bleibt auch bei einem einzelnen Bild zweidimensional.
`(D,)` ist keine erlaubte Rückgabeform, deshalb muss ein Konsument den Fall der
einzelnen Zeile nie gesondert behandeln. Text liefert einen schlichten Tensor
statt eines `Results`, weil eine Zeichenkette keine Bildquelle ist: Sie an
`model(...)` zu übergeben bedeutet weiterhin einen Pfad oder eine URL, und die
Bibliothek rät nie, dass eine Zeichenkette Prosa ist.

Der kanonische Task-Key ist `embed`. `embedding`, `embeddings`,
`face-recognition`, `facial-recognition`, `recognition`, `face`, `faceid` und
`reid` normalisieren alle darauf, `task="reid"` und `task="embed"` wählen also
genau dasselbe aus.

## Modelle

Vier Familien bedienen den Task, und sie teilen sich sauber danach, ob sie
vorher etwas lokalisieren.

| Familie | Form | Dimension | Unterstützt außerdem |
|---|---|---|---|
| [LibreFaceRec](/docs/models/librefacerec) | Region, eine Zeile pro erkanntem Gesicht | 512 | Nichts; `embed` ist ihr einziger Task |
| [CLIP](/docs/models/clip) | Ganzes Bild, mit gepaartem Text-Tower | 512 bei `b32` und `b16`, 768 bei `l14` | `classify`, was ihr Standard bleibt |
| [SigLIP 2](/docs/models/siglip2) | Ganzes Bild, mit gepaartem Text-Tower | 768 bei `b16`, 1152 bei `so400m` | `classify`, was ihr Standard bleibt |
| [DINOv2](/docs/models/dinov2) | Ganzes Bild, nur Bild | 384 | `semantic`, `classify` |

CLIP und SigLIP 2 behalten `classify` als ihren Standard-Task, `task="embed"`
muss also angefordert werden. Ihr vorhandener `-cls`-Checkpoint ist das
gemeinsame Zwei-Tower-Artefakt; für identische Gewichte wird kein doppelter
`-embed`-Checkpoint veröffentlicht.

`embed_text` gibt es nur bei CLIP und SigLIP 2, den beiden Familien mit einem
Text-Tower. DINOv2 hat keinen. Das Embedding von DINOv2 umgeht die Heads für
Semantik und Klassifikation und liest das finale normalisierte CLS-Token bei
224 Pixeln; die Varianten `n`, `s`, `m` und `l` teilen sich alle den
DINOv2-S-Encoder, deshalb liefern alle vier `D = 384`.

Die reinen Klassifikations-Backbones, die in diesem Release dazugekommen sind,
[ViT](/docs/models/vit), [Swin](/docs/models/swin) und [DeiT](/docs/models/deit),
deklarieren nur `classify` und bedienen diesen Task nicht.

<code-tabs name="predict" />

`model.embed(source, **kwargs)` ist die Batch-Abkürzung: Sie führt `predict` aus
und verkettet jede Zeile jedes Ergebnisses zu einem einzigen
`(N_total, D)`-float32-Tensor auf der CPU und löst einen Fehler aus, wenn die
Zeilen gemischte Dimensionen haben. Eine Familie ohne `embed` in ihren
unterstützten Aufgaben löst `NotImplementedError` aus.

## Result-Payloads

`result.embeddings` ist ein `Embeddings`-Payload. Sein `data` ist immer
`(N, D)`-float32, vom Inferenzpfad bereits L2-normalisiert, und eine nicht
zweidimensionale Eingabe löst einen Fehler aus, statt still umgeformt zu werden.

| Member | Bedeutung |
|---|---|
| `.data` | Die `(N, D)`-Matrix |
| `.dim` | `D` |
| `.normalized` | Dieselben Zeilen, vorsorglich neu normalisiert |
| `.similarity(other)` | `(N, M)` gegen eine andere Menge oder `(N,)` gegen einen einzelnen `(D,)`-Vektor |
| `.verify(i, j, threshold=0.4)` | Ob die Zeilen `i` und `j` dasselbe Subjekt zeigen |

`result.identities` ist ein `Identities`-Payload und nur dann vorhanden, wenn
eine Gallery übergeben wurde. Es ist ein schlichter Container, kein Tensor,
deshalb bleibt es unberührt, wenn ein `Results` zwischen Geräten wandert.

| Member | Bedeutung |
|---|---|
| `.name` | Liste der Namen, `None`, wo nichts den Schwellenwert überschritten hat |
| `.score` | `(N,)` float32, bester Kosinus-Score, auch dann behalten, wenn der Name `None` ist |
| `.data` | Liste von `(name, score)`-Tupeln |

<code-tabs name="similarity" />

Vektoren bleiben in `summary()` und `to_json()` standardmäßig außen vor, da eine
Zeile aus 512 Floats rund zwei Kilobyte pro Subjekt ausmacht. Jede Zeile meldet
stattdessen `embedding_dim`, plus `identity` und `identity_score`, wenn eine
Gallery genutzt wurde. Übergib `summary(embeddings=True)`, um die Zahlen
aufzunehmen.

## Galleries

Eine `Gallery` ist eine benannte Menge von Referenzzeilen. Sie speichert jede
Referenz einzeln, statt sie zu mitteln, deshalb wird ein Name über seine einzige
beste passende Referenz bewertet, und ein schlechtes Foto kann den Schwerpunkt
einer Identität nicht verschieben.

<code-tabs name="gallery" />

`Gallery(model)` bindet sich an die Gewichte, die ihre Vektoren erzeugen werden.
`enroll(name, sources, select="best")` führt auf jeder Quelle eine Vorhersage
aus und behält pro Ergebnis die Zeile mit der höchsten Confidence;
`select="all"` behält stattdessen jede Zeile, was du willst, wenn ein
Referenzbild berechtigterweise mehrere Subjekte enthält.
`enroll_embedding(name, vector)` überspringt die Inferenz und nimmt einen Vektor
direkt entgegen, normalisiert ihn und weist eine Zeile aus lauter Nullen zurück.

`FaceGallery` ist ein dauerhafter Alias derselben Klasse, und Archive aus
früheren, rein gesichtsbezogenen Releases laden weiterhin.

### Matching und Schwellenwerte

Das Matching ist eine dichte Matrixmultiplikation gegen jede gespeicherte
Referenz, über das Maximum auf einen Score pro Name reduziert. Es gibt keinen
approximativen Index, was die Zahlen exakt hält und der Größe einer Gallery eine
praktische Obergrenze setzt.

Zwei Einstiegspunkte unterscheiden sich darin, was sie unterhalb des
Schwellenwerts tun. `match()` liefert pro Zeile `[(name, score), ...]`, wobei
alles unter dem Schwellenwert wegfällt, eine Zeile ohne Treffer ist also eine
leere Liste. `identify()` liefert ein `Identities`-Payload, das immer den besten
Score behält und den Namen auf `None` setzt, wenn er unter dem Schwellenwert
liegt. Keines von beiden schiebt je den nächstliegenden Namen unterhalb des
Schwellenwerts unter.

Der Standard-Schwellenwert ist überall `0.4`. Er ist ein Kosinuswert, keine
Wahrscheinlichkeit, und der richtige Arbeitspunkt ist eine Eigenschaft deiner
Daten und deiner Toleranz für Fehltreffer, teste ihn also über gelabelte Paare
hinweg, statt den Standard zu übernehmen. `libreyolo enroll` und das
Vorhersage-Argument `gallery=` nutzen dieselbe Zahl.

### Persistenz

`save(path)` schreibt ein komprimiertes `.npz` mit den Vektoren, den Namen und
einem Metadatenblock, der die Formatversion, die Embedding-Dimension und einen
Fingerabdruck der Gewichte trägt, die die Zeilen erzeugt haben.
`Gallery.load(path, model=...)` prüft beides, bevor irgendetwas verglichen wird,
deshalb löst eine Gallery, die auf ein anderes Modell zeigt, einen Fehler aus,
statt still Vektoren aus zwei unverwandten Räumen gegeneinander zu bewerten.
Eine leere Gallery zu speichern wird verweigert.

## Kommandozeile

| Befehl | Zweck |
|---|---|
| `libreyolo enroll` | Einen Baum mit einem Ordner pro Identität durchlaufen und eine `.npz`-Gallery schreiben oder ergänzen |
| `libreyolo compare` | Das Hauptmotiv in zwei Bildern einbetten und die Kosinusähnlichkeit melden |
| `libreyolo verify` | Derselbe Befehl unter einem zweiten Namen |
| `libreyolo predict gallery=...` | Identitäten an einen gewöhnlichen Vorhersagelauf anhängen |

<code-tabs name="cli" />

Jeder LibreYOLO-Befehl akzeptiert sowohl `key=value` als auch `--key value`,
`gallery=refs.npz` und `--gallery refs.npz` sind also dasselbe Argument.

`enroll` nimmt `model`, `source` und `gallery`, dazu optional `face-detector`,
`device`, `--json` und `--quiet`. Es liest einen Ordner pro Identität, wobei der
Ordnername die Identität ist und jedes Bild darin Referenzen beisteuert:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

Ein Bild, das nichts liefert, wird mit einer Zeile auf stderr übersprungen,
statt den Lauf abzubrechen, und die Zusammenfassung meldet, wie viele Referenzen
für jeden Namen gespeichert wurden. Eine vorhandene Gallery-Datei wird an Ort
und Stelle ergänzt, Identitäten können also über die Zeit dazukommen.

`compare` und `verify` sind eine Funktion, die zweimal registriert ist. Sie
nehmen `model`, `source`, `source2` und optional ein `threshold` und geben die
Kosinusähnlichkeit aus, das Urteil gleich oder verschieden und den
Schwellenwert, der es erzeugt hat. `--json` gibt dieselben drei Felder als
Objekt aus.

Bei `predict` zeigt `gallery` auf ein gespeichertes `.npz`, und
`gallery_threshold` überschreibt den Standard `0.4`. Einem Modell, dessen Task
nicht `embed` ist, eine Gallery zu übergeben ist ein Fehler und kein stiller
Leerlauf, und eine fehlende Gallery-Datei schlägt den Befehl `libreyolo enroll`
vor, der sie anlegen würde.

## Gesichter

Die Gesichtserkennung ist die Region-Form dieses Tasks, und sie ist die einzige
ausgelieferte Implementierung dieser Form. Sie ergänzt eine Erkennungs- und
Ausrichtungsstufe vor dem Embedding-Head, dazu eine `verify()`-Methode, ein
Argument für eigene Boxen, veröffentlichte Accuracy-Zahlen und eine Anleitung
zur Kalibrierung des Schwellenwerts. All das steht unter
[Gesichtserkennung](/docs/tasks/face-recognition), der Anleitung, der du folgen
solltest, wenn es um Gesichter geht. Alles auf dieser Seite gilt dort unverändert.

## Training, Validierung und Export

Nichts in diesem Task trainiert innerhalb von LibreYOLO. Der Head für
Gesichts-Embeddings ist ein ONNX-Artefakt, dessen `train()`, `val()` und
`export()` alle einen Fehler auslösen; trainiere einen Head upstream und lade die
Datei über ihren Pfad. CLIP, SigLIP 2 und DINOv2 trainieren und exportieren über
ihre Klassifikations- und Segmentierungsaufgaben, nicht über `embed`.

Es gibt keinen Retrieval-Validator. Miss die Verifikations-Accuracy an gelabelten
Paaren, indem du `threshold` durchfährst, und die Identifikations-Accuracy,
indem du eine Gallery registrierst und auf zurückgehaltenen Bildern
`identities.name` und `identities.score` liest, wobei ein Name `None` als
Zurückweisung zählt.
