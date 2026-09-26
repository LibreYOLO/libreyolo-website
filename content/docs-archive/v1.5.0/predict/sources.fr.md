---
title: Sources de prédiction
seo_title: Sources de prédiction dans LibreYOLO
description: "Toutes les sources acceptées par predict\_: images, dossiers, URL, fichiers vidéo, webcams, RTSP, YouTube, capture d'écran, listes d'images et fichiers .streams."
lead: >-
  L'argument source est classé avant toute ouverture. Un même appel traite donc
  un fichier JPEG, un dossier, un fichier MP4, un indice de webcam, une URL
  RTSP, une région de l'écran ou une liste de caméras.
keywords:
  - yolo inférence vidéo python
  - rtsp
  - webcam détection objets python
  - prédire dossier images
  - capture écran détection objets
  - plusieurs flux rtsp
  - fichier streams
  - inférence youtube
  - vid_stride
  - stream=True
last_verified: 1.5.0
verification: >-
  Classification des sources lue dans libreyolo/utils/source.py
  (classify_source, SourceKind, StreamSource, MultiStreamSource). Types d'images
  acceptés et extensions des répertoires lus dans
  libreyolo/utils/image_loader.py. Extensions vidéo et chemins d'enregistrement
  lus dans libreyolo/utils/video.py. Syntaxe de capture d'écran lue dans
  libreyolo/utils/screen.py. Formes de retour et valeurs par défaut des
  arguments lues dans InferenceRunner.__call__ dans
  libreyolo/models/base/inference.py.
snippets:
  images:
    - label: Une image
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Une source image unique renvoie un objet Results, et non une liste.
        result = model(SAMPLE_IMAGE)
        print(len(result.boxes), "detections")
    - label: Images en mémoire
      language: python
      code: |
        import numpy as np
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        pil_image = Image.open(SAMPLE_IMAGE)
        array = np.asarray(pil_image)
        raw_bytes = open(SAMPLE_IMAGE, "rb").read()

        for source in (pil_image, array, raw_bytes):
            result = model(source)
            print(type(source).__name__, len(result.boxes))
    - label: Un dossier
      language: python
      code: >
        from pathlib import Path

        from PIL import Image


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        folder = Path("sample_folder")

        folder.mkdir(exist_ok=True)

        image = Image.open(SAMPLE_IMAGE)

        for index in range(3):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")


        # Un dossier renvoie une liste, avec un objet Results par image, triée
        par chemin.

        results = model(str(folder))

        print(len(results), "images")
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  video:
    - label: Un fichier vidéo (fournissez votre propre clip)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Remplacez clip.mp4 par un fichier vidéo présent sur le disque.
        for result in model("clip.mp4", stream=True):
            print(result.frame_idx, len(result.boxes))
    - label: 'Une image sur trois, écrite sur le disque'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("clip.mp4", stream=True, vid_stride=3, save=True):
            pass
  live:
    - label: Webcam (une caméra doit être connectée)
      language: python
      code: >
        import itertools


        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Webcam d'indice 0. Les sources en direct ne se terminent jamais,
        limitez donc la boucle.

        for result in itertools.islice(model(0, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
    - label: RTSP (nécessite une URL de caméra accessible)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        source = "rtsp://user:password@192.168.1.64:554/Streaming/Channels/101"

        for result in itertools.islice(model(source, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  streams:
    - label: Un fichier .streams (fournissez vos propres caméras)
      language: python
      code: >
        import itertools

        from pathlib import Path


        from libreyolo import LibreYOLO


        Path("cameras.streams").write_text(
            "# one source per line, blank lines and comments are skipped\n"
            "rtsp://192.168.1.64:554/Streaming/Channels/101\n"
            "rtsp://192.168.1.65:554/Streaming/Channels/101\n",
            encoding="utf-8",
        )


        model = LibreYOLO("LibreYOLO9s.pt")

        for result in itertools.islice(model("cameras.streams", stream=True),
        100):
            print(result.frame_idx, len(result.boxes))
    - label: Une liste de caméras
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        cameras = [0, "rtsp://192.168.1.64:554/Streaming/Channels/101"]

        for result in itertools.islice(model(cameras, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  screen:
    - label: Une capture d'écran (nécessite mss et une session de bureau)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Sans stream=True, une seule image est capturée.
        result = model("screen")
        print(len(result.boxes), "detections")
    - label: 'Une région d''un écran, en continu'
      language: python
      code: >
        import itertools


        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # "screen <monitor> <left> <top> <width> <height>"

        for result in itertools.islice(model("screen 1 100 200 512 256",
        stream=True), 50):
            print(len(result.boxes))
source_hash: c371965951dd0181
---

## Méthode de classification d'une source

`classify_source` inspecte la valeur avant toute ouverture ou tout
téléchargement, dans l'ordre suivant. La première règle correspondante gagne.

| Source | Interprétation |
|---|---|
| `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` | Capture d'écran |
| Un `int` positif ou nul, ou une chaîne de chiffres sans fichier portant ce nom | Webcam |
| Une URL `rtsp://`, `rtmp://`, `tcp://` ou `udp://` | Flux réseau |
| Une URL `http(s)://` dont le chemin se termine par `.m3u8` | Flux réseau |
| Une URL de page YouTube | Flux réseau |
| Une liste ou un tuple dont toutes les entrées sont en direct ou vidéo | Plusieurs flux en direct |
| Toute autre liste ou tout autre tuple | Batch d'images |
| Un chemin se terminant par `.streams` | Plusieurs flux en direct |
| Un chemin avec une extension vidéo | Fichier vidéo |
| Un répertoire existant | Dossier d'images |
| Toute autre valeur | Image unique |

Une liste qui mélange des sources en direct et des images lève `TypeError`. Un
indice de webcam négatif lève `ValueError`.

Le classificateur n'accède jamais au réseau. Une URL erronée ne produit donc
une erreur qu'à l'ouverture de la capture, et non lors de l'appel à `predict`.

## Images

<code-tabs name="images" />

Une source image unique accepte sept types.

| Type | Interprétation |
|---|---|
| `str` ou `pathlib.Path` | Fichier local, `http(s)://`, `s3://` ou `gs://` |
| `PIL.Image.Image` | Convertie en RGB |
| `numpy.ndarray` | Nuances de gris 2D, ou HWC ou CHW 3D\u00a0; un tableau 4D utilise sa première image |
| `torch.Tensor` | CHW ou NCHW, lu comme RGB\u00a0; un tenseur par batch utilise sa première image |
| `bytes` | Données d'image encodées |
| `io.BytesIO` | Données d'image encodées |

Tout est converti en RGB avant le prétraitement. Les tableaux NumPy sont le
seul cas où l'ordre des canaux est ambigu, `color_format` le contrôle donc\u00a0:
`"auto"` (valeur par défaut) conserve le tableau tel quel, tandis que `"bgr"`
inverse les canaux, comme l'exige une image lue avec OpenCV.

Les tableaux de flottants sont remis à l'échelle selon leur propre plage\u00a0: les
valeurs inférieures ou égales à `1.0` sont multipliées par 255, tandis que les
valeurs supérieures sont écrêtées dans `[0, 255]`. Un tableau RGBA perd son
canal alpha.

Chaque type de chemin distant nécessite un package, dont aucun n'est installé
par défaut\u00a0: `requests` pour `http(s)://`, `boto3` pour `s3://` et `gcsfs`
pour `gs://`.

## Dossiers

Un répertoire est parcouru récursivement et trié. Chaque fichier portant l'un
des suffixes suivants devient une image\u00a0: `.jpg`, `.jpeg`, `.png`, `.gif`,
`.webp`, `.bmp`, `.tiff`, `.tif`. Tout autre fichier du dossier est ignoré. Un
dossier vide renvoie une liste vide au lieu de lever une erreur.

Les dossiers et les listes sont les deux sources qui acceptent `batch`, lequel
exécute une passe forward empilée par groupe sur les familles compatibles.
Consultez les [performances d'inférence](/docs/predict/performance).

## Fichiers vidéo

<code-tabs name="video" />

Un chemin est considéré comme une vidéo lorsque son suffixe est `.asf`, `.avi`,
`.gif`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.mpeg`, `.mpg`, `.ts`, `.wmv` ou
`.webm`.

`.gif` figure dans les deux listes. Un chemin `.gif` transmis directement à
`predict` est ouvert comme une vidéo, car la vérification vidéo intervient en
premier. Un fichier `.gif` placé dans un dossier parcouru est chargé comme une
image fixe.

`vid_stride` traite une image sur N et vaut `1` par défaut. Sans `stream=True`,
la vidéo entière est décodée dans une liste. Au-delà de 500 images après
l'application du stride, un avertissement recommande d'utiliser `stream=True`.

Chaque objet `Results` provenant d'une vidéo contient `frame_idx`.

## Webcams, flux réseau et YouTube

<code-tabs name="live" />

Les sources en direct sont illimitées, elles nécessitent donc `stream=True`.
Sans cette option, `predict` lève `ValueError` au lieu de tenter de constituer
une liste infinie.

Les images sont lues dans un thread d'arrière-plan, un par capture. Par défaut,
la file ne contient que l'image la plus récente. Un modèle plus lent que la
caméra saute donc des images au lieu d'accumuler du retard.
`stream_buffer=True` conserve chaque image capturée, ce qui les préserve au
prix d'une latence croissante.

Un indice de webcam est un `int` ou une chaîne de chiffres. Sous Windows, la
capture est d'abord ouverte avec le backend DirectShow, puis se rabat sur le
backend par défaut en cas d'échec.

Les URL de pages YouTube sont résolues en URL de média directe sans télécharger
la vidéo, ce qui nécessite `yt-dlp`\u00a0:

```bash
pip install "libreyolo[stream]"
```

Les étiquettes de flux sont expurgées avant leur journalisation ou leur
utilisation comme noms de fichiers. Une URL contenant des identifiants apparaît
sous la forme `user:***@host`, tandis que les chaînes de requête sont retirées
des étiquettes de flux directs, car elles peuvent contenir des URL signées et
des tokens bearer. L'identifiant d'une vidéo YouTube est conservé puisqu'il ne
s'agit pas d'un identifiant secret.

## Plusieurs caméras simultanées

<code-tabs name="streams" />

Un fichier `.streams` contient une source par ligne. Les lignes vides et celles
qui commencent par `#` sont ignorées. Chaque autre ligne doit être un indice de
webcam, un flux réseau, une URL YouTube ou un chemin de fichier vidéo. Toute
autre valeur lève `ValueError` en indiquant le numéro de ligne. Un fichier vide
lève une erreur au lieu de démarrer sans caméra.

Une liste ou un tuple de sources en direct produit le même résultat sans
fichier.

Chaque capture reçoit son propre thread et les images de toutes les captures
sont multiplexées dans un générateur unique. Chaque passe interroge chaque flux
actif et produit tout élément prêt. Une caméra lente ne bloque donc pas une
caméra rapide et les images des différentes caméras s'entrelacent. Un flux qui
se termine quitte la rotation tandis que les autres continuent.

## Capture d'écran

<code-tabs name="screen" />

Une source d'écran est le mot `screen` suivi de zéro, un, quatre ou cinq
entiers. Tout autre nombre lève `ValueError`.

| Forme | Capture |
|---|---|
| `"screen"` | Tous les écrans, fusionnés |
| `"screen 1"` | Écran 1 |
| `"screen 100 200 512 256"` | Une zone du bureau fusionné |
| `"screen 1 100 200 512 256"` | Une zone de l'écran 1 |

Les coordonnées de la zone sont `left top width height`, par rapport au coin
supérieur gauche de l'écran choisi. Une source écran indique une cadence égale
à 30 divisé par `vid_stride`, utilisée pour écrire une vidéo enregistrée. La
capture nécessite le package `mss`\u00a0:

```bash
pip install mss
```

Sans `stream=True`, une source écran capture une image et renvoie un seul objet
`Results`, l'équivalent d'une prédiction sur un fichier image pour une capture
d'écran. Avec `stream=True`, la capture continue jusqu'à l'interruption de la
boucle.

## Valeur renvoyée par predict

La forme de la valeur renvoyée dépend de la source et de `stream`.

| Source | `stream=False` | `stream=True` |
|---|---|---|
| Image unique | Un objet `Results` | Générateur d'un objet `Results` |
| Liste d'images | Liste d'objets `Results` | Générateur |
| Dossier | Liste d'objets `Results` | Générateur |
| Fichier vidéo | Liste d'objets `Results` | Générateur |
| Écran | Un objet `Results` | Générateur illimité |
| Webcam, flux réseau, `.streams` | `ValueError` | Générateur illimité |

Une image unique renvoie directement l'objet `Results`. L'indexer sélectionne
une détection et non une image. `result[0]` sur une prédiction mono-image
correspond donc à la première bounding box et non à la première image. Pour
connaître le contenu de ces objets, consultez
[Travailler avec les résultats](/docs/predict/results).

## Emplacement des enregistrements

`save=True` écrit la sortie annotée à côté d'un répertoire d'exécution au lieu
de la renvoyer.

Les images sont placées dans les répertoires auto-incrémentés
`runs/detect/predict`, `runs/detect/predict2` et ainsi de suite, en conservant
le nom du fichier source. Toutes les images d'un même processus sont placées
dans le même répertoire, deux dossiers d'entrée contenant un fichier de même
nom s'écrasent donc mutuellement. Les images en mémoire ne possèdent aucun nom
de fichier à réutiliser et sont numérotées `image0`, `image1` et ainsi de suite.

Les sources vidéo et en direct sont écrites dans un seul fichier `.mp4` nommé
d'après la source.

`output_path` remplace le répertoire. Un chemin avec un suffixe est interprété
comme un fichier, un chemin sans suffixe comme un répertoire.
`output_file_format` sélectionne l'encodage des images fixes et accepte `jpg`,
`png` ou `webp`.

Après un enregistrement, le chemin écrit est aussi joint au résultat sous la
forme `result.saved_path`.
