---
title: Triton Inference Server
seo_title: Servir un modèle LibreYOLO sur NVIDIA Triton
description: "Servez un export ONNX LibreYOLO à travers NVIDIA Triton\_: la structure du model repository, le config.pbtxt généré et la prédiction contre une URL de modèle HTTP."
lead: >-
  Triton Inference Server héberge un model repository et répond aux requêtes
  d'inférence en HTTP. LibreYOLO exporte le graphe ONNX, génère un config.pbtxt
  qui transporte les métadonnées de l'export sous forme d'un unique paramètre
  Triton, et traite une URL de modèle comme un chemin de modèle chargeable.
keywords:
  - libreyolo triton
  - triton inference server
  - servir un modèle yolo avec triton
  - config.pbtxt yolo
  - tritonclient http
  - model repository triton
  - inférence yolo à distance
last_verified: 1.5.0
meta:
  - label: Appel
    value: 'LibreYOLO("http://127.0.0.1:8000/yolo9")'
    mono: true
  - label: Fonction utilitaire
    value: >-
      create_triton_config(onnx_path, config_path, model_name=...,
      max_batch_size=8)
    mono: true
  - label: Extra
    value: 'pip install "libreyolo[onnx,triton]"'
    mono: true
  - label: Protocole
    value: >-
      Inférence V2 en HTTP et HTTPS uniquement. Pas de gRPC, d'authentification,
      de mémoire partagée, ni de chargement et déchargement de modèles.
  - label: Timeouts
    value: Les timeouts de connexion et de réseau sont de 30 secondes par défaut
verification: >-
  Lu depuis libreyolo/backends/triton.py, libreyolo/models/__init__.py,
  docs/triton.md et pyproject.toml sur la branche dev. Les commandes de
  conteneur sont celles fixées dans docs/triton.md.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        pip install "libreyolo[onnx,triton]"
  repo:
    - label: Exporter dans la structure du repository
      language: python
      code: |
        from pathlib import Path

        from libreyolo import LibreYOLO

        model_dir = Path("triton_repo/yolo9/1")
        model_dir.mkdir(parents=True, exist_ok=True)

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            output_path=str(model_dir / "model.onnx"),
            dynamic=True,
            simplify=False,
        )
    - label: Générer config.pbtxt
      language: python
      code: |
        from libreyolo import create_triton_config

        create_triton_config(
            "triton_repo/yolo9/1/model.onnx",
            "triton_repo/yolo9/config.pbtxt",
            model_name="yolo9",
            max_batch_size=8,
        )
    - label: Structure résultante
      language: text
      code: |
        triton_repo/
          yolo9/
            config.pbtxt
            1/
              model.onnx
  serve:
    - label: Démarrer le serveur
      language: bash
      code: |
        docker run --rm --name libreyolo-triton \
          -p 8000:8000 -p 8002:8002 \
          -v "$(pwd)/triton_repo:/models:ro" \
          nvcr.io/nvidia/tritonserver:26.04-py3 \
          tritonserver --model-repository=/models --exit-on-error=true
    - label: Attendre que le serveur soit prêt
      language: bash
      code: >
        until curl --fail --silent http://127.0.0.1:8000/v2/health/ready; do
        sleep 1; done
    - label: L'arrêter
      language: bash
      code: |
        docker stop libreyolo-triton
  run:
    - label: Prédire contre le modèle servi
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9")
        result = remote.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Comparer avec le modèle local
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9").predict(SAMPLE_IMAGE)
        native = LibreYOLO("LibreYOLO9t.pt").predict(SAMPLE_IMAGE)

        print(len(remote.boxes), len(native.boxes))
        print(remote.boxes.xyxy[:3])
        print(native.boxes.xyxy[:3])
    - label: 'Fixer une version, ou changer le timeout'
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.backends.triton import TritonBackend

        # Un deuxième segment de chemin choisit la version du modèle. Sans lui,
        # la politique de versions configurée dans Triton décide.
        pinned = LibreYOLO("http://127.0.0.1:8000/yolo9/1")

        # Les timeouts de connexion et de réseau valent 30 secondes par défaut.
        patient = TritonBackend("http://127.0.0.1:8000/yolo9", timeout=120)
source_hash: 0652e4faf0224df3
---

## Installation

<code-tabs name="install" />

L'extra `triton` installe `tritonclient[http]`. Les extras gRPC et mémoire
partagée sont exclus volontairement : cette intégration ne fait que de
l'inférence V2 en HTTP et HTTPS. `onnx` est nécessaire parce que l'artefact servi
et le générateur de config travaillent tous deux à partir d'un graphe ONNX.

## Construire le model repository

Exportez avec un axe de batch dynamique, dans la structure de répertoires
attendue par Triton.

<code-tabs name="repo" />

Triton ne conserve pas les métadonnées ONNX personnalisées dans sa réponse de
configuration de modèle, donc les métadonnées exportées complètes doivent voyager
autrement. `create_triton_config` les encode sous forme d'un unique paramètre de
type chaîne JSON nommé `libreyolo_metadata` dans `config.pbtxt`, émet les
déclarations d'entrées et de sorties dans l'ordre du graphe, gère l'échappement
JSON et fixe le modèle à `KIND_CPU`.

La fonction utilitaire valide avant d'écrire. Elle exige exactement une entrée de
graphe ONNX, au moins une sortie, des formes de tenseur résolvables, et des
métadonnées dont la table `names` définit tous les indices de classe de 0 à
`nc - 1`. Un modèle qui échoue à l'une de ces vérifications est rejeté au moment
de la génération du config, plutôt qu'à la première requête.

`max_batch_size: 8` correspond à un export dynamique et permet au serveur de
regrouper jusqu'à huit images par requête. Pour un graphe ONNX à batch fixe de 1,
utilisez `max_batch_size=0` ; LibreYOLO envoie alors les images les unes après
les autres.

## Démarrer le serveur

<code-tabs name="serve" />

Les commandes fixent Triton Server 26.04 et omettent délibérément les flags GPU
de Docker, puisque `KIND_CPU` dans le config généré empêche de toute façon le
placement sur GPU.

## Exécuter l'artefact

Une URL de modèle Triton est un chemin de modèle. `LibreYOLO()` vérifie la
présence d'un schéma `http` ou `https` avant tout traitement de chemin local et
renvoie un backend qui dialogue avec le serveur, si bien que le site d'appel est
identique à celui d'un checkpoint local, tout comme l'objet `Results` renvoyé.

<code-tabs name="run" />

La forme de l'URL est `http(s)://host:port/model` avec un segment de version
optionnel. Le port doit être explicite. Les identifiants intégrés, une query
string et un fragment sont tous rejetés, de même qu'un chemin de plus de deux
segments.

`device` est accepté et ignoré avec une ligne de log, parce que le placement est
la décision du serveur.

## Contraintes

Le backend échoue avec une erreur directe plutôt qu'avec un résultat dégradé
lorsque le contrat n'est pas respecté : métadonnées LibreYOLO absentes du config
du modèle, plus d'une entrée de modèle, désaccord entre les sorties configurées
et les métadonnées du modèle, type de données d'entrée qu'il ne prend pas en
charge, ou serveur ou modèle qui n'est pas prêt.

Hors contrat dans cette version : gRPC, l'authentification, la mémoire partagée,
et le chargement ou le déchargement de modèles via l'API.

N'importe quel format que Triton prend lui-même en charge peut être servi, mais
le paramètre de métadonnées et le config généré ont ici une forme ONNX, donc le
chemin LibreYOLO passe par [ONNX](/docs/export/onnx) dans le repository. Pour un
pipeline vidéo complet plutôt qu'un serveur requête-réponse, voir
[DeepStream](/docs/export/deepstream).
