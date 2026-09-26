---
title: PicoSAM3
families:
  - picosam3
seo_title: 'PicoSAM3 : segmentation edge guidée par boîte dans LibreYOLO'
description: >-
  Utilisez PicoSAM3 dans LibreYOLO pour la segmentation de régions guidée par
  boîte sur des capteurs edge. Installez, prédisez et exportez le checkpoint
  pico sous Apache-2.0.
lead: >-
  PicoSAM3 est un CNN compact distillé à partir de SAM 2.1 et SAM 3, conçu pour
  la segmentation de régions d'intérêt guidée par boîte sur des capteurs tels
  que le Sony IMX500. LibreYOLO le prend en charge par une factory LibreSAM
  dédiée, distincte de la factory de détecteurs LibreYOLO(), et uniquement avec
  des requêtes par boîte.
keywords:
  - PicoSAM3
  - Segment Anything
  - segmentation edge
  - région d'intérêt
  - requête par boîte
  - inférence dans le capteur
  - IMX500
  - distillation de connaissances
last_verified: 1.5.0
snippets:
  predict:
    - label: Requête par boîte
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # PicoSAM3 n'a qu'une taille, "pico", aucun autre alias n'est
        nécessaire.

        model = LibreSAM("picosam3")


        # bboxes= est la seule requête prise en charge : [x1, y1, x2, y2] ou une

        # liste de boîtes, un masque par boîte. Chaque boîte est agrandie de 10
        %,

        # rendue carrée, limitée à l'image et redimensionnée à 96x96 avant le
        CNN.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        print(result.masks.xy)      # un polygone par masque

        print(result.boxes.xyxy)    # boîte ajustée dérivée du masque
    - label: 'Encoder une fois, fournir plusieurs requêtes'
      language: python
      code: >
        from libreyolo import LibrePicoSAM3, SAMPLE_IMAGE


        model = LibrePicoSAM3()


        # set_image() met l'image source en cache ; PicoSAM3 exécute une passe
        CNN

        # complète par boîte. Cela évite le chargement/décodage de l'image, pas
        une

        # passe d'encodeur comme pour les autres familles SAM.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(bboxes=[300, 200, 900, 700])

        b = model.predict(bboxes=[100, 100, 400, 400])

        model.reset_image()
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibrePicoSAM3

        model = LibrePicoSAM3()
        model.export(format="onnx", output_path="LibrePicoSAM3pico.onnx")

        # opset (13 par défaut) et dynamic (True par défaut, axe de batch seul)
        # sont les seuls arguments d'export acceptés par cette famille.
    - label: Utiliser le fichier exporté
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # PicoSAM3 exporte son CNN ROI 96x96 brut : roi_image -> mask_logits.

        # Aucun pré/post-traitement LibreYOLO n'est réutilisable ici, car
        export()

        # ne repasse pas par LibreYOLO() comme pour un checkpoint de détecteur.

        session = ort.InferenceSession("LibrePicoSAM3pico.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 96, 96),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 5d60ff14fe61ba29
---

## Installer

PicoSAM3 nécessite l'extra `sam` : le téléchargement de poids propre à
LibreYOLO passe toujours par les outils Hugging Face de `transformers`, même si
l'inférence s'exécute sur un CNN natif indépendant de `transformers`.

```bash
pip install "libreyolo[sam]"
```

## Prédire

`LibreSAM(...)` (ou `LibrePicoSAM3(...)`, propre à la famille) constitue un
point d'entrée distinct de `LibreYOLO(...)` : il renvoie un segmenteur guidable
et non un détecteur, car une passe n'a ici aucun sens sans requête. Il n'existe
pas de commande CLI `libreyolo predict` pour cette famille ; utilisez l'API
Python.

<code-tabs name="predict" />

PicoSAM3 accepte uniquement `bboxes=`. Fournir `points=`, `labels=`, `masks=`,
`text=`, `multimask=True` ou omettre la boîte pour tout segmenter lève dans tous
les cas une `ValueError` explicite, car aucun de ces modes n'existe dans le
modèle amont. `conf` filtre selon la qualité prédite du masque (IoU), et non une
confiance de détection, et doit être compris entre `0.0` et `1.0`. Chaque masque
porte l'identifiant de classe `0`, nommé `"object"`. `train()`, `val()` et
`track()` lèvent `NotImplementedError` ; utilisez LibreSAM2 ou LibreSAM3 pour
les requêtes par point, texte, masque ou de segmentation complète. Consultez la
[prédiction](/docs/predict) pour les types de sources.

## Variantes

Une seule taille, pico, avec une entrée ROI fixe de 96 px : PicoSAM3 exécute une
passe CNN complète par boîte au lieu d'encoder l'image entière une seule fois.

## Exporter

<export-matrix />

PicoSAM3 est la seule famille de la catégorie SAM qui s'exporte : elle transmet
son CNN ROI 96x96 brut à ONNX, `roi_image -> mask_logits`, sans NMS ni
post-traitement de masque intégré. Les autres familles SAM lèvent
`NotImplementedError` avec `export()`, car leur séparation encodeur/décodeur ne
dispose pas encore d'un contrat d'export pour le runtime. Un graphe PicoSAM3
exporté ne se recharge pas par `LibreYOLO()` ; exécutez-le directement avec un
runtime comme `onnxruntime`, en appliquant le même prétraitement de ROI carrée
avec une marge de 10 % présenté ci-dessus.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box>

PicoSAM3 est distillé à partir de SAM 2.1 et SAM 3 utilisés comme modèles
enseignants. LibreYOLO n'intègre ni ne redistribue le code ou les poids d'aucun
de ces enseignants dans cette famille ; seuls le CNN élève compact et son
checkpoint converti sont fournis.

</provenance-box>

## Citation

<citation-block />
