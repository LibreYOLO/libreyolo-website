---
title: "Alternative à SuperGradients : YOLO-NAS en vision par ordinateur temps réel"
description: "SuperGradients n'a plus publié de version depuis le rachat de Deci par NVIDIA en 2024. LibreYOLO est une alternative maintenue qui exécute les mêmes poids YOLO-NAS : prédiction, entraînement et export avec une API sous licence MIT."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, supergradients-alternative, yolo-nas, object-detection]
faq:
  - q: "SuperGradients est-il toujours maintenu ?"
    a: "Non. La dernière version, 3.7.1, date d'avril 2024, juste avant le rachat de Deci par NVIDIA. Depuis, le dépôt n'a publié aucune version, les issues restent sans réponse et le site de documentation est hors ligne. Le projet n'est pas officiellement archivé, mais il n'y a plus personne aux commandes."
  - q: "Quelle est la meilleure alternative à SuperGradients ?"
    a: "Pour exécuter YOLO-NAS, LibreYOLO charge les mêmes poids Deci avec une API maintenue sous licence MIT, et ajoute l'entraînement, la validation et l'export. Si vous dépendez des recettes d'entraînement de SuperGradients pour d'autres architectures, l'ancien dépôt fonctionne toujours si vous épinglez ses dépendances."
  - q: "Puis-je utiliser YOLO-NAS à des fins commerciales ?"
    a: "Les poids pré-entraînés de Deci sont soumis à une licence non commerciale, quelle que soit la bibliothèque qui les charge. L'architecture elle-même est sous licence permissive. En entraînant YOLO-NAS à partir de zéro sur vos propres données, vous obtenez donc un modèle qui n'a aucun checkpoint Deci dans son historique."
  - q: "LibreYOLO remplace-t-il tout ce que faisait SuperGradients ?"
    a: "Pas tout. LibreYOLO ne propose pas d'export CoreML pour YOLO-NAS et son chemin TensorRT pour cette famille n'a pas été testé. Les recettes d'entraînement avec quantification aware training de SuperGradients n'ont pas non plus d'équivalent direct. Pour ces usages, gardez l'ancien dépôt."
---

SuperGradients était la bibliothèque d'entraînement open source de Deci et le projet qui hébergeait YOLO-NAS, l'un des détecteurs temps réel les plus précis jamais publiés. En mai 2024, NVIDIA a racheté Deci et SuperGradients est devenu silencieux. La dernière version, 3.7.1, est sortie le 8 avril 2024. La documentation sur supergradients.com a été mise hors ligne. Environ 120 issues restent ouvertes, et celle intitulée [« Is this project dead? »](https://github.com/Deci-AI/super-gradients/issues/2062) n'a reçu aucune réponse d'un mainteneur, ce qui constitue en soi une réponse.

Le dépôt n'est pas archivé, il semble donc encore actif au premier abord. En pratique, l'abandon se fait sentir dès l'installation : `super-gradients` épingle `torchmetrics==0.8`, qui entre en conflit avec les piles PyTorch actuelles, et ajoute au passage hydra, omegaconf, boto3 et tensorboard. À chaque mois qui passe, ces versions épinglées s'accordent plus difficilement avec le reste de votre environnement, et aucun correctif n'est prévu.

Rien de tout cela ne rend YOLO-NAS moins performant. La variante large atteint toujours 52.2 mAP sur COCO en temps réel. Le modèle va bien, c'est son environnement qui est en ruines.

<iframe
  src="https://visionanalysis.org/embed/scatter?highlight=yolonas-s%2Cyolonas-m%2Cyolonas-l"
  width="100%"
  height="420"
  style="border:0;border-radius:12px;overflow:hidden"
  loading="lazy"
  title="Précision de YOLO-NAS en fonction du nombre de paramètres - visionanalysis.org">
</iframe>

## Exécuter les mêmes poids avec LibreYOLO

LibreYOLO charge directement les checkpoints YOLO-NAS depuis le CDN de Deci, avec la même API que pour toutes ses autres familles de modèles. Vous n'avez pas de configurations hydra à écrire ni de wrapper de prédiction à retirer :

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

Les variantes de détection S, M et L fonctionnent toutes, tout comme la variante pose : remplacez par `LibreYOLONASs-pose.pt` pour obtenir les points clés COCO. Comme toutes les familles renvoient le même objet `Results`, comparer YOLO-NAS à RF-DETR ou D-FINE sur vos propres données se fait en changeant une seule ligne, au lieu de maintenir une deuxième base de code.

## Entraînement et export, pas seulement l'inférence

SuperGradients était avant tout une bibliothèque d'entraînement, une véritable alternative doit donc permettre d'entraîner. LibreYOLO affine YOLO-NAS sur votre dataset avec le même appel que pour toutes les autres familles :

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

Vous pouvez aussi entraîner un modèle initialisé de façon aléatoire, ce qui compte pour les licences, comme expliqué plus bas. La validation renvoie des métriques mAP sur n'importe quel dataset dans votre format, et l'export prend en charge ONNX, TorchScript, OpenVINO, NCNN et TFLite. Cette matrice d'export est plus étendue que celle que la bibliothèque d'origine a jamais proposée pour YOLO-NAS. Tous les détails figurent sur la [page de documentation YOLO-NAS](https://www.libreyolo.com/docs/models/yolo-nas), et cette note plus courte indique que [YOLO-NAS est toujours maintenu](/articles/yolo-nas-with-libreyolo).

## Quand l'ancien dépôt reste le bon choix

Par souci de transparence, voici trois cas où vous devriez garder SuperGradients installé :

**CoreML.** LibreYOLO n'exporte pas YOLO-NAS vers CoreML. Si vous déployez sur des appareils Apple et avez besoin d'un `.mlpackage`, SuperGradients dispose encore d'un chemin fonctionnel.

**TensorRT.** SuperGradients documentait et testait son flux TensorRT pour YOLO-NAS, y compris les particularités de la taille de batch. La prise en charge de TensorRT par LibreYOLO pour cette famille n'a pas été testée.

**Recettes d'entraînement avec quantification aware training.** YOLO-NAS a été conçu pour INT8, et SuperGradients fournissait les recettes QAT qui lui permettaient d'exceller dans ce domaine. LibreYOLO exporte avec une quantification INT8 et FP16, mais ne propose pas d'équivalent à ces recettes d'entraînement.

L'ancien dépôt fonctionne toujours si vous l'utilisez dans un environnement de 2024 dont vous avez épinglé les dépendances. Évitez simplement de construire quelque chose de nouveau sur une base que personne ne maintient.

## À propos des poids

Les poids YOLO-NAS pré-entraînés appartiennent à Deci, sont publiés sous une licence non commerciale et restent soumis à cette licence dans toute bibliothèque qui les charge. LibreYOLO ne les héberge pas et n'en fournit aucun miroir. Le téléchargement provient du CDN public de Deci, qui affiche ses conditions avant de démarrer. Dans le cadre de la recherche et d'un usage non commercial, les deux solutions vous conviennent.

Si vous avez besoin d'un YOLO-NAS à usage commercial, une voie claire existe désormais : l'architecture elle-même est sous licence permissive. L'entraînement à partir de zéro sur vos propres données produit donc un modèle qui ne dérive d'aucun checkpoint Deci. LibreYOLO prend précisément cela en charge avec `LibreYOLONAS(None, size="s")`.

## Essayer

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASl.pt")
results = model("image.jpg", save=True)
```

LibreYOLO est sous licence MIT, fonctionne sous Linux, Mac et Windows, et tourne sur GPU, Apple Silicon et CPU sans aucune modification du code. Une seule API couvre YOLO-NAS, RF-DETR, D-FINE, DEIM, YOLOX, RTMDet et bien d'autres modèles, pour la détection, la segmentation, l'estimation de pose, la classification, la profondeur et le suivi.

Ajoutez une étoile sur GitHub : [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentation : [libreyolo.com/docs](https://www.libreyolo.com/docs)
