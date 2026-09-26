---
title: Dome-DETR
families:
  - domedetr
seo_title: "Dome-DETR\_: la détection d'objets minuscules dans LibreYOLO"
description: >-
  Utilisez Dome-DETR dans LibreYOLO pour la détection d'objets minuscules sur
  des images aériennes et de drone. Convertissez les poids d'amont, prédisez,
  faites du fine-tuning et validez avec du code sous licence MIT.
lead: "Un spécialiste des objets minuscules construit sur D-FINE\_: une tête de densité décide où se trouvent les objets, l'attention de l'encodeur est restreinte aux fenêtres qui les contiennent, et le nombre de queries est dimensionné à partir de cette densité au lieu d'être fixe. LibreYOLO le prend en charge pour la détection."
keywords:
  - Dome-DETR
  - détection d'objets minuscules
  - détection de petits objets
  - imagerie aérienne
  - détection par drone
  - télédétection
  - VisDrone
  - AI-TOD
  - DETR
  - queries adaptatives à la densité
last_verified: 1.5.0
snippets:
  predict:
    - label: 'Convertir, puis prédire'
      language: bash
      code: |
        # LibreYOLO n'héberge aucun poids Dome-DETR, le checkpoint est donc
        # récupéré depuis le dépôt d'amont et converti une seule fois.
        hf download RicePasteM/Dome-DETR --include 'best_ckpts_dome_2026/*' \
          --local-dir dome-ckpts

        python weights/convert_domedetr_weights.py \
          dome-ckpts/best_ckpts_dome_2026/dome-s-visdrone_converted.pth \
          LibreDOMEDETRs-visdrone.pt --size s --variant visdrone
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Un chemin local, pas un simple nom : rien ne se télécharge ici.
        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        result = model("drone-frame.jpg", save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDOMEDETRs-visdrone.pt
        source=drone-frame.jpg save=True
    - label: Noms de classes
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Il n'y a pas de checkpoint COCO, les classes viennent donc du dataset
        # d'entraînement des poids, lues dans les métadonnées du checkpoint.
        aitod = LibreYOLO("LibreDOMEDETRs-aitod.pt")
        print(aitod.model.names)     # 9 classes AI-TOD-V2

        visdrone = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        print(visdrone.model.names)  # 12 classes VisDrone
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

## Installation

Dome-DETR ne demande aucun extra optionnel. Tout ce qu'il importe est présent
dans l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Il n'y a rien à télécharger automatiquement. LibreYOLO n'héberge pas ces poids,
le déroulé est donc le suivant : récupérer le checkpoint d'amont, le convertir
une fois, puis charger le fichier converti par chemin. [Licence](#licensing)
explique pourquoi.

<code-tabs name="predict" />

L'objet `Results` renvoyé est celui que renvoie chaque famille, remplacer un
détecteur par un autre est donc une modification d'une ligne. `conf` et
`max_det` filtrent la sélection des queries ; `iou` est accepté par parité d'API
mais n'a aucun effet, car le décodeur est un prédicteur d'ensemble sans étape de
NMS. Voir [la prédiction](/docs/predict) pour les sources, le streaming et le
traitement des résultats.

Deux capacités sont désactivées pour cette famille. La capture de graphe CUDA
est désactivée, car le nombre de queries de PAQI dépend des données et la passe
avant change donc de forme d'une image à l'autre, ce qui est exactement ce que
la capture de graphe ne peut pas absorber. L'augmentation au moment du test
tourne à une seule taille carrée fixe, une demande de TTA multi-échelle est donc
sans effet.

## Variantes

Trois tailles, s, m et l, toutes en 800 par 800. La taille sélectionne le
backbone, et le dataset dont viennent les poids sélectionne la profondeur du
décodeur et le budget de queries, un code de taille seul n'identifie donc pas un
graphe. Les poids AI-TOD-V2 sélectionnent entre 300 et 1500 queries par image,
les poids VisDrone entre 250 et 500, et le grand modèle exécute quatre couches
de décodeur sur AI-TOD-V2 contre six sur VisDrone.

Dome-DETR est D-FINE avec trois ajouts. DeFE prédit une carte de densité. MWAS
utilise cette carte pour restreindre l'attention de l'encodeur aux fenêtres qui
contiennent effectivement des objets, plutôt que de porter l'attention partout.
PAQI dimensionne l'ensemble de queries à partir de cette même densité au lieu de
décoder un nombre fixe de 300. Le gain se concentre là où les objets sont les
plus petits, et se réduit à mesure qu'ils grandissent : l'ablation d'amont
elle-même fait passer l'AP sur les objets très minuscules de 14.0 à 17.8 tandis
que l'AP sur les objets moyens ne passe que de 45.4 à 46.4. Considérez-le comme
un complément de [D-FINE](/docs/models/d-fine) pour l'imagerie aérienne, de
drone et de télédétection, pas comme un remplacement.

LibreYOLO ne publie aucune ligne de benchmark pour cette famille, parce qu'elle
ne publie aucun checkpoint à évaluer.

## Entraîner

Dome-DETR est entraînable. L'entraînement exécute l'objectif complet d'amont :
les losses de D-FINE plus la supervision de densité et de comptage de DeFE, avec
les queries de remplissage masquées hors des termes de classification et des
masques d'attention de denoising par image, pour que le remplissage d'une image
ne puisse pas fuiter dans celui d'une autre.

<code-tabs name="train" />

La configuration hérite de la recette de D-FINE et change ce que MWAS exige.
`imgsz` vaut 800, `lr0` vaut `2e-4`, le groupe de paramètres du backbone est mis
à l'échelle par `backbone_lr_mult=0.1`, et `multi_scale` est forcé à off, car
les fenêtres MWAS ont besoin que l'entrée reste divisible par le stride 8.
`batch` vaut 4 par défaut plutôt que les 16 de D-FINE : PAQI complète chaque
batch jusqu'à son membre le plus large, la mémoire suit donc l'image la plus
chargée du batch plutôt que l'image moyenne.

Une réserve honnête sur l'exactitude. L'amont entraîne pendant 160 époques avec
`MultiStepLR(milestones=[80, 120], gamma=0.8)`, tandis que ces valeurs par
défaut exécutent le planning flat-cosine de D-FINE sur les mêmes 160 époques. Ce
planning n'a pas été reproduit ici et les chiffres d'AP de l'article n'ont pas
été reproduits non plus, lisez-les donc comme les résultats des auteurs d'amont
plutôt que comme une promesse que cette recette les atteint. Fournissez le
planning d'amont si l'objectif est d'égaler l'article.

Voir [l'entraînement](/docs/train) pour les datasets, l'augmentation, le
multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire indexé par nom de métrique, et affiche les
résultats par classe quand `verbose` est laissé activé.

<code-tabs name="val" />

La validation s'exécute sur votre propre dataset, dans le format sur lequel vous
avez entraîné. Le contrôle de validation COCO de la bibliothèque ne s'applique
pas ici, puisqu'il n'existe aucun checkpoint COCO à mesurer pour cette famille.

## Exporter

L'export n'est pris en charge pour aucun format, et en demander un lève une
erreur au lieu de produire un fichier.

La raison est PAQI. Il décide du nombre de queries par image, à partir de
propositions filtrées par densité et d'une boucle gloutonne de suppression
adaptative à la densité, la longueur de sortie du décodeur est donc une
propriété de l'entrée et non du graphe. Le tracing fige le nombre que l'image de
tracing a produit, ce qui donne un artefact qui renvoie silencieusement de
mauvais résultats pour toute autre image. Une formulation statique devrait
dérouler cette suppression sur l'ensemble des 250 à 1500 candidats, et se
rabattre sur un top-k fixe supprimerait exactement le rappel sur les objets
minuscules pour lequel cette famille existe. S'il vous faut un transformer de
détection exportable, [D-FINE](/docs/models/d-fine) est celui vers lequel se
tourner.

## Checkpoints

Il n'y en a aucun à lister. LibreYOLO ne publie aucun poids Dome-DETR, et aucun
nom de la forme `LibreDOMEDETR<size>-<dataset>.pt` ne se résout en un
téléchargement.

L'amont publie six checkpoints, s, m et l pour chacun des deux datasets :
AI-TOD-V2 avec 9 classes et VisDrone avec 12. Il n'y a pas de checkpoint COCO,
un nom de fichier canonique porte donc toujours le suffixe du dataset, et les
noms de classes voyagent dans les métadonnées du checkpoint au lieu de venir
d'une constante de famille. Demander un simple `LibreDOMEDETRs.pt` lève
immédiatement une erreur, avec un message qui nomme les deux vrais noms de
fichiers et la commande de conversion, plutôt que de tenter un téléchargement
qui renverrait une 404.

`weights/convert_domedetr_weights.py` fait la conversion. Il reconstruit le
graphe LibreYOLO, y charge les tenseurs d'amont, et refuse d'écrire quoi que ce
soit si une seule clé est manquante, inattendue ou de mauvaise forme, un fichier
converti est donc soit une correspondance exacte, soit inexistant. Pointez-le
vers un `.pth` d'amont et passez la taille et la variante :

```bash
python weights/convert_domedetr_weights.py \
    dome-ckpts/best_ckpts_dome_2026/aitod-s-best.pth \
    LibreDOMEDETRs-aitod.pt --size s --variant aitod
```

Sur la fidélité numérique, `weights/parity_domedetr.py` compare ce portage à
l'implémentation d'amont sur les six checkpoints et rapporte
`max_abs_diff == 0.0` sur `pred_logits` comme sur `pred_boxes`, après avoir
d'abord vérifié bit à bit le masque de fenêtre MWAS, et compare séparément
chaque terme de loss au critère d'amont. Soyez clair sur ce que c'est : un
script manuel qui a besoin du checkout d'amont et des checkpoints publiés sur le
disque, lancé à la main. Il ne fait pas partie de l'intégration continue, et
aucun job de CI ne le reproduit.

## Licence

<provenance-box>

Les poids sont la raison pour laquelle cette famille n'est pas mise en miroir.
La model card d'amont ne porte aucun champ de licence dans ses métadonnées, et
sa prose indique que le projet est en Apache-2.0 tout en restreignant le
matériel à des fins de recherche académique uniquement. Ces deux lectures ne
concordent pas, et la plus stricte n'est pas une autorisation de
redistribution, LibreYOLO renvoie donc vers le dépôt d'amont au lieu de copier
les fichiers, en attendant clarification. C'est le même raisonnement qui
gouverne [YOLO-NAS](/docs/models/yolo-nas) ici.

Le code est une question distincte, et plus claire. Le dépôt d'amont est en
Apache-2.0, le portage de LibreYOLO est en MIT, et les poids que vous entraînez
vous-même sur vos propres données sont à vous.

</provenance-box>

## Citation

Dome-DETR a été publié à ACM Multimedia 2025 sous le titre « Dome-DETR: DETR
with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object
Detection ». La prépublication est sur
[arxiv.org/abs/2505.05741](https://arxiv.org/abs/2505.05741). Les auteurs ne
publient aucun bloc BibTeX dans leur dépôt, aucun n'est donc reproduit ici au
lieu d'être assemblé à la main.

<citation-block />
