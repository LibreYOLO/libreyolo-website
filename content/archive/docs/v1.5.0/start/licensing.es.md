---
title: Licencias
seo_title: 'Licencias de LibreYOLO: código y pesos'
description: >-
  El código propio de LibreYOLO es MIT. El código upstream incorporado y los
  checkpoints publicados llevan sus propias licencias, y varias de ellas son no
  comerciales.
lead: >-
  LibreYOLO reúne tres cosas con licencias separadas: su propio código, el
  código upstream incorporado en una familia de modelos y los checkpoints
  preentrenados. A menudo no comparten la misma licencia.
keywords:
  - libreyolo licencia
  - licencia libreyolo uso comercial
  - pesos de modelos no comercial
  - licencia checkpoints yolo
  - biblioteca vision artificial mit
  - apache-2.0 deteccion de objetos
last_verified: 1.5.0
source_hash: 83536fea4dc4eaec
---

## El código propio de LibreYOLO

La biblioteca es MIT. Eso cubre la API de Python, la CLI, los entrenadores, los
validadores y los exportadores, los cargadores de datasets y los scripts de
conversión que hay bajo `weights/`. Úsala en un producto comercial o de código
cerrado, conserva la línea de copyright y el texto de la licencia en cualquier
copia que redistribuyas, y la obligación termina ahí.

La concesión se detiene en el código. El archivo
[`LICENSE`](https://github.com/LibreYOLO/libreyolo/blob/release/LICENSE)
lo dice sin rodeos:

> Esas licencias varían y no todas son permisivas: algunos pesos publicados son
> no comerciales o están restringidos de otro modo, y esta licencia MIT no se
> extiende a ellos. Elegir un modelo significa elegir su licencia.

## Código upstream, por familia

La mayoría de familias son ports de investigación publicada, y varias
incorporan código fuente upstream directamente. Un archivo incorporado conserva
su cabecera de copyright original y su licencia original. MIT no la sobrescribe,
y LibreYOLO no relicencia el trabajo de nadie. Apache-2.0 y BSD-3-Clause son las
dos que aparecen con más frecuencia.

Apache-2.0 cubre la línea DETR y buena parte del trabajo con transformers: DETR
de Meta AI (FAIR), Deformable DETR de SenseTime, LW-DETR de Baidu, OV-DEIM de
Leilei Wang y coautores, la implementación de SegFormer que LibreYOLO porta de
Hugging Face Transformers, PP-OCRv5 de los PaddlePaddle Authors, SwinIR del
Computer Vision Lab de ETH Zúrich y Depth Anything 3 de ByteDance Seed. También
cubre los clasificadores derivados de timm, de Ross Wightman y los
contribuidores de timm, entre ellos ResNet, DeiT, EfficientNetV2, MobileNetV4 y
Swin, cuyos nombres de módulo replican los de timm para que sus tensores de
ImageNet carguen sin cambios.

BSD-3-Clause cubre todo lo derivado de torchvision: Faster R-CNN,
Mask R-CNN, FCOS, RetinaNet, SSD300, AlexNet, VGG, FCN y DeepLabv3.

MIT cubre un grupo más reducido, que incluye NAFNet de Megvii, CenterNet de
Xingyi Zhou y YOLOv7 tal como lo republicaron sus propios autores, Kin-Yiu Wong
y Hao-Tang Tsui, en MultimediaTechLab. Las familias de YOLOv1 a YOLOv4
reproducen arquitecturas del proyecto Darknet, de Joseph Redmon y, en el caso de
YOLOv4, de Alexey Bochkovskiy. Darknet es de dominio público, así que esas no
conllevan obligación alguna.

Uno de los subárboles incluidos no está bajo una licencia de código abierto. La
familia DEIMv2 distribuye código del backbone DINOv3 de Meta Platforms bajo el
DINOv3 License Agreement, una licencia propia no reconocida por la OSI.
Redistribuir ese código implica entregar con él una copia del acuerdo, y el
acuerdo prohíbe su uso en actividades sujetas a ITAR, con fines militares o
bélicos, en industrias nucleares, para espionaje y para el desarrollo de armas.
Esos términos vinculan únicamente a ese subárbol.

Dos archivos del repositorio contienen el cuadro completo.
[`NOTICE`](https://github.com/LibreYOLO/libreyolo/blob/release/NOTICE) enumera
cada subárbol de terceros incluido, con su ruta, su archivo de licencia y su
origen upstream.
[`THIRD_PARTY_NOTICES.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/THIRD_PARTY_NOTICES.txt)
enumera los proyectos upstream de los que deriva LibreYOLO y reproduce íntegro
el texto de cada licencia.

## Pesos, por checkpoint

Ningún archivo de pesos preentrenados se distribuye dentro del paquete. Los
checkpoints publicados están en Hugging Face, bajo la [organización
LibreYOLO](https://huggingface.co/LibreYOLO), y cada repositorio lleva su propio
`LICENSE` y su atribución, que reflejan el proyecto del que provienen los pesos.

Ese repositorio es la fuente autorizada de los términos. No esta página, ni la
página del modelo, ni el resumen que hay en el árbol de fuentes. Consulta
[checkpoints y pesos](/docs/weights) para saber cómo se nombran los archivos y
desde dónde se descargan.

Las licencias difieren entre familias, y difieren entre archivos dentro de una
misma familia. Dos ejemplos de lo segundo:

- Los checkpoints de YOLO9 para COCO son MIT. `LibreYOLO9P2s-visdrone.pt`,
  entrenado con VisDrone2019-DET, es CC BY-NC-SA 3.0, que es no comercial.
- Los checkpoints de detección de RF-DETR son Apache-2.0. Los checkpoints de
  cajas orientadas son CC BY 4.0, porque se les hizo fine-tuning sobre un
  dataset de Roboflow Universe publicado bajo CC BY 4.0 y los pesos arrastran
  el requisito de atribución de ese dataset.

Entre familias el abanico es más amplio, y varios checkpoints publicados no
pueden usarse en un producto comercial:

- SegFormer es la separación más clara entre las dos capas. La implementación es
  un port Apache-2.0 del código de Hugging Face Transformers. Los checkpoints
  publicados de ADE20K están convertidos a partir de la publicación de NVIDIA
  bajo la NVIDIA Source Code License, que permite la redistribución pero limita
  el uso a investigación o evaluación no comerciales, y traslada ese límite a
  las obras derivadas. Esos checkpoints no están cubiertos por los términos
  permisivos de LibreYOLO.
- Los checkpoints de OV-DEIM son CC BY-NC 4.0, confirmado por el autor upstream.
  Cada predicción carga además la torre de texto MobileCLIP-B(LT) de Apple, cuya
  licencia restringe el uso a investigación, un término más estricto que el del
  propio checkpoint.
- El código de SenseNova-Vision es Apache-2.0 y sus pesos son CC BY-NC 4.0. El
  cargador imprime el aviso de uso no comercial antes de cada descarga
  automática.

Algunas familias no tienen ningún checkpoint alojado por LibreYOLO, y sus
páginas lo indican en la fila Weights. SAM 3 está bajo acceso restringido en
Hugging Face con la SAM License propia de Meta y se descarga directamente de
Meta. Los assets de release de MiDaS se obtienen de las URLs oficiales y se
verifican por hash en lugar de rehospedarse. Dome-DETR se enlaza al upstream
porque su model card no declara ninguna licencia en sus metadatos mientras que
su prosa afirma Apache-2.0 y a la vez restringe el uso a investigación
académica, y ambas cosas no concuerdan. Las arquitecturas TEED y DexiNed son
MIT, pero los checkpoints publicados por sus autores se entrenaron con BIPED,
cuyos términos de dataset son no comerciales, así que LibreYOLO ni los incluye
ni los descarga automáticamente.

Varios checkpoints de torchvision no llevan un archivo de licencia propio.
LibreYOLO los replica bajo la licencia que usa el proyecto que los publica,
indica en cada model card que la base es implícita y no una concesión por
checkpoint, y repite la propia advertencia de torchvision de que los términos de
un modelo preentrenado pueden derivarse de los datos de entrenamiento.

## Encontrar los términos de un modelo concreto

La página del modelo lleva una fila **Licenses** en su cabecera, con la forma
`Code X, weights Y`, que enlaza hacia abajo con la sección Licensing de la
página. Esa sección enumera el trabajo original y sus autores, la licencia
upstream, el origen upstream, la licencia del código de LibreYOLO, los pesos y
una interpretación de lo que permiten los términos. La tabla Checkpoints de esa
misma página tiene una columna **Weights license**, con una fila por archivo
publicado, de modo que una familia con términos mixtos los muestra archivo por
archivo.

Todo eso se renderiza a partir de los mismos datos contra los que se comprueba
la biblioteca, y por eso esta página no lo repite en forma de tabla. Una matriz
de licencias escrita a mano se queda desactualizada antes de que pase una
versión, y equivocarse aquí sale caro.

En el árbol de fuentes, los equivalentes son `NOTICE` para el código incluido,
`THIRD_PARTY_NOTICES.txt` para los proyectos upstream y sus textos de licencia, y
[`weights/LICENSE_NOTICE.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/weights/LICENSE_NOTICE.txt)
para un resumen por familia de los checkpoints publicados.

Después consulta el repositorio de Hugging Face del archivo exacto que vas a
descargar. Es la fuente autorizada, y puede cambiar sin que una página de
documentación cambie con él.

## Uso comercial

El código rara vez es el problema. MIT, Apache-2.0 y BSD-3-Clause permiten todas
el uso comercial y de código cerrado. Cada una te pide que conserves su texto de
licencia y sus avisos de atribución en las copias que redistribuyas, Apache-2.0
concede además una licencia de patentes, y ninguna impone condiciones sobre el
código de tu propia aplicación.

Donde los productos se atascan es en los checkpoints. Un checkpoint no comercial
sigue siendo no comercial por muy permisivo que sea el código que lo rodea, y
convertir el archivo no cambia los términos que se le aplican, que es lo que dice
directamente `weights/LICENSE_NOTICE.txt`. Un artefacto ONNX o TensorRT
construido a partir de un checkpoint restringido hereda la restricción.

Cuando una licencia traslada su restricción a las obras derivadas, como hace la
NVIDIA Source Code License, hacer fine-tuning tampoco te libra de ella. Entrenar
la misma arquitectura desde cero con datos que tienes derecho a usar sí: el
código es permisivo, así que un modelo que entrenas tú es tuyo, y los términos
del checkpoint preentrenado nunca entran en él. La página de SegFormer lo detalla
para sus propios pesos; lee la fila Interpretation en la página de la familia que
tengas previsto llevar a producción.

Resuelve la cuestión de la licencia cuando elijas el modelo, y no cuando vayas a
publicar, y lee los términos del archivo que realmente descargaste, porque una
familia con un checkpoint permisivo puede tener otro restringido al lado.

## Esto no es asesoramiento legal

Esta página describe las licencias implicadas. Es una descripción, no
asesoramiento legal, y no crea ninguna garantía. Si la respuesta importa a nivel
comercial, lee tú mismo las licencias y busca tu propio asesoramiento jurídico.
