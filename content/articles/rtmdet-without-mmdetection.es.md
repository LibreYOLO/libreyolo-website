---
title: Cómo ejecutar RTMDet sin instalar MMDetection
description: RTMDet es uno de los detectores precisos más rápidos. La instalación oficial falla con cualquier versión de PyTorch publicada en los últimos dos años. Esta es la alternativa.
date: 2026-06-27
author: Xuban
tags: [LibreYOLO, rtmdet, object-detection, tutorial]
faq:
  - q: "¿Por qué falla mmcv al instalarlo con versiones recientes de PyTorch?"
    a: "La versión más reciente de mmcv, la 2.2.0 de abril de 2024, solo ofrece binarios precompilados hasta torch 2.4 / CUDA 12.1. Con cualquier versión posterior de PyTorch, mmcv debe compilar sus operaciones de C++/CUDA desde el código fuente, lo que tarda entre 10 y 30 minutos y requiere un toolkit de CUDA compatible, nvcc y un compilador de C++ compatible. De ahí vienen errores como la falta del módulo mmcv._ext."
  - q: "¿Puedo ejecutar RTMDet sin instalar mmdetection?"
    a: "Sí. LibreYOLO carga pesos de RTMDet convertidos desde los checkpoints originales de OpenMMLab, y la inferencia es equivalente bit a bit a mmdetection con el mismo checkpoint. Carga LibreRTMDets.pt por nombre y se descarga automáticamente. Las cinco variantes, de Tiny a X, funcionan a 640 px."
  - q: "¿RTMDet es gratis para uso comercial?"
    a: "Sí. MMDetection y RTMDet usan Apache 2.0, los pesos convertidos mantienen las mismas condiciones de Apache 2.0 y el código de LibreYOLO usa MIT. No hay restricciones comerciales."
  - q: "¿Cuándo sigue siendo mmdetection la opción adecuada?"
    a: "Si necesitas entrenar o hacer fine-tuning de RTMDet con todo el pipeline de aumento de datos de MMDetection, o ya trabajas a fondo en el ecosistema OpenMMLab y tienes mmcv funcionando. Para la inferencia y el despliegue, LibreYOLO es la mejor opción."
---

RTMDet es un detector en tiempo real de OpenMMLab que logra una alta precisión en COCO y mantiene la velocidad necesaria para el despliegue. La arquitectura es limpia. La instalación, no.

Poner en marcha RTMDet desde la fuente oficial implica montar un stack de OpenMMLab de cuatro capas:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

Los rangos de versiones compatibles son estrechos. mmdet 3.3.0 requiere `mmcv < 2.2.0`, pero `mim install mmcv>=2.0.0` instala sin problema la versión 2.2.0, que luego falla en una aserción durante la ejecución. Tienes que fijar la versión manualmente.

Luego está el problema mayor. La versión más reciente de mmcv es la 2.2.0, publicada en abril de 2024 y sin actualizar desde entonces. Solo ofrece binarios precompilados hasta **torch 2.4 / CUDA 12.1**. Si hoy ejecutas `pip install torch` en una instalación nueva, obtienes PyTorch 2.12. Esa diferencia obliga a compilar las operaciones de C++/CUDA de mmcv desde el código fuente: entre 10 y 30 minutos, un toolkit de CUDA compatible, `nvcc` y un compilador de C++ compatible. De ahí viene la larga lista de errores documentados:

* `ModuleNotFoundError: No module named 'mmcv._ext'` -- las operaciones compiladas no se generaron o no son compatibles,

* `nvcc fatal: Unsupported gpu architecture 'compute_86'` en cualquier tarjeta RTX de la serie 30,

* `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'` -- el stack te obliga a volver a Python 3.8,

* fallo de segmentación al importar por una incompatibilidad entre versiones de GCC.

La propia FAQ de OpenMMLab indica que instales mmcv con pip en lugar de mim para evitar el problema de versiones. Su página Get Started indica que uses mim. Ambos documentos son oficiales. Se contradicen.

Una vez que el stack funciona, la inferencia sigue requiriendo elegir un archivo de configuración cuyo nombre codifica la receta de entrenamiento, además de descargar por separado un checkpoint con un nombre que incluye un hash y conectarlo mediante un registro que primero tienes que aprender a usar.

LibreYOLO sustituye todo eso:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
```

Sin `mim`, sin una matriz de compatibilidad de versiones entre cuatro paquetes, sin compilar desde el código fuente, sin archivos de configuración, sin buscar checkpoints por hash y sin tener que fijar Python 3.8. Los pesos se convierten desde los checkpoints originales de OpenMMLab y la inferencia es equivalente bit a bit a mmdetection con el mismo checkpoint.

Hay cinco variantes disponibles: Tiny, Small, Medium, Large y X. Todas funcionan a 640 px.

```python
print(results[0].boxes.xyxy)  # xyxy coordinates
print(results[0].boxes.conf)  # confidence scores
```

## Cuándo la opción original sigue siendo la adecuada

Si necesitas entrenar o hacer fine-tuning de RTMDet con todo el pipeline de aumento de datos de MMDetection, o ya trabajas a fondo en el ecosistema OpenMMLab y tienes mmcv funcionando, sigue con esa opción. LibreYOLO es la mejor alternativa para la inferencia y el despliegue.

## Una nota sobre la licencia

MMDetection y RTMDet usan Apache 2.0. El código de LibreYOLO usa MIT. Los pesos mantienen las mismas condiciones de Apache 2.0 que los originales. No hay restricciones comerciales.

## Pruébalo

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDetl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO usa la licencia MIT, funciona en Linux, Mac y Windows, y funciona en GPU, Apple Silicon y CPU sin cambiar el código. Una sola API cubre RTMDet, RT-DETR, RF-DETR, D-FINE, YOLOX, YOLO-NAS, segmentación, pose, profundidad y más.

Dale una estrella en GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentación: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
