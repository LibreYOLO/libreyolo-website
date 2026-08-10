---
title: Hailo
seo_title: "Ejecutar modelos LibreYOLO en aceleradores Hailo"
description: "Despliega un modelo LibreYOLO en un Hailo-8 o un Hailo-8L: la exportación ONNX estática, la etapa del Dataflow Compiler que ejecutas tú, y qué arquitecturas compilan."
lead: "Los aceleradores Hailo se compilan con el Hailo Dataflow Compiler, un SDK propietario que se distribuye a través de la Developer Zone de Hailo. La parte del flujo que le toca a LibreYOLO no es más que una exportación ONNX estática; el parseo, la cuantización y la compilación a un HEF ocurren después, dentro del DFC."
keywords:
  - libreyolo hailo
  - hailo-8
  - hailo-8l
  - raspberry pi ai kit
  - ai hat+
  - hailo dataflow compiler
  - compilar hef hailo
  - hailortcli
last_verified: "1.5.0"
meta:
  - label: Paso de LibreYOLO
    value: 'export(format="onnx", imgsz=640, dynamic=False)'
    mono: true
  - label: No es un formato
    value: 'No existe format="hef". El DFC no puede ser una dependencia de pip.'
  - label: Extra
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Host de compilación
    value: "Linux x86_64, incluido WSL2 con Ubuntu 22.04. La compilación no puede ejecutarse en ARM."
  - label: Compila
    value: "Grafos de CNN pura y forma fija. La atención, las formas dinámicas y los diseños dominados por LayerNorm, no."
  - label: Estado
    value: "Ninguna familia de LibreYOLO se ha llevado de principio a fin a través del DFC hasta un HEF en funcionamiento."
verification: "Leído de skills/libreyolo-export-hailo/SKILL.md, libreyolo/export/onnx.py y libreyolo/cli/commands/export.py en la rama dev. Las restricciones del DFC son las registradas en esa skill; no se ha compilado ni medido ningún HEF de LibreYOLO."
snippets:
  install:
    - label: Lado de LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Lado de Hailo, lo instalas tú
      language: text
      code: |
        Prerequisites, none of them installable from PyPI:

        - A Linux x86_64 machine. WSL2 Ubuntu 22.04 works. The Raspberry Pi is a
          runtime target, never the compile host.
        - The Dataflow Compiler wheel (hailo_sdk_client) from the Hailo Developer
          Zone, which is free to register for.
        - For Hailo-8 and Hailo-8L, the Hailo Model Zoo v2.x line, for its
          recipes and NMS configurations.
        - A GPU on the compile host is strongly recommended: the quantization
          step takes hours without one.
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Hailo necesita batch 1, una resolución fija y ningún eje dinámico.
        # La API de Python usa dynamic=True por defecto: desactívalo explícitamente.
        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    - label: CLI
      language: bash
      code: |
        # La CLI ya usa formas estáticas por defecto.
        libreyolo export --model LibreYOLOXs.pt --format onnx --imgsz 640
    - label: Confirmar que el grafo es estático antes de compilar
      language: python
      code: |
        import onnx

        graph = onnx.load("weights/LibreYOLOXs.onnx").graph
        shape = graph.input[0].type.tensor_type.shape
        print([d.dim_value or d.dim_param for d in shape.dim])
  compile:
    - label: Parsear, cuantizar y compilar
      language: python
      code: |
        from pathlib import Path

        import numpy as np
        from hailo_sdk_client import ClientRunner
        from PIL import Image

        ONNX = "weights/LibreYOLOXs.onnx"
        HW_ARCH = "hailo8"     # hailo8 | hailo8l | hailo10h
        IMGSZ = 640

        runner = ClientRunner(hw_arch=HW_ARCH)

        # Para YOLOX, traduce una vez sin end_node_names: el log del DFC imprime
        # los nodos finales que sugiere. Vuelve a ejecutar con esos.
        runner.translate_onnx_model(ONNX)

        # La normalización debe coincidir con el preprocesado de LibreYOLO. YOLOX
        # y YOLO9 no necesitan media ni desviación estándar, solo la escala de
        # 0-255 a 0-1.
        script = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"

        # Opcional: deja que Hailo se encargue de NMS. La configuración es
        # específica tanto del número de clases como del tamaño de entrada, así
        # que una config de COCO-80 es incorrecta para un modelo de tres clases
        # con fine-tuning. Sin esta línea el HEF emite los tensores en bruto de
        # la cabeza y la aplicación los decodifica.
        # script += 'nms_postprocess("yolox_nms_config.json", meta_arch=yolox, engine=cpu)\n'

        runner.load_model_script(script)

        # Las imágenes de calibración deben ser representativas de los datos de
        # despliegue. Con imágenes aleatorias la compilación funciona igual,
        # pero arruina la precisión en silencio.
        calib_paths = sorted(Path("calib_images").glob("*.jpg"))[:128]
        calib = np.stack([
            np.asarray(
                Image.open(p).convert("RGB").resize((IMGSZ, IMGSZ)),
                dtype=np.float32,
            )
            for p in calib_paths
        ])

        runner.optimize(calib)
        Path("libreyoloxs.hef").write_bytes(runner.compile())
    - label: Nodos finales de YOLO9
      language: python
      code: |
        # Los grafos de LibreYOLO usan el prefijo "/head/...", no el prefijo
        # "model.N" que aparece en configuraciones escritas para otras
        # exportaciones. Una config copiada no coincidirá. Confirma los nombres
        # en tu propio grafo si falla el parseo.
        END_NODES = [
            "/head/cv2.0/cv2.0.2/Conv", "/head/cv3.0/cv3.0.2/Conv",
            "/head/cv2.1/cv2.1.2/Conv", "/head/cv3.1/cv3.1.2/Conv",
            "/head/cv2.2/cv2.2.2/Conv", "/head/cv3.2/cv3.2.2/Conv",
        ]
        runner.translate_onnx_model(ONNX, end_node_names=END_NODES)
  device:
    - label: Raspberry Pi 5 con el AI Kit o el AI HAT+
      language: bash
      code: |
        sudo apt install dkms hailo-all
        hailortcli fw-control identify       # comprueba el dispositivo, y da el nombre de la arquitectura
        hailortcli run libreyoloxs.hef       # smoke test y throughput
---

## Instalación

En LibreYOLO no existe `format="hef"` ni va a existir. El Hailo Dataflow Compiler
es un SDK propietario que se distribuye como una wheel privada tras registrarse en
la Developer Zone, así que no puede ser una dependencia ni un extra. El despliegue
tiene dos etapas: LibreYOLO escribe un archivo ONNX estático, y tú ejecutas el DFC
sobre él.

```text
Libre<Model>.pt  ->  ONNX  ->  HAR (parse)  ->  HAR (quantize INT8)  ->  HEF
                 [libreyolo]           [Hailo DFC, installed by you]
```

<code-tabs name="install" />

## Exportación

<code-tabs name="export" />

No pases `half=True`. El DFC acepta ONNX en FP32 y hace su propia cuantización INT8.
Tampoco pases `nms=True`: de NMS se encarga Hailo mediante `nms_postprocess` o
bien la aplicación, y un subgrafo de NMS es peso muerto más allá de los nodos
finales. El opset por defecto funciona; si el parser del DFC protesta, vuelve a
exportar con `opset=11`.

El DFC corta el grafo en los nodos finales que le indicas, que son las
convoluciones de la cabeza de detección, y descarta todo lo que va después. Por
eso el ONNX decodificado normal de LibreYOLO es una entrada aceptable: el parser
simplemente ignora la cola de decodificación.

## Compilación

<code-tabs name="compile" />

Elige `hw_arch` según el destino: `hailo8` para el Hailo-8, el AI HAT+ de 26 TOPS
y los módulos M.2 y PCIe; `hailo8l` para el Hailo-8L, el Raspberry Pi AI Kit y el
AI HAT+ de 13 TOPS; `hailo10h` para el Hailo-10H, que necesita un DFC y un Model
Zoo más nuevos y compatibles entre sí. `hailortcli fw-control identify` en el dispositivo
responde a la pregunta cuando no lo tienes claro.

Dos familias encajan en una meta-arquitectura de NMS de HailoRT, así que Hailo
puede encargarse de la supresión dentro del pipeline compilado: YOLOX mediante
`meta_arch=yolox`, y YOLO9 mediante la meta-arquitectura de cabeza desacoplada de
Hailo, cuya disposición de cabeza es idéntica. Toma del Hailo Model Zoo la
configuración de `nms_postprocess` correspondiente y ajústala a tu número de
clases y a tu tamaño de entrada. Cualquier otro detector convolucional compila
como un grafo sin meta-arquitectura equivalente: el HEF emite los tensores en
bruto de la cabeza y la aplicación ejecuta la decodificación y NMS en la CPU.

Guarda el log de compilación cuando algo falle. Toda corrección depende del nombre
exacto de la capa o del operador que falla.

## Ejecutar el artefacto

<code-tabs name="device" />

La inferencia de la aplicación usa la API de Python `hailo_platform`. Con
`nms_postprocess` compilado dentro, la salida es `(batch, num_classes, max_dets, 5)`
y lleva `[y1, x1, y2, x2, score]` en coordenadas del modelo, que escalas de vuelta
a la imagen de origen tú mismo. El pipeline `Results` de LibreYOLO no interviene en
tiempo de ejecución; el HEF es un artefacto autónomo, y el preprocesado y el
posprocesado corren por cuenta de la aplicación.

## Restricciones

Que un modelo pueda apuntar a Hailo-8 o a Hailo-8L es una propiedad de su
arquitectura, no de su nombre, así que la regla de abajo se aplica también a las
familias añadidas después de escribir esta página.

Un modelo no compilará si contiene alguna de estas cosas:

- Atención de cualquier tipo: self, cross, deformable o por ventanas. Eso descarta
  todos los detectores estilo DETR, todos los detectores de vocabulario abierto o
  condicionados por texto, todos los backbones ViT, y todas las torres de lenguaje
  o de visión-lenguaje. El propio zoo de Hailo incluye unos pocos HEF de
  transformer ajustados a mano; eso es trabajo a medida del fabricante y no es
  prueba de que un grafo de atención cualquiera compile.
- Formas dinámicas o flujo de control dependiente de los datos. El DFC compila una
  única forma de entrada fija y un grafo estático, así que quedan fuera los
  recuentos variables de queries, los prompts de texto, el top-k dinámico,
  `NonZero`, `Gather` o `TopK` con índices dinámicos, y `grid_sample`.
- Un diseño dominado por LayerNorm o por GELU. BatchNorm se pliega limpiamente
  dentro de las convoluciones; el soporte de LayerNorm es pobre y GELU no es una
  activación nativa, así que una pila estilo ConvNeXt encaja mal aunque sea
  nominalmente convolucional.
- Trabajo de imagen a imagen a resolución nativa. Los modelos de restauración
  funcionan a resolución de entrada completa y superan los presupuestos prácticos
  de SRAM de Hailo.

Una familia es candidata cuando es solo convolución, usa BatchNorm con ReLU o
SiLU, y tiene un tamaño de entrada fijo. En esta biblioteca eso significa los
detectores CNN de una etapa, con YOLOX y YOLO9 como objetivos principales; otros
detectores convolucionales como PicoDet, YOLO-NAS y RTMDet, con decodificación del
lado de la aplicación; los clasificadores CNN ResNet, MobileNetV4-conv y
EfficientNetV2, de los cuales ResNet es el mejor soportado porque el Model Zoo de
Hailo incluye recetas para él; y cabezas de tarea convolucionales pequeñas como la
detección de puntos FOMO y la mirada L2CS sobre un backbone ResNet, que son
compilables en principio pero no tienen receta de Hailo.

Una salvedad sobre el estado, que es la razón por la que nada de esta página se
presenta como soportado: ninguna familia de LibreYOLO se ha llevado de principio a
fin por el DFC hasta un HEF en funcionamiento. Las reglas de arriba predicen la
compilabilidad a partir de la arquitectura. El comportamiento del parser, la
cuantización y la precisión siguen sin demostrarse hasta que se compile y se mida
un HEF, así que trata cada candidato como algo que requiere su propia evidencia
registrada: un HEF compilado desde el checkpoint exacto con las versiones de DFC,
Model Zoo y HailoRT anotadas, una calibración documentada, y una comparación de
precisión en el dispositivo contra la baseline FP32 en vez de un número de
throughput.

Si el modelo queda descartado, las alternativas son los runtimes con paridad
registrada: [ONNX](/docs/export/onnx), [TensorRT](/docs/export/tensorrt) y
[OpenVINO](/docs/export/openvino).
