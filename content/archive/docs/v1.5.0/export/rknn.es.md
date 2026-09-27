---
title: RKNN
seo_title: Exportar a RKNN para NPUs de Rockchip
description: >-
  Compila un detector LibreYOLO a un artefacto .rknn de Rockchip: el SDK del
  fabricante que instalas tú, las cuatro variantes validadas en RK3588 y la
  paridad en el simulador.
lead: >-
  RKNN es el formato compilado de NPU de Rockchip. LibreYOLO exporta un
  intermedio ONNX de opset 19, lo compila con el SDK RKNN Toolkit2 y puede
  comparar el grafo compilado contra ONNX Runtime en el simulador de host de
  Toolkit2 sin necesidad de placa.
keywords:
  - exportar yolo rknn
  - npu rockchip
  - rk3588
  - rknn-toolkit2
  - paridad simulador rknn
  - inferencia rockchip orange pi
last_verified: 1.5.0
meta:
  - label: Flag
    value: 'export(format="rknn", name="rk3588")'
    mono: true
  - label: Escribe
    value: >-
      Un archivo .rknn, un sidecar .rknn.metadata.json y un informe
      .rknn.parity.json cuando verify=True
  - label: Extra
    value: Ninguno en PyPI. rknn-toolkit2 es un SDK del fabricante que instalas tú.
  - label: Se recarga con
    value: >-
      No a través de LibreYOLO. El artefacto se ejecuta en la placa con el
      runtime de Rockchip.
  - label: Formas
    value: 'Cuadrada fija, batch 1, opset 19. Las tres se imponen.'
  - label: Precisión
    value: >-
      La build en coma flotante del fabricante. half=True e int8=True se
      rechazan.
  - label: Alcance
    value: >-
      Cuatro variantes de detección en RK3588: YOLO9-t, YOLO9-E2E-t, PicoDet-s y
      YOLO-NAS-s
verification: >-
  Leído de libreyolo/export/rknn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py y docs/rknn.md en la rama dev. Los números de
  paridad medidos vienen del registro de validación con fecha 2026-08-04 en
  docs/rknn.md.
snippets:
  install:
    - label: Lado de LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'SDK del fabricante, lo instalas tú'
      language: bash
      code: |
        # rknn-toolkit2 es un SDK de Rockchip con licencia aparte. LibreYOLO ni
        # lo empaqueta ni lo instala. Solo Linux x86_64; en Windows usa WSL2 o
        # un contenedor de Linux.
        #
        # Toolkit2 2.3.2 necesita setuptools<81 y falla con ONNX 1.19 o
        # superior, que eliminó onnx.mapping mientras su compilador lo importa.
        pip install "setuptools==80.9.0" "onnx==1.18.0"

        # Después instala la wheel de rknn-toolkit2 correspondiente desde el
        # repositorio de wheels de Rockchip, y confirma que importa:
        python -c "import rknn.api; print('rknn-toolkit2 ready')"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Escribe weights/LibreYOLO9t.rknn y
        weights/LibreYOLO9t.rknn.metadata.json

        path = model.export(format="rknn", name="rk3588", imgsz=640,
        verify=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format rknn --name rk3588 \
          --imgsz 640 --verify
    - label: Argumentos
      language: python
      code: |
        model.export(
            format="rknn",
            name="rk3588",     # plataforma de destino; target= y target_platform= también funcionan
            imgsz=640,         # debe coincidir con el lienzo registrado de la variante
            batch=1,           # cualquier otro valor lanza NotImplementedError
            dynamic=False,     # True lanza ValueError
            opset=19,          # cualquier otro valor lanza NotImplementedError
            verify=False,      # True ejecuta el simulador para PC y exige paridad
        )
  parity:
    - label: Paridad sin placa contra un artefacto ONNX que ya tengas
      language: python
      code: |
        import numpy as np
        from libreyolo.export import verify_rknn_simulator_parity

        input_tensor = np.random.default_rng(0).standard_normal(
            (1, 3, 640, 640), dtype=np.float32
        )
        metrics = verify_rknn_simulator_parity(
            "weights/LibreYOLO9t.onnx",
            input_tensor,
            target_platform="rk3588",
            rtol=1e-3,
            atol=1e-4,
            raise_on_failure=False,
        )
        print(metrics)
  support:
    - label: Comprobar una familia y una tarea antes de compilar
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c659713cc3c8cc9e
---

## Instalación

La compilación necesita el RKNN Toolkit2 de Rockchip, que se distribuye como un
SDK del fabricante bajo la licencia propia de Rockchip y no es una dependencia de
LibreYOLO. No existe un extra `libreyolo[rknn]`, y nada de este formato se
instala con una sola línea.

<code-tabs name="install" />

No hace falta una placa para compilar ni para comprobar la paridad numérica. Sí
hace falta una placa RK3588 para medir latencia, consumo y comportamiento
térmico, y ninguna de esas medidas está registrada.

## Exportación

<code-tabs name="export" />

La petición se valida contra una lista de variantes de modelo exactas antes de
compilar nada, y el lienzo se valida también: pasar un `imgsz` distinto de aquel
con el que se registró la variante lanza un error en lugar de compilar en
silencio algo sin probar. LibreYOLO escribe un intermedio ONNX de opset 19, lo
compila, opcionalmente lo simula y después elimina el intermedio.

Los metadatos van en un sidecar llamado `<model>.rknn.metadata.json`, porque el
formato RKNN no tiene un campo de metadatos portable.

`verify=True` ejecuta el simulador para PC de Toolkit2 dentro de la misma sesión
que compiló el artefacto, compara cada salida contra ONNX Runtime con la misma
entrada y escribe `<model>.rknn.parity.json` con métricas de error por salida.
Los umbrales son una similitud coseno de al menos 0.9999 y un RMSE normalizado de
como mucho 0.02, aplicados a cualquier salida que no sea ya cercana elemento a
elemento; la build en coma flotante del fabricante baja los tensores internos a
media precisión, así que un `allclose` estricto no se cumple ni siquiera cuando
los bounding boxes decodificados son estables. Una ejecución fallida escribe
`<model>.rknn.failed.parity.json`, descarta el candidato y deja intacta cualquier
exportación anterior que hubiera terminado bien en esa ruta.

Para comparar un artefacto ONNX que ya tengas, sin volver a exportar:

<code-tabs name="parity" />

El simulador de Toolkit2 ejecuta el grafo en memoria que producen `load_onnx` y
`build`. No puede recargar un archivo `.rknn` específico de un target sin una
placa, y por eso `verify=True` hace la compilación, la exportación y la
simulación en una sola sesión.

## Ejecutar el artefacto

No hay ninguna entrada de RKNN en `libreyolo/backends`, así que `LibreYOLO()` no
carga un archivo `.rknn`. El artefacto compilado se despliega en la placa y lo
ejecuta el runtime propio de Rockchip, y allí el preprocesado, el decodificado,
el NMS y el reescalado de coordenadas son responsabilidad de la aplicación.

`<model>.rknn.metadata.json` lleva los nombres de clase, el tamaño de entrada, la
tarea y la plataforma de destino, que es lo que una aplicación necesita para
reproducir el postprocesado de LibreYOLO. Distribúyelo junto al modelo compilado.

Para una comprobación en el host que no necesite la placa, guarda un artefacto
ONNX con la misma forma fija y compáralo en el simulador, como arriba.

## Restricciones

Compilan cuatro combinaciones, y son variantes de modelo en lugar de familias:

| Variante | Tarea | Lienzo | Target |
|---|---|---:|---|
| YOLO9-t | detect | 640 | RK3588 |
| YOLO9-E2E-t | detect | 640 | RK3588 |
| PicoDet-s | detect | 320 | RK3588 |
| YOLO-NAS-s | detect | 640 | RK3588 |

Todo lo demás se rechaza antes de compilar, con el mensaje de que RKNN en esta
versión se limita a las variantes de detección exactas probadas en el simulador.
Existen resultados de solo compilación para otros modelos, pero deliberadamente
no se presentan como soporte: en la misma tanda de medición, RF-DETR dejó dos
nodos `GridSample` del decoder sin bajar, y D-FINE, RT-DETR, RT-DETRv2,
RT-DETRv4, DEIM, DEIMv2 y EC compilaron y simularon con salidas decodificadas
materialmente incorrectas.

Batch 1, formas estáticas, opset 19. `half=True` se rechaza, porque RKNN no
expone el contrato `half` de LibreYOLO, e `int8=True` se rechaza hasta que
existan una calibración representativa y resultados de precisión por tarea.

Los demás targets de Rockchip se rechazan: `rk3588` es la única plataforma
validada.

Para la rejilla completa de familias y tareas, consulta
[la matriz de exportación](/docs/reference/export-matrix). Para una sola
combinación:

<code-tabs name="support" />
