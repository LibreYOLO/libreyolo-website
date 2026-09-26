---
title: Entrenar en una GPU alquilada
seo_title: Entrenar LibreYOLO en una GPU cloud alquilada
description: >-
  Ejecuta un entrenamiento de LibreYOLO en una GPU alquilada o serverless:
  prepara los datos, instala, lanza, míralo en vivo, recupera los pesos y deja
  de pagar.
lead: >-
  Una GPU alquilada convierte un entrenamiento en un job con un inicio, un final
  y una factura. El trabajo es el mismo que entrenar en local; lo que cambia es
  meter los datos, mirar desde fuera, sacar los pesos y apagar la máquina.
keywords:
  - entrenar en gpu cloud
  - alquilar una gpu
  - vast.ai entrenamiento
  - modal gpu serverless
  - beam gpu
  - entrenamiento remoto
  - subir dataset a hugging face
  - coste gpu por epoch
last_verified: 1.5.0
snippets:
  install:
    - label: En la máquina
      language: bash
      code: |
        pip install libreyolo

        # Añade solo los extras que necesite la ejecución. rfdetr para entrenar
        # RF-DETR, lora para fine-tuning eficiente, onnx para exportar después.
        pip install "libreyolo[rfdetr,lora]"
    - label: Comprueba la GPU antes que nada
      language: python
      code: |
        import torch

        print(torch.__version__, torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))

        # Una wheel compilada para otra arquitectura devuelve True y luego falla
        # en el primer kernel real, así que lanza uno.
        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  stage:
    - label: 'Empaqueta y sube una sola vez, desde tu máquina'
      language: bash
      code: >
        tar cf my-dataset.tar my-dataset/

        huggingface-cli upload my-org/my-dataset my-dataset.tar --repo-type
        dataset
    - label: Preparar los datos en la máquina
      language: python
      code: |
        import tarfile

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "my-org/my-dataset", "my-dataset.tar", repo_type="dataset"
        )
        with tarfile.open(path) as archive:
            archive.extractall("/root/data")
  launch:
    - label: 'En segundo plano, para que el job sobreviva a una desconexión'
      language: bash
      code: |
        nohup libreyolo train \
          model=LibreYOLO9s.pt \
          data=/root/data/my-dataset/data.yaml \
          epochs=100 batch=-1 imgsz=640 \
          project=/root/runs name=run1 \
          > /root/train.log 2>&1 &
    - label: 'Multi-GPU, desde un archivo Python'
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="/root/data/my-dataset/data.yaml",
                epochs=100,
                batch=64,          # batch global entre todas las GPUs
                device="0,1,2,3",
                project="/root/runs",
                name="run1",
            )
  watch:
    - label: Una lectura barata
      language: bash
      code: |
        cat /root/runs/run1/status.json
    - label: Desde un script
      language: python
      code: |
        import json

        with open("/root/runs/run1/status.json") as handle:
            status = json.load(handle)

        print(status["state"], status["current_epoch"], status["eta_seconds"])
        print(status.get("metrics"))
    - label: 'En el navegador, por un túnel SSH'
      language: bash
      code: |
        # En la máquina (escucha en 127.0.0.1:8420 por defecto):
        libreyolo monitor /root/runs/run1 --no-browser

        # Desde tu máquina, y luego abre http://localhost:8420 en local:
        #   ssh -L 8420:localhost:8420 <user>@<host>
  push:
    - label: Sube los pesos a un sitio permanente
      language: bash
      code: |
        huggingface-cli upload my-org/my-run \
          /root/runs/run1/weights/best.pt best.pt
source_hash: 75d314de06aca3b6
---

## Antes de alquilar nada

Dos decisiones cuestan más luego de lo que cuestan ahora.

Pon el dataset en un CDN primero. Empaquetarlo como un único tar en un
repositorio de datasets de Hugging Face funciona igual en todos los proveedores,
sirve rápido a todos ellos y no necesita más que un `HF_TOKEN` en el entorno del
job cuando el repositorio es privado. Subir un dataset desde una conexión
doméstica, o descargarlo desde un origen lento ya en la máquina, es tiempo de GPU
facturado que se va en esperar.

<code-tabs name="stage" />

Luego dimensiona el disco. Los proveedores que facturan almacenamiento facturan
por capacidad asignada, no por capacidad usada, y un disco no se puede reducir
después de crearlo. Suma los datos preparados, los checkpoints y en torno a un 30
por ciento de margen, y para ahí.

## Instalar en la máquina

<code-tabs name="install" />

Instala PyTorch primero si la imagen no trae ya una build de CUDA que encaje con
la tarjeta, y luego LibreYOLO, para que pip no resuelva su propio torch solo de
CPU. El segundo snippet no es ceremonia opcional: una wheel compilada para la
arquitectura de GPU equivocada devuelve `torch.cuda.is_available() == True` y
luego falla en la primera operación real con `CUDA error: no kernel image is
available for execution on the device`. Una sola multiplicación de matrices lo
detecta antes; una hora de instalación, no.

Apunta `HF_HOME` a almacenamiento persistente si el proveedor ofrece un volumen,
para que las descargas de checkpoints y datasets sobrevivan entre ejecuciones.

## Lanzar

Lanza el job en segundo plano. Una sesión interactiva que muere con tu conexión
de red se lleva el entrenamiento con ella.

<code-tabs name="launch" />

`batch=-1` merece la pena precisamente aquí, porque normalmente estás en una
tarjeta con la que no has entrenado antes. Sondea el modelo en modo entrenamiento
con un backward pass real y elige la mayor potencia de dos que cabe, lo cual es
más rápido que descubrir el techo con un error de falta de memoria veinte minutos
después. Consulta [Hiperparámetros](/docs/train/hyperparameters).

En una máquina multi-GPU, `device="0,1,2,3"` lanza por sí solo un worker por GPU,
y `batch` sigue siendo el batch global entre todas ellas. La guarda `__main__` es
obligatoria, porque cada worker vuelve a importar el script. Eso, y el resto del
comportamiento distribuido, está en
[Entrenamiento multi-GPU](/docs/train/multi-gpu).

## Míralo desde fuera

Cada ejecución escribe `status.json` en su directorio de ejecución, reescrito de
forma atómica en cada epoch. Es la lectura barata: unos pocos cientos de bytes
con el estado, la epoch actual, el ETA y las últimas métricas, sin parsear un
log.

<code-tabs name="watch" />

`metrics.jsonl` a su lado tiene el historial completo por epoch, y `train.log`
tiene la salida de consola. `libreyolo monitor` sirve un dashboard de navegador
sobre los tres usando solo la biblioteca estándar, así que no necesita nada
instalado en la máquina más allá del propio LibreYOLO. Accede a él mediante un
reenvío de puerto SSH.

Ninguna de estas opciones toca el proceso de entrenamiento, así que sirven para
engancharse a una ejecución en vivo, reabrir una terminada o inspeccionar una que
ha fallado.

## Saca los pesos antes de dejar de pagar

La máquina es desechable. Sube los checkpoints en hitos intermedios, no solo al
final, porque si no un fallo, una interrupción o quedarse sin crédito pierden la
ejecución entera.

<code-tabs name="push" />

`weights/best.pt` y `weights/last.pt` se escriben en cada epoch y en cada mejora.
`save_period=N` añade además snapshots `weights/epoch_<N>.pt`, que es lo que
abarata una subida a mitad de ejecución. `summary.json` y `results.csv`, donde la
familia los escriba, son pequeños y también merece la pena llevárselos.

Un callback en `on_train_epoch_end` es la forma limpia de automatizar la subida.
Consulta [Loggers de experimentos](/docs/train/loggers), donde los backends
alojados además te dan las métricas sin tocar la máquina para nada.

## Dejar de pagar

Esta es la parte que cuesta dinero de verdad cuando sale mal, y la regla cambia
según el modelo del proveedor.

En un marketplace donde alquilas una máquina en bruto, la facturación corre por
reloj de pared hasta que se destruye la instancia. Una GPU inactiva factura
exactamente igual que una ocupada, así que matar el proceso de entrenamiento no
ahorra nada por sí solo. Una instancia parada sigue facturando su disco.

En una plataforma serverless donde el job es una función decorada, el contenedor
escala a cero cuando la función retorna, así que es mucho menos probable
olvidarse una máquina encendida. Un job colgado sin timeout sigue facturando, así
que ponle siempre uno.

Parar en lugar de destruir es una palanca real, y una trampa real. Medido en una
máquina alquilada con 8x RTX 4090 y un disco de 250 GB el 31/07/2026: en marcha
facturaba 3,4828 $ por hora, parada facturaba 0,0694 $ por hora solo por el
disco, y destruida no facturaba nada. Eso es un ahorro del 98 por ciento
manteniendo en su sitio el entorno, los datos preparados y los checkpoints.

La tarifa de parada es una cuenta que puedes hacer antes de alquilar:

```text
stopped $/hr = allocated_GB * storage_cost_per_GB_per_month / 730
             = 250 * 0.20 / 730 = $0.0694/hr
```

Compárala con lo que cuesta reconstruir: volver a alquilar, descargar la imagen,
instalar y volver a preparar los datos. En esa misma máquina, una reconstrucción
salía por unos 15 minutos de instalación más 43 GB de transferencia entrante, en
torno a 1,00 $ en total. Frente a 0,0694 $ por hora, volver antes de unas 14 horas
favorece parar, y un hueco más largo favorece destruir y reconstruir desde la
copia preparada.

Un riesgo hace que parar sea inseguro con hardware escaso: parar libera las GPUs.
Nada las reserva, así que reiniciar solo funciona si el host todavía las tiene
libres. Tu disco está a salvo; tus GPUs no.

## Serverless, como una función

Si prefieres no gestionar una máquina, tanto Modal como Beam ejecutan una función
Python decorada en una GPU y escalan a cero cuando retorna. La propia suite de
tests nocturnos de LibreYOLO se ejecuta en Modal, y `tools/ci/modal_nightly.py`
en el repositorio de la biblioteca es el ejemplo real del que copiar.

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")   # bibliotecas de sistema de OpenCV
    .pip_install("libreyolo[rfdetr]")
)
app = modal.App("libreyolo-train")
cache = modal.Volume.from_name("libreyolo-cache", create_if_missing=True)


@app.function(gpu="A100", timeout=6 * 60 * 60, volumes={"/cache": cache})
def train():
    import os

    os.environ["HF_HOME"] = "/cache/hf"          # cachea los pesos entre ejecuciones

    from libreyolo import LibreYOLO

    model = LibreYOLO("LibreYOLO9s.pt")
    model.train(data="coco8.yaml", epochs=100, project="/cache/runs")
    cache.commit()                                # persiste el volumen


@app.local_entrypoint()
def main():
    train.remote()
```

Ejecútalo con `modal run modal_train.py`. El sistema de archivos del contenedor
es efímero, así que todo lo que merezca la pena guardar va al volumen o se sube
fuera. Pon `timeout=` explícitamente; es lo único que se interpone entre una
ejecución colgada y una factura sin límite.

Beam toma la misma forma con un decorador `@function`, un `Volume` y
`train.remote()` llamado desde `__main__`.

## Ajusta el tamaño según el coste por job

El $/hora es la cifra equivocada para optimizar. Un modelo pequeño deja a medio
gas una tarjeta grande, así que una GPU más barata y más lenta suele salir más
barata por epoch. Lanza el profiler durante unos pocos pasos en la tarjeta
alquilada antes de comprometerte a una ejecución larga: si el veredicto es
`dataloader` o `host / launch`, una GPU más rápida no aporta nada y más workers o
un batch más grande aportan mucho. Consulta
[Rendimiento del entrenamiento](/docs/train/performance).

## Relacionado

- [Datasets](/docs/train/datasets) para la estructura que debería tener el
  archivo preparado, y el comando doctor que detecta problemas antes de que una
  GPU esté facturando.
- [Entrenamiento multi-GPU](/docs/train/multi-gpu) para máquinas de varias
  tarjetas.
