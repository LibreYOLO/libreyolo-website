---
title: Solución de problemas
seo_title: Corregir los errores más comunes de LibreYOLO
description: >-
  Los errores que LibreYOLO lanza con más frecuencia, qué significa cada uno y
  cómo solucionarlo. Incluye dos fallos que producen resultados incorrectos en
  lugar de lanzar una excepción.
lead: >-
  Errores agrupados por el mensaje que ves. Las dos últimas entradas cubren el
  problema contrario: código que se ejecuta, devuelve algo plausible y está mal.
keywords:
  - error libreyolo
  - modulenotfounderror libreyolo
  - libreyolo cuda out of memory
  - libreyolo notimplementederror
  - solucionar errores libreyolo
  - libreyolo no descarga pesos
last_verified: 1.5.0
source_hash: e271ab29b789865a
---

Los errores están agrupados por el texto que ves. Si tu mensaje no está aquí,
las [FAQ](/docs/faq) responden a las preguntas que no son fallos, y
`libreyolo models` indica qué puede cargar realmente tu instalación.

## ModuleNotFoundError con un paquete que nunca importaste

Algunas familias necesitan un extra opcional. El mensaje nombra el paquete que
falta en lugar del extra, así que la solución no siempre resulta obvia a partir
del traceback.

Ejecuta `libreyolo models`. Toda familia a la que le falte una dependencia se
imprime junto al comando pip exacto que la habilita, de forma que no tengas que
deducir tú mismo a qué extra corresponde cada paquete. `libreyolo models --json`
imprime lo mismo como un objeto.

La [página de instalación](/docs/install) enumera todos los extras y lo que
cubre cada uno.

## ONNX inference requires onnxruntime

```
ImportError: ONNX inference requires onnxruntime. Install with: pip install onnxruntime
```

El paquete base no depende de ningún runtime, porque cuál te interesa depende de
tu hardware. Instala `onnxruntime` para CPU u `onnxruntime-gpu` para CUDA. Ambos
proporcionan el mismo módulo `onnxruntime`, así que instala uno, no los dos.

## ONNX model not found

```
FileNotFoundError: ONNX model not found: <path>
```

La ruta se resuelve respecto al directorio de trabajo, no respecto al script.
Esto también aparece cuando una exportación escribió en otro sitio sin avisar:
`export()` devuelve la ruta en la que escribió, así que captura el valor
devuelto en lugar de dar por supuesto un nombre.

## NotImplementedError desde train()

No todas las familias entrenan. Algunas están portadas solo para predicción,
validación y exportación, y su `train()` lanza una excepción en lugar de fingir
que se ejecuta.

La [entrada de las FAQ](/docs/faq) explica el razonamiento. Para comprobar una
familia concreta antes de escribir un script de entrenamiento, la página de su
modelo indica si entrena.

## NotImplementedError desde export()

Una familia puede soportar una tarea y aun así no exportarla. EoMT es el caso
con el que se topa la gente: `export()` acepta la tarea semántica y lanza una
excepción para `segment` y `panoptic`, porque el contrato de runtime de
query-mask que necesitan no está definido.

```
NotImplementedError: LibreEoMT instance and panoptic export need query-mask runtime contracts.
```

La página de cada familia incluye una matriz de exportación que muestra qué
combinaciones de tarea y formato están validadas.

## CUDA out of memory

Reduce primero `batch` y después `imgsz`. Ambos cambian la memoria de forma
aproximadamente proporcional a su tamaño, pero batch es el que puedes bajar sin
cambiar lo que ve el modelo.

Si falla en la validación y no en el entrenamiento, la validación usa su propio
tamaño de batch, así que baja también ese.

En Windows, una GPU que además maneja la pantalla tiene un segundo modo de fallo
que parece un error aleatorio de CUDA más que una falta de memoria: el driver
reinicia una GPU que deja de responder durante más tiempo que el timeout, y mata
lo que estuviera ejecutándose. Los kernels largos en la tarjeta que gobierna tu
monitor pueden dispararlo.

## Los pesos no se descargan

Los pesos se descargan de Hugging Face en el primer uso y se cachean en local.
Las [FAQ](/docs/faq) explican dónde vive la caché y cómo trabajar totalmente sin
conexión.

Si una descarga devuelve 404, revisa el nombre de archivo que pasaste. La URL se
deriva de él, incluido el sufijo de tarea, así que un nombre que no se
corresponda con un checkpoint publicado produce una URL que no existe. La tabla
de checkpoints de cada página de modelo enumera los nombres de archivo exactos
que se han publicado.

## El entrenamiento se queda colgado o se reinicia en Windows

Windows no tiene `fork`, así que los workers del dataloader arrancan
reimportando tu script. Sin una guarda `if __name__ == "__main__":`, cada worker
vuelve a ejecutar tu llamada de entrenamiento, lo que provoca un deadlock o
genera procesos sin fin.

```python
def main():
    ...  # construye el modelo y llama a train()

if __name__ == "__main__":
    main()
```

Poner `workers=0` también lo evita, a costa del rendimiento. La guarda es la
mejor solución.

## Dos fallos que no lanzan excepción

El resto de esta página trata sobre errores. Estos dos son peores, porque el
código se ejecuta y te devuelve algo que parece correcto.

### Indexar un único resultado

`predict()` devuelve un `Results` para una imagen, y una lista para varias.
Indexar el valor devuelto para una sola imagen selecciona una *detección*, no
una imagen:

```python
result = model.predict("image.jpg")   # un Results
result.boxes                          # todas las detecciones, correcto
result[0].boxes                       # UNA detección, en silencio
```

No se lanza nada, porque indexar un `Results` es una operación válida que
devuelve un subconjunto. El código escrito para la forma de lista informa en
silencio de un box por imagen. Indexa solo lo que sabes que es una lista.

### Leer las métricas como atributos

`val()` devuelve un diccionario simple con el nombre de la métrica como clave,
no un objeto con acceso por atributos:

```python
metrics = model.val(data="coco8.yaml")
metrics["metrics/mAP50-95"]   # correcto
metrics.box.map               # AttributeError
```

Las claves llevan los prefijos `metrics/` y `speed/`. Imprime el diccionario una
vez para ver qué ha producido tu tarea, ya que el conjunto varía según la tarea.

## Comprobar un dataset antes de entrenar

La mayoría de los fallos de entrenamiento son problemas del dataset.
`libreyolo doctor data.yaml` ejecuta comprobaciones de salud sobre un dataset de
detección e informa de los hallazgos por gravedad, lo que es más rápido que leer
un traceback de la primera época.

```python
from libreyolo import doctor

report = doctor.diagnose("data.yaml", imgsz=640)
if report.errors:
    ...
```

Consulta el [comando doctor](/docs/cli/doctor) para ver el catálogo de
comprobaciones.
