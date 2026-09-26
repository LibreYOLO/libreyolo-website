---
title: Solução de problemas
seo_title: Corrigir os erros mais comuns do LibreYOLO
description: >-
  Os erros que o LibreYOLO lança com mais frequência, o que cada um significa e
  como corrigir. Inclui duas falhas que produzem resultados errados em vez de
  lançar uma exceção.
lead: >-
  Erros agrupados pela mensagem que você vê. As duas últimas entradas cobrem o
  problema oposto: código que roda, devolve algo plausível e está errado.
keywords:
  - erro libreyolo
  - modulenotfounderror libreyolo
  - libreyolo cuda out of memory
  - libreyolo notimplementederror
  - resolver erros libreyolo
last_verified: 1.5.0
source_hash: e271ab29b789865a
---

Os erros estão agrupados pelo texto que você vê. Se a sua mensagem não estiver
aqui, o [FAQ](/docs/faq) responde às perguntas que não são falhas, e
`libreyolo models` informa o que a sua instalação consegue realmente carregar.

## ModuleNotFoundError citando um pacote que você nunca importou

Algumas famílias precisam de um extra opcional. A mensagem cita o pacote que
falta, e não o extra, então a correção nem sempre é óbvia pelo traceback.

Rode `libreyolo models`. Toda família com dependência faltando é impressa junto
com o comando pip exato que a habilita, então você não precisa mapear o pacote
de volta para o extra por conta própria. `libreyolo models --json` imprime a
mesma coisa como um objeto.

A [página de instalação](/docs/install) lista todos os extras e o que cada um
cobre.

## ONNX inference requires onnxruntime

```
ImportError: ONNX inference requires onnxruntime. Install with: pip install onnxruntime
```

O pacote base não depende de um runtime, porque qual deles você quer depende do
seu hardware. Instale `onnxruntime` para CPU ou `onnxruntime-gpu` para CUDA. Os
dois fornecem o mesmo módulo `onnxruntime`, então instale um, não os dois.

## ONNX model not found

```
FileNotFoundError: ONNX model not found: <path>
```

O caminho é resolvido em relação ao diretório de trabalho, não ao script. Isso
também aparece quando uma exportação gravou em outro lugar sem avisar:
`export()` devolve o caminho que gravou, então capture o valor de retorno em vez
de presumir um nome.

## NotImplementedError vindo de train()

Nem toda família treina. Algumas foram portadas apenas para predição, validação
e exportação, e o `train()` delas lança um erro em vez de fingir que roda.

A [entrada do FAQ](/docs/faq) explica o motivo. Para conferir uma família
específica antes de escrever um script de treinamento, a página do modelo diz se
ela treina.

## NotImplementedError vindo de export()

Uma família pode suportar uma tarefa e ainda assim não exportá-la. O EoMT é o
caso em que as pessoas esbarram: `export()` aceita a tarefa semantic e lança
erro para `segment` e `panoptic`, porque o contrato de runtime de query-mask que
essas tarefas exigem não está definido.

```
NotImplementedError: LibreEoMT instance and panoptic export need query-mask runtime contracts.
```

A página de cada família traz uma matriz de exportação mostrando quais
combinações de tarefa e formato são validadas.

## CUDA out of memory

Reduza `batch` primeiro, depois `imgsz`. Os dois alteram o uso de memória mais
ou menos na proporção do seu valor, mas o batch é o que você pode diminuir sem
mudar o que o modelo enxerga.

Se a falha acontecer na validação e não no treinamento, a validação usa o
próprio tamanho de batch, então reduza esse também.

No Windows, uma GPU que também controla a tela tem um segundo modo de falha que
parece um erro aleatório de CUDA em vez de falta de memória: o driver reinicia
uma GPU que para de responder por mais tempo que o timeout, matando o que
estivesse rodando. Kernels longos na placa que controla o seu monitor podem
disparar isso.

## Os pesos não baixam

Os pesos são buscados no Hugging Face no primeiro uso e ficam em cache
localmente. O [FAQ](/docs/faq) explica onde fica o cache e como rodar totalmente
offline.

Se um download der 404, confira o nome de arquivo que você passou. A URL é
derivada dele, incluindo o sufixo da tarefa, então um nome que não corresponde a
um checkpoint publicado produz uma URL que não existe. A tabela de checkpoints
em cada página de modelo lista os nomes de arquivo publicados exatos.

## O treinamento trava ou reinicia no Windows

O Windows não tem `fork`, então os workers do dataloader começam reimportando o
seu script. Sem uma guarda `if __name__ == "__main__":`, cada worker roda de
novo a sua chamada de treinamento, o que trava em deadlock ou gera processos sem
parar.

```python
def main():
    ...  # monte o modelo e chame train()

if __name__ == "__main__":
    main()
```

Definir `workers=0` também evita isso, ao custo de throughput. A guarda é a
solução melhor.

## Duas falhas que não lançam erro

O resto desta página fala de erros. Estas duas são piores, porque o código roda
e devolve algo que parece certo.

### Indexar um único resultado

`predict()` devolve um `Results` para uma imagem, e uma lista para várias.
Indexar o retorno de imagem única seleciona uma *detecção*, não uma imagem:

```python
result = model.predict("image.jpg")   # um Results
result.boxes                          # todas as detecções, correto
result[0].boxes                       # UMA detecção, silenciosamente
```

Nada é lançado, porque indexar um `Results` é uma operação válida que devolve um
subconjunto. Código escrito para a forma de lista relata silenciosamente um
bounding box por imagem. Indexe apenas o que você sabe que é uma lista.

### Ler as métricas como atributos

`val()` devolve um dicionário simples com chaves de nome de métrica, não um
objeto com acesso por atributo:

```python
metrics = model.val(data="coco8.yaml")
metrics["metrics/mAP50-95"]   # correto
metrics.box.map               # AttributeError
```

As chaves têm os prefixos `metrics/` e `speed/`. Imprima o dicionário uma vez
para ver o que a sua tarefa produziu, já que o conjunto varia por tarefa.

## Conferir um dataset antes de treinar

A maioria das falhas de treinamento são problemas de dataset. `libreyolo doctor
data.yaml` roda verificações de saúde sobre um dataset de detecção e informa os
achados por severidade, o que é mais rápido do que ler um traceback da primeira
época.

```python
from libreyolo import doctor

report = doctor.diagnose("data.yaml", imgsz=640)
if report.errors:
    ...
```

Veja o [comando doctor](/docs/cli/doctor) para o catálogo de verificações.
</content>
</invoke>
