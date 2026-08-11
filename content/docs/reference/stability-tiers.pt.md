---
title: Níveis de estabilidade
seo_title: O que significa cada nível de suporte do LibreYOLO
description: >-
  O vocabulário de níveis que o LibreYOLO usa: os três níveis de suporte a
  exportação, os quatro níveis de API, os seis grupos de cobertura e o que
  nenhum deles promete.
lead: >-
  O LibreYOLO usa a palavra nível para três coisas diferentes: a evidência por
  trás de um caminho de exportação, o contrato de chamada que uma família de
  modelos atende, e o grupo de cobertura em que essa família está inscrita. Esta
  página define cada um deles e diz o que não está implícito.
keywords:
  - nível de suporte libreyolo
  - validated available blocked
  - níveis de exportação libreyolo
  - grupos de cobertura libreyolo
  - g0 g1 g2 g3 g4
  - níveis de modelos libreyolo
last_verified: 1.5.0
verification: >-
  Níveis de exportação de docs/adr/0011-export-support-tiers.md e
  libreyolo/export/support.py; grupos de cobertura e contagens por família de
  MODEL_GROUPS em libreyolo/models/registry.py; a verificação de treinamento do
  zero de libreyolo/models/base/model.py e libreyolo/cli/commands/train.py; o
  inventário da CLI lido de libreyolo/models/inventory.py; níveis de API dos
  docstrings dos pacotes libreyolo/models/sam/, openvocab/ e vlm/ e dos
  contratos de base.py, tudo na v1.5.0. Os rótulos de grupo que o leitor vê
  (Flagship, Core, Supported, Inference only, Museum, Sibling tier) são o
  vocabulário próprio do site para esses mesmos grupos, de
  src/data/docs/registry.json.
snippets:
  usage:
    - label: Ler as duas classificações de uma família
      language: python
      code: |
        from libreyolo.models.registry import GROUPS, group_of
        from libreyolo.export.support import get_support, validated_alternatives

        family = "yolo9"

        group = group_of(family)
        print(group, GROUPS[group])

        print(get_support(family, "detect", "onnx").tier)
        print(validated_alternatives(family, "detect"))
source_hash: de545894b0d125e4
---

## Níveis de suporte a exportação

O nível que decide se uma chamada tem sucesso. Ele se aplica à tripla
`(family, task, format)`, e toda combinação tem exatamente um.

| Nível | Significado | O que acontece no `export()` |
|---|---|---|
| `validated` | A paridade numérica é coberta no CI ou em uma execução noturna documentada | Roda |
| `available` | A conversão está implementada, mas não há evidência registrada de paridade numérica em runtime | Roda |
| `blocked` | Não há caminho suportado | Levanta `NotImplementedError` no preflight, com o motivo |

Validated e available seguem em frente sem pedir uma confirmação nem emitir um
aviso genérico. A diferença é a evidência, não a permissão: uma entrada
validated tem um teste de paridade por trás e uma versão `since`, e uma entrada
available ainda não. Uma conversão para CoreML sem uma execução de predição no
macOS, por exemplo, é available e não validated.

Uma combinação blocked falha antes das verificações de dependências, do
carregamento de calibração, do tracing ou da criação do artefato, então nada
parcial é escrito.

Cada célula validated carrega uma restrição que descreve a configuração de onde
veio o número de paridade, normalmente um canvas de entrada fixo, batch 1, FP32
e uma versão nomeada do runtime. Leia isso como uma afirmação sobre aquela
configuração, e não sobre o formato em geral. As regras que preenchem as células
sem entrada explícita estão na página da
[matriz de exportação](/docs/reference/export-matrix).

<code-tabs name="usage" />

## Níveis de API

O nível que decide com que cara uma chamada fica. Uma família fica em
exatamente um deles, escolhido pelo contrato de chamada e não pela arquitetura.

| Nível | Factory | Contrato |
|---|---|---|
| Factory de detectores | `LibreYOLO` | Um único forward sem prompt devolve todos os objetos que encontrou, com scores calibrados. Os membros se registram sozinhos reconhecendo um checkpoint |
| Segmentação com prompt | `LibreSAM` | Um forward não significa nada sem um prompt espacial ou de conceito por imagem, fornecido na hora da chamada. Interativo e com estado: codifique uma vez, faça prompts muitas vezes |
| Detecção de vocabulário aberto | `LibreOpenVocab` | Detectores discriminativos condicionados por texto. A lista de classes é um prompt, definida com `set_classes` |
| Visão-linguagem | `LibreVLM` | Um modelo generativo conduzido como detector. A lista de classes é um prompt e a confiança é um valor de preenchimento |

Os três níveis irmãos deliberadamente não se registram na factory de
detectores, e é por isso que `LibreYOLO("some-alias")` não chega até eles. Eles
carregam por alias de tamanho e download automático, e não por inspeção do
checkpoint.

Os quatro devolvem o mesmo `Results`, então o código que vem depois não muda
entre eles. O que difere é quais métodos funcionam: os níveis irmãos levantam
`NotImplementedError` em `train()`, `val()` e `export()`, e os níveis do SAM e
de vocabulário aberto levantam em `track()` também. A página de cada nível
lista as próprias exclusões.

## Grupos de cobertura

A classificação que decide quais famílias entram em uma execução de testes
entre famílias, e a que um leitor tem mais chance de encontrar em uma página de
modelo. Toda família registrada está inscrita em exatamente um grupo, e um teste
falha quando uma família registrada não aparece nessa inscrição. `GROUPS` em
`libreyolo/models/registry.py` é a fonte da coluna Significado abaixo;
`MODEL_GROUPS`, no mesmo arquivo, atribui cada família, e a coluna Famílias
conta essa atribuição diretamente. A coluna Rótulo é o nome mais curto que o
site usa para o mesmo grupo no cabeçalho de uma página de modelo.

| Grupo | Rótulo | Famílias | Significado |
|---|---|---|---|
| `g0` | Flagship | 2 | Âncoras principais obrigatórias na cobertura de recursos compartilhados |
| `g1` | Core | 10 | Conjunto de cobertura de detectores treináveis |
| `g2` | Supported | 14 | Conjunto adicional de cobertura de famílias treináveis |
| `g3` | Inference only | 35 | Famílias sem implementação de treinamento |
| `g4` | Museum | 5 | Famílias históricas com cobertura de inferência |
| `s` | Sibling tier | 21 | APIs irmãs (SAM, vocabulário aberto, VLM, zero-shot) cobertas separadamente |

São 87 famílias em seis grupos. Só o `g3` reúne mais famílias do que todos os
outros grupos juntos, porque a maior parte do registro é linhagem só de
inferência e cobertura de museu, e não detectores treinados ativamente.

Para um leitor escolhendo um modelo, o grupo diz onde esperar atenção de
engenharia, não o quão acurada uma família é. `g0` e `g1` são onde um recurso
novo é projetado e onde ele chega primeiro; `g2` é mantido verde no CI, mas um
recurso chega lá de forma oportunista e não na mesma onda de versão. `g3`
declara uma ausência, não um limite: predizer, validar e, quando a família dá
suporte, exportar continuam funcionando, e `train()` em uma família `g3` ou
`g4` levanta `NotImplementedError` nomeando o motivo em vez de fazer algo
parcial em silêncio. As famílias `s` não entram nesse compromisso de jeito
nenhum, porque carregam pela própria factory e não por `LibreYOLO()`.
Veja [conceitos básicos](/docs/concepts) para entender como um grupo se encaixa
ao lado de tarefa, família e tamanho na leitura do nome de arquivo de um
checkpoint.

Um grupo não concede nem restringe por si só nenhuma capacidade voltada ao
usuário. O suporte vem da API implementada da família e das verificações de
capacidade específicas de cada formato, nunca só do fato de pertencer a um
grupo. Os grupos classificam famílias, não tarefas, então uma execução de
cobertura limitada a uma tarefa nomeia a tarefa explicitamente, como em
"g1 detect".

Dois lugares leem o grupo em tempo de execução, e não só nos testes.
`collect_model_inventory()`, em `libreyolo/models/inventory.py`, anexa o grupo a
cada entrada que o inventário da CLI imprime, e `pretrained=False` dispara o
caminho especial de reinicialização do zero apenas para famílias em `g0` e
`g1`. Fora desses dois grupos, a verificação em
`libreyolo/models/base/model.py` é pulada por completo, então
`pretrained=False` chega ao `train()` da própria família como um argumento
nomeado comum.

## Treinamento

Uma família em `g3` ou `g4` não tem implementação de treinamento, e chamar
`train()` em uma delas levanta erro. Isso é uma propriedade do código da
família, não do grupo dela: o grupo registra o fato, não o causa.

Para uma família que treina, se um botão individual de data augmentation chega
ou não ao pipeline é uma questão separada, com o próprio vocabulário de três
valores, `used`, `gated_by_mosaic` e `ignored`. Veja a
[matriz de data augmentation](/docs/reference/augmentation-matrix).

## O que um nível não te diz

Um nível não é uma afirmação sobre acurácia. Uma exportação validated diz que o
artefato reproduz o modelo nativo dentro de um limiar declarado; não diz nada
sobre quão bem o modelo nativo pontua em um dataset. Os números de benchmark
ficam nas páginas dos modelos.

Um nível também não é uma declaração de licenciamento. As licenças dos pesos
variam dentro de uma mesma família e o repositório que hospeda um checkpoint
específico é quem manda. Uma família estar na factory de detectores não diz nada
sobre se os pesos publicados dela permitem uso comercial.
