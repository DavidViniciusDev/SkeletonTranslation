# Encoder por grafo do esqueleto (`--encoder-type gcn`)

`--encoder-type gcn` troca a projeção de entrada do encoder por camadas de
convolução em grafo (GCN) sobre o esqueleto. Como no `parts`, só muda o jeito
de transformar cada frame no vetor de `d_model` números; a conv temporal, o
`--ds-steps`, o Transformer, o adaptador, o CTC e o decoder **não mudam**.

| `--encoder-type` | Projeção de entrada |
|---|---|
| `flat` (padrão) | um `Linear(1035, d_model)` sobre o frame achatado — o original |
| `parts` | uma MLP por parte do corpo + fusão ([ENCODER_PARTS.md](ENCODER_PARTS.md)) |
| `gcn` | camadas GCN sobre o grafo do esqueleto + média por parte + fusão |

Código: [`skeltrans/common/skeleton_graph.py`](skeltrans/common/skeleton_graph.py)
(o grafo) e
[`skeltrans/training/models/graph_projection.py`](skeltrans/training/models/graph_projection.py)
(a rede).

## Motivação

No `flat` e no `parts`, os pontos são colunas soltas: a rede não sabe que a
ponta do indicador está ligada à articulação de baixo, nem que o punho da mão
é o pulso do braço. No `gcn`, cada um dos 115 pontos é um **nó** com seus 9
canais (posição, velocidade, aceleração), e cada camada mistura a informação
de um nó só com os **vizinhos no esqueleto**. A forma da mão passa a ser
representada pela própria estrutura da rede — e é a forma da mão que costuma
separar sinais parecidos (as substituições, que dominam o WER de glosas).

```
frame (115 nós × 9 canais)
  -> 3 camadas GCN  (9 -> 64 -> 64 -> 128 canais por nó)
  -> média dos nós PRESENTES de cada parte: pose · mão esq · mão dir · face  (4 × 128)
  -> + fluxo das --extra-features (MLP, como no parts)
  -> concatena -> Linear -> d_model
```

Com `--d-model 512`: 128 canais por parte (`d_model / 4`) e 64 nas camadas
intermediárias (metade).

## O grafo

128 arestas, todas listadas à mão em `skeleton_graph.py`:

| Bloco | Arestas |
|---|---|
| Pose | as 35 conexões do MediaPipe Pose + nariz–cantos da boca (2) + nariz–ombros (2, o "pescoço") |
| Mãos | as 21 conexões do MediaPipe Hands, em cada mão |
| Face | 42 arestas pelos contornos dos 40 pontos de `FACE_MASK`: lábios (22), olhos (5 + 5), sobrancelhas (4 + 4), sobrancelha–pálpebra (2) |
| Entre blocos | pulso da pose → punho de cada mão (2); nariz → lábio superior e cantos internos dos olhos (3) |

- **Pescoço:** nas conexões oficiais do MediaPipe a cabeça não se liga ao
  tronco (e a boca não se liga a nada). Sem essas 4 arestas, rosto e face
  ficariam isolados do corpo — o teste de conectividade pegou isso.
- **Face:** as conexões oficiais do FaceMesh passam por pontos que não estão
  entre os 40 selecionados, então as arestas seguem os contornos restritos a
  eles.
- **Mãos não se ligam à face nem uma à outra** diretamente; a informação
  passa pelo corpo.

### Partições de vizinhança (ST-GCN)

Cada aresta entra numa de 3 partições, pela distância (em arestas) até os
ombros, que são o centro do corpo:

| Partição | Vizinho |
|---|---|
| 0 | o próprio nó e vizinhos à mesma distância do centro |
| 1 | vizinhos mais perto do centro (centrípetos: da ponta do dedo para a mão) |
| 2 | vizinhos mais longe (centrífugos: da mão para a ponta do dedo) |

Cada partição tem seus próprios pesos, então a rede distingue de onde vem a
informação. As matrizes são normalizadas como no GCN usual
(`D^-1/2 A D^-1/2`).

### Grafo aprendido

Uma matriz `B` (3 × 115 × 115, começando em zero) é somada ao grafo fixo, como
no AGCN/SL-GCN. A rede pode criar ligações que o esqueleto não tem — por
exemplo, ponta do dedo ↔ boca, comum em sinais feitos tocando o rosto. São
~40 mil parâmetros.

## Pontos ausentes

Um ponto não detectado chega como 9 zeros. Três cuidados evitam que isso
contamine o resto:

1. A agregação de cada partição é a **média só sobre os vizinhos presentes**:
   uma mão perdida pelo MediaPipe não puxa o pulso para zero.
2. Nós ausentes **continuam zerados** em todas as camadas.
3. A média por parte **ignora os nós ausentes**; uma parte inteira ausente
   (a mão sumiu) vira um vetor de zeros.

Com `--input-norm`, o z-score transforma o 0 de um ponto ausente em
`-média/desvio`. Por isso o encoder calcula a presença de cada ponto **antes**
da normalização e a entrega à projeção.

## Uso

```bash
# braços de comparação, mesmos hiperparâmetros no resto (uma pasta cada)
python3 slt_model.py ... --encoder-type flat  --out-dir ck_flat
python3 slt_model.py ... --encoder-type parts --out-dir ck_parts
python3 slt_model.py ... --encoder-type gcn   --out-dir ck_gcn

# variante: sem o grupo A (os ângulos dos dedos são parte do que o grafo
# deveria aprender sozinho; as duas coisas podem ser redundantes)
python3 slt_model.py ... --encoder-type gcn --extra-features B,C,F --input-norm
```

Conferir sem dados: `python3 slt_model.py --smoke-test --encoder-type gcn --use-ctc`.

## Custo

Medido no encoder isolado (RTX 5070 Ti, bf16, `--grad-checkpoint`, entrada com
`--extra-features A,B,C,F`, lote de 32 frases):

| Lote | `flat` | `gcn` |
|---|---|---|
| 110 frames por frase | 20 ms · 0,3 GB | 141 ms · 2,2 GB |
| comprimentos realistas, maior frase com 475 frames (76% de padding) | 44 ms · 0,6 GB | 191 ms · 2,3 GB |

- **Parâmetros:** a projeção tem **menos** parâmetros que a `flat` (0,46 M
  contra 0,64 M nessa entrada). Se houver ganho, ele vem da estrutura.
- **Tempo e memória:** a GCN processa 115 nós por frame em vez de um vetor,
  então custa bem mais que a `flat` no encoder. No passo de treino completo, o
  decoder T5 também pesa, então o aumento relativo é menor que o da tabela.
- **Padding:** as camadas GCN rodam **só nos frames reais**. Um frame de
  padding é todo zero, então sua saída é sempre a mesma e é copiada — o
  resultado é idêntico ao do lote inteiro (verificado nos testes), mas o custo
  deixa de depender da maior frase do lote. Sem isso, o lote de 475 frames
  acima precisaria de ~10 GB.
- **`--grad-checkpoint`** também recalcula as camadas GCN no backward
  (gradientes idênticos; economia modesta, ~350 MB no lote de 110 frames).
- Se faltar memória nas TITAN V (12 GB) com o decoder `nl36`, reduza o
  `--batch-size` (por exemplo, de 32 para 16).

## Checkpoints

- O tipo é detectado **pelos pesos** (chaves `encoder.input_proj.gcn_layers.*`),
  como no `parts`: `evaluate_slt.py` e `infer_slt.py` não precisam de flag.
- O grafo fixo vai no `.pt` (buffer `encoder.input_proj.A`), junto com o
  grafo aprendido: o checkpoint guarda exatamente o grafo com que foi treinado.
- Checkpoints `flat`, `parts` e `gcn` não são intercambiáveis.
- Com o padrão (`flat`) ou com `parts`, nada muda: arquitetura, `state_dict` e
  treino iguais aos de antes.

## Testes

```bash
python3 tests/test_encoder_gcn.py
```

Cobre:

- **grafo:** arestas conferidas contra uma lista escrita à mão, contagem por
  bloco, nenhuma ligação mão–face ou mão–mão, conectividade, distâncias ao
  centro, matriz simétrica e normalizada, cada aresta em exatamente uma
  partição;
- **pontos ausentes:** continuam 0, não diluem os vizinhos, mão inteira
  ausente vira 0 no pooling, presença calculada antes do `--input-norm`;
- **rede:** formas com e sem extras, mãos independentes dentro do alcance de 3
  camadas, gradiente em todos os pesos (inclusive o grafo aprendido), pular o
  padding dá saída e gradiente idênticos, `--grad-checkpoint` dá gradientes
  idênticos;
- **checkpoints:** salvar → carregar, tipo detectado pelos pesos, erro com
  grupos incompatíveis, `flat`/`parts` sem chaves do `gcn`.
