# SkelTrans — Fundamentação Técnica

> **Tradução de Língua de Sinais (SLT) contínua Libras→Português a partir de
> landmarks 100% sintéticos.**
>
> Documento técnico de defesa da proposta. Público-alvo mínimo: mestrando em IA
> familiarizado com aprendizado profundo, arquiteturas Transformer e tradução
> automática neural (NMT). Complementa o `INSIGHT.md` (versão não-técnica) com o
> detalhamento matemático, arquitetural e experimental do pipeline.

---

## 1. Tese e formulação do problema

Buscamos aprender uma função de tradução

$$
f_\theta : \; \mathbf{X} \in \mathbb{R}^{T \times 115 \times C} \;\longrightarrow\; \mathbf{y} = (y_1, \dots, y_L),\quad y_i \in \mathcal{V}_{\text{pt}}
$$

onde $\mathbf{X}$ é uma sequência de $T$ frames de **landmarks esqueléticos**
($115$ pontos, $C$ canais por ponto) representando uma sentença sinalizada
contínua, e $\mathbf{y}$ é a sentença-alvo em português, uma sequência de tokens
sub-palavra do vocabulário $\mathcal{V}_{\text{pt}}$ do decoder.

Este é um problema **seq2seq cross-modal**: a "língua de origem" é uma série
temporal contínua e geométrica; a "língua de destino" é texto discreto. A
formulação é idêntica à de NMT, o que justifica a escolha de uma arquitetura
Encoder-Decoder com atenção.

**A restrição dominante é o regime *low-resource*.** SLT supervisionado exige
corpora de vídeo contínuo alinhados a texto (e.g. PHOENIX-14T para alemão), que
**inexistem em escala para a Libras**. A contribuição central da proposta é
contornar isso por **síntese de dados**: compor sentenças sinalizadas a partir de
um léxico de sinais isolados, guiada por um corpus de texto paralelo. O risco
inerente — o *domain shift* entre esqueletos sintetizados e sinalização real — é
o objeto de estudo experimental (Seção 8).

---

## 2. Ativos de dados e o insight de síntese

| Ativo | Natureza | Escala | Papel |
|---|---|---|---|
| **V-LIBRASIL** | Vídeos `.mp4` (chroma key) de sinais **isolados** | ~1.3k palavras, com augmentations | Fonte do léxico visual → landmarks |
| **pt-br2libras-gloss** (VLibrasBD) | Corpus paralelo **texto**: (PT, glosa Libras) | >127k sentenças | Fonte da estrutura sintática das frases |

Nenhuma das bases contém, isoladamente, o que a SLT supervisionada exige (vídeo
contínuo ↔ texto). O insight é **fatorar o problema**:

$$
P(\text{sentença sinalizada}) \approx \underbrace{P(\text{ordem/tokens})}_{\text{corpus textual}} \circ \underbrace{P(\text{sinal} \mid \text{token})}_{\text{léxico V-LIBRASIL}}
$$

O corpus textual fornece a *sequência de glosas* de dezenas de milhares de frases
reais; o léxico fornece a *realização esquelética* de cada glosa. A composição
das duas gera pares (landmarks contínuos, texto PT) em escala arbitrária — o
dataset que não existe.

### 2.1 Augmentation no nível do sinal

Cada glosa do V-LIBRASIL existe em múltiplas variantes já pré-computadas:
`none`, `upsample` (interpolação temporal → mais lento), `downsample` (mais
rápido), `horizontal-flip` (troca de mão dominante) e suas combinações. No
arquivo resolvido `pt_br2libras_gloss_sentence_videos.json`, cada sentença traz
um campo `base` (augmentation `none`) e uma lista `sentences` com as variantes
(≈17,8 variantes/sentença observadas). Isso multiplica o volume efetivo de dados
e introduz invariância a velocidade de execução e lateralidade.

---

## 3. Por que landmarks e não pixels — argumento formal

A escolha de representação esquelética não é estética; é uma decisão de **viés
indutivo e complexidade amostral**.

1. **Redução de dimensionalidade massiva.** Um frame RGB $224\times224$ tem
   ~150k dimensões; nosso frame esquelético tem $115 \times 9 = 1035$. Reduzimos
   ~2 ordens de grandeza *antes* de qualquer aprendizado, cortando drasticamente
   a complexidade amostral necessária — decisivo no regime low-resource.

2. **Invariâncias impostas por construção, não aprendidas.** Fundo, iluminação,
   textura de roupa, tom de pele e identidade do sinalizador são **descartados na
   extração**. Um modelo em pixels precisaria de dados e capacidade para aprender
   a marginalizar esses fatores; nós os eliminamos a priori. Isso ataca
   diretamente o *domain shift*: as fontes de variação nuisance mais óbvias entre
   "sintético" e "real" simplesmente não estão na representação.

3. **Composicionalidade.** Sinais em espaço de coordenadas são **geometricamente
   interpoláveis**. Concatenar e suavizar landmarks (Passo 4) é matematicamente
   bem-definido; "costurar" clipes de vídeo real produziria descontinuidades de
   pixel intratáveis. A representação esquelética é o que *viabiliza* a síntese.

4. **Green AI.** O custo computacional cai a ponto de tornar o treino viável em
   GPUs de consumo (RTX 3090/4090) ou instâncias T4/A10G, sustentando o argumento
   de democratização para línguas de recursos escassos.

O preço a pagar: a extração introduz ruído e perde informação sutil (contato,
oclusão fina). Tratamos isso explicitamente (Seções 4–5).

---

## 4. Pipeline — Passos 1 a 3 (extração e engenharia de características)

### Passo 1 — Sanitização e indexação
- **Texto/glosa:** caixa alta, remoção de pontuação não-gramatical, preservando
  marcadores de contexto (`?`, `!`). Tokenização das glosas em unidades-palavra.
- **Indexação léxica:** mapa glosa→caminho de vídeo. Artefatos derivados:
  `libras_gloss_vocab.txt`, `videos_words_by_word.json` (chave = palavra em caixa
  alta → lista de realizações em vídeo com caminho absoluto).

### Passo 2 — Extração espacial (`extract_landmarks.py`)
- **MediaPipe Holistic**, `model_complexity=2`, `refine_face_landmarks=True`,
  processando cada item frame a frame.
- **Fonte de frames desacoplada** (`skeltrans/extraction/frame_sources.py`): o
  passo consome uma *sequência de frames RGB*, sem saber de onde vêm. `open_frames`
  despacha pelo tipo do caminho — **arquivo de vídeo** (`.mp4`…, V-LIBRASIL) ou
  **diretório de imagens** (`.png`/`.jpg`… ordenadas, ex.: PHOENIX-2014T). Isso
  torna a extração agnóstica de base: uma nova fonte é um gerador novo, sem tocar
  no s3.
- **Máscara de 115 pontos** (redução deliberada dos ~543 do Holistic):

  | Bloco | Índices | Pontos | Conteúdo |
  |---|---|---|---|
  | Pose | `[0:33)` | 33 | corpo (X,Y,Z + visibilidade) |
  | Mão esquerda | `[33:54)` | 21 | landmarks da mão |
  | Mão direita | `[54:75)` | 21 | landmarks da mão |
  | Face | `[75:115)` | 40 | máscara manual: contorno de olhos, sobrancelhas e lábios |

  A seleção facial de 40 pontos é crítica: expressões não-manuais (sobrancelhas,
  boca) são **gramaticais** em Libras (marcam interrogação, negação,
  intensidade). Descartar a face inteira perderia sinal linguístico; manter os
  468 pontos do FaceMesh inflaria a dimensão sem ganho.
- **Tratamento de detecção ausente:** frames sem mão detectada recebem vetores
  zero `[0,0,0]` em vez de serem interpolados ou repetidos — evita "congelamento"
  fantasma e mantém uma semântica explícita de ausência.
- **Saída:** `landmarks_115/<video>.npy`, shape $(T, 115, 3)$, `float32`.

### Passo 3 — Normalização geométrica + dinâmica (`normalize_landmarks.py`)
Transforma $(T,115,3) \to (T,115,9)$, impondo invariância a translação/escala e
injetando dinâmica temporal. Por frame:

1. **Centralização.** Origem = ponto médio dos ombros (índices pose 11 e 12) —
   ou nariz (índice 0), configurável:
   $$\mathbf{P}'_{t,i} = \mathbf{P}_{t,i} - \mathbf{o}_t,\qquad \mathbf{o}_t = \tfrac{1}{2}(\mathbf{P}_{t,11}+\mathbf{P}_{t,12})$$
   Frames sem origem válida não são transladados ($\mathbf{o}_t = 0$).

2. **Escalonamento** pela distância biacromial (invariância de escala de câmera):
   $$s_t = \lVert \mathbf{P}_{t,11} - \mathbf{P}_{t,12} \rVert_2,\qquad \mathbf{P}''_{t,i} = \mathbf{P}'_{t,i}/s_t$$
   *Guard* de divisão por zero: se ombros ausentes ou $s_t < \varepsilon$, reusa a
   última escala válida (fallback 1.0) — mantém continuidade sem gerar `NaN`.

3. **Dinâmica temporal.** Derivadas de 1ª e 2ª ordem por diferenças finitas:
   $$\mathbf{V}_t = \mathbf{P}''_t - \mathbf{P}''_{t-1},\qquad \mathbf{A}_t = \mathbf{V}_t - \mathbf{V}_{t-1}$$
   Em língua de sinais o **movimento** é discriminativo; entregar velocidade e
   aceleração explicitamente alivia o encoder de reconstruí-las.

4. **Concatenação → 9 canais/ponto:** $[X,Y,Z,V_x,V_y,V_z,A_x,A_y,A_z]$.

**Propagação de máscara de ausência (detalhe não-trivial).** A máscara "presente"
é calculada da **entrada crua** (ponto ≠ vetor zero) e reimposta ao final: pontos
ausentes permanecem $[0,0,0]$ na posição — impede que a centralização os
"ressuscite" para uma posição espúria. A velocidade só é válida se o ponto está
presente em $t$ **e** $t-1$; a aceleração exige presença em $t,t-1,t-2$; caso
contrário são zeradas. Isso evita saltos fantasma nas derivadas ao redor de gaps
de detecção.

---

## 5. Passo 4 — Keyframe Blending (núcleo de inovação, `build_sentence_features.py`)

Dada a lista ordenada de sinais de uma sentença (já resolvida em vídeos por
4.1–4.3), sintetizamos uma **sequência esquelética contínua**.

### 5.1 Verificação de cobertura e taxa de descarte
Uma variante só é gerada se **todos** os seus sinais têm `.npy` disponível. O
primeiro sinal ausente descarta a variante inteira; o evento é contabilizado por
sinal faltante. O `manifest.stats.json` reporta `discard_rate_missing_landmark` e
o ranking `top_missing_signs` — métrica exigida pelo paper para caracterizar a
cobertura do léxico.

### 5.2 Interpolação de transição
Entre o último frame do sinal $A$ ($\mathbf{a}_{-1}$) e o primeiro do sinal $B$
($\mathbf{b}_0$), inserimos $K=5$ frames sintéticos (configurável). Dois modos:

- **LERP** (padrão):
  $$\mathbf{t}_k = (1-\alpha_k)\,\mathbf{a}_{-1} + \alpha_k\,\mathbf{b}_0,\qquad \alpha_k = \tfrac{k}{K+1},\; k=1..K$$

- **Spline cúbica** (`scipy.interpolate.CubicSpline`): usa uma janela de `anchor`
  frames de cada lado como nós, com um vão de $K$ no eixo temporal, e amostra o
  vão. Fornece continuidade $C^2$ (aceleração contínua) na emenda — mais físico
  que o LERP, que é apenas $C^0$.

- **`none`**: colagem seca (butt-join), sem frames de transição → é o **braço de
  controle do ablation study** (Seção 8).

**Interpolação ciente de presença.** Um ponto só é interpolado se estiver
presente **nos dois** lados da fronteira ($\text{present}(\mathbf{a}_{-1,i}) \wedge
\text{present}(\mathbf{b}_{0,i})$); caso contrário permanece $[0,0,0]$. Isso
impede movimento fantasma em direção à origem quando um lado tem a mão ausente —
coerente com a semântica de ausência do Passo 3. No modo cúbico, pontos sem
presença completa em todos os nós recorrem ao LERP.

### 5.3 Ordem de operações (decisão de projeto)
As derivadas $V,A$ são computadas **sobre a sequência já concatenada e
interpolada**, não por sinal isolado. Consequência: o vetor de velocidade
**atravessa as transições**, capturando a dinâmica inter-sinal — que é
justamente o que distingue sinalização contínua de uma sequência de sinais
soltos. Concretamente, `build_sentence_features.py` monta a sequência em
posições $(T,115,3)$ e só então aplica `normalize_array` do Passo 3 sobre o todo,
produzindo $(T,115,9)$. O `--feature-mode positions` permite exportar $(T,115,3)$
cru quando desejado.

### 5.4 Comprimento e contabilidade
Para uma frase de $n$ sinais com durações $T_1,\dots,T_n$:
$$T_{\text{total}} = \sum_{j=1}^{n} T_j + (n-1)\,K \quad \text{(modo lerp/cubic)}$$
O manifesto registra `sign_frames`, `transition_frames` e `num_frames` por
amostra, permitindo auditoria (a soma deve fechar).

### 5.5 Artefatos de saída e prevenção de vazamento
Por amostra gerada `sentXXXXXX_i{interp}_{aug}.npy`, três produtos:
- **`manifest.json`** — registro rico: `id`, `feature_file`, **`sentence_id`**,
  `pt_br`, `libras_gloss`, `tokens`, `interpreter`, `augmentation`, shape,
  contagem de frames, vídeos-fonte, + bloco `config` (K, modo, versão numpy) para
  reprodutibilidade.
- **`manifest.csv`** — versão plana para o `Dataset` do PyTorch.
- **`manifest.stats.json`** — estatísticas de descarte.

O campo **`sentence_id`** é a salvaguarda contra *data leakage*: **todas** as
variantes de augmentation de uma mesma frase compartilham o id, garantindo que o
split treino/val/teste (Passo 6) seja feito **por sentença** — nunca com a base
de uma frase no treino e sua versão `horizontal-flip` no teste.

---

## 6. Passo 5 — Arquitetura Encoder-Decoder

### 6.1 Encoder espacial-temporal
Entrada $\mathbf{X}\in\mathbb{R}^{T\times 1035}$ (achatando $115\times9$).

1. **Projeção + Conv1D.** Projeção linear $1035 \to D_{\text{model}}{=}512$
   seguida de convolução 1D temporal (`kernel_size=3`). A convolução impõe um
   viés indutivo de **localidade temporal**: mistura frames vizinhos antes da
   atenção, capturando micro-dinâmica (transição de configuração de mão) e
   suavizando jitter do MediaPipe. Complementa as features $V,A$ do Passo 3 com
   padrões locais aprendidos.
2. **Positional Encoding senoidal.** Self-attention é permutation-invariant;
   sem posição, a ordem temporal se perde. A variante senoidal (não-aprendida)
   não gasta parâmetros e **extrapola** para $T$ maiores que os vistos no
   treino — relevante dado o alto desvio de comprimento das frases.
3. **Pilha Transformer.** 4–6 `TransformerEncoderLayer`, *multi-head
   self-attention* com 8 cabeças. Modela dependências de longo alcance
   (concordância, referência espacial entre sinais distantes). A profundidade
   moderada é deliberada: um encoder superdimensionado memorizaria os artefatos
   das transições sintéticas em vez de generalizar.

Saída: $\mathbf{H}\in\mathbb{R}^{T\times 512}$, os "embeddings de origem".

### 6.2 Decoder textual por transfer learning (PTT5)
- Carregamos **PTT5-Base/Large** (T5 pré-treinado em português) do Hugging Face.
- **Enxerto na cross-attention:** substituímos a fonte da cross-attention do T5
  — que originalmente atende ao encoder de texto — por $\mathbf{H}$. Como o T5
  base opera em $d_{\text{model}}=768$ (Base), a saída do encoder de landmarks
  deve casar essa dimensão (projeção de compatibilização se $512 \neq 768$; ver
  Seção 7). O decoder pré-treinado passa a "ler" landmarks como uma língua
  estrangeira e decodifica autoregressivamente o texto PT.
- **Racional:** gerar português fluente do zero é inviável no nosso regime de
  dados. O PTT5 já carrega o *prior* linguístico (morfologia, sintaxe, fluência);
  restringimos o aprendizado à parte inédita — o alinhamento landmark→semântica.
  É o pilar low-resource/Green AI da proposta.

### 6.3 Otimização e regularização
| Componente | Escolha | Racional |
|---|---|---|
| Perda | Cross-entropy com `ignore_index` no padding | evita otimizar tokens de preenchimento em batches de comprimento variável |
| Otimizador | AdamW, LR $3\times10^{-5}$, *linear decay* | LR baixa típica de fine-tuning; evita *catastrophic forgetting* do PTT5 |
| Regularização | Dropout 0.2 no encoder | **defesa direta contra o próprio método**: força o modelo a não depender dos artefatos das transições LERP/spline, mitigando o *domain shift* sintético→real |

O dropout elevado no encoder é a peça conceitualmente mais importante: sabemos
que injetamos um viés artificial (transições suaves demais); a regularização
impede o modelo de explorá-lo como atalho.

---

## 7. Riscos técnicos e mitigações

| Risco | Impacto | Mitigação |
|---|---|---|
| **Domain shift** sintético→real | Modelo não generaliza para vídeo real | dropout no encoder; invariâncias por construção; ablation quantifica o efeito da síntese |
| Artefatos de transição (LERP $C^0$) | "Trapaça" do modelo em pistas não-linguísticas | opção de spline cúbica $C^2$; dropout; ablation seca vs. suave |
| Falhas de detecção do MediaPipe | Gaps e ruído nos landmarks | zero-fill explícito + máscara de presença propagada às derivadas |
| Cobertura léxica incompleta | Descarte de frases | taxa de descarte medida e reportada; síntese permite priorizar extração dos sinais mais frequentes |
| **Mismatch de dimensão** encoder (512) ↔ PTT5 (768) | Incompatibilidade na cross-attention | camada de projeção de compatibilização; ou fixar $D_{\text{model}}$ = dim do T5 |
| Distribuição de comprimento $T$ enviesada por augmentation | *Shortcut* de velocidade | up/downsample como augmentation balanceia; PE senoidal extrapola |
| **Leakage** entre variantes da mesma frase | Métricas infladas | split por `sentence_id` (Seção 5.5) |

---

## 8. Passo 6 — Protocolo experimental

- **Split estrito por sentença:** 80/10/10 (treino/val/teste), agrupando por
  `sentence_id` — nenhuma frase e suas augmentations cruzam a fronteira do split.
- **Métricas:** BLEU-4 (SacreBLEU), ROUGE-L, chrF — o conjunto padrão de SLT/NMT,
  cobrindo precisão n-gram, recall de subsequência longa e robustez a
  morfologia rica do português.
- **Ablation study (obrigatório):** comparação controlada
  **colagem seca (`--interp-mode none`)** vs. **pipeline completo (Keyframe
  Blending + normalização)**. É o experimento que isola e quantifica a
  contribuição da inovação central. Tabelas comparativas por métrica.
- Extensões naturais: variar $K$; LERP vs. cúbica; com/sem canais dinâmicos
  ($C{=}3$ vs. $C{=}9$); base-only vs. augmentation completa.

---

## 9. Contribuições e originalidade

1. **Metodológica:** um pipeline de **síntese de corpus de SLT contínuo** a
   partir de léxico isolado + corpus textual paralelo — reaproveitável por
   qualquer língua de sinais de baixos recursos.
2. **De representação:** validação de que landmarks normalizados $(T,115,9)$ com
   dinâmica explícita são suficientes para SLT end-to-end, dispensando pixels.
3. **Arquitetural:** enxerto de um encoder esquelético na cross-attention de um
   LLM de português pré-treinado (PTT5), unindo percepção geométrica e geração
   textual num único fluxo diferenciável.
4. **Empírica:** medição direta do *domain gap* sintético→real via ablation,
   contribuindo para a questão em aberto de treinar SLT com dados fabricados.

---

## 10. Reprodutibilidade

Todos os estágios são scripts determinísticos e parametrizáveis, com artefatos
versionáveis:

- `extract_landmarks.py` — Passo 2 → `landmarks_115/*.npy` $(T,115,3)$.
- `normalize_landmarks.py` — Passo 3 → `landmarks_115_norm9/*.npy` $(T,115,9)$.
- `build_sentence_features.py` — Passo 4 → `sentence_features/*.npy` + manifesto
  (JSON/CSV/stats). Parâmetros expostos: `--landmarks-dir`, `--out-dir`,
  `--transition-frames`, `--interp-mode {lerp,cubic,none}`,
  `--feature-mode {positions,normalized}`, `--variants {all,base}`, `--origin`.
- O `config` embutido em cada manifesto (parâmetros + versão numpy) fecha o
  rastro de proveniência de cada dataset gerado.

> **Síntese da defesa técnica:** reduzimos SLT a um problema NMT cross-modal
> tratável, geramos o dado inexistente por composição léxico×corpus sobre uma
> representação esquelética leve e interpolável, transferimos o conhecimento
> linguístico de um LLM pré-treinado, e isolamos experimentalmente a contribuição
> do *Keyframe Blending* — tudo sob orçamento computacional de GPU de consumo.
