# Contexto do Projeto: Sign Language Translation (SLT) Baseado em Landmarks Sintéticos

## 1. Visão Geral e Objetivo do Paper
O objetivo desta pesquisa científica é investigar a viabilidade, os limites e a eficiência do uso de dados de movimento esquelético (landmarks) 100% sintéticos para o treinamento de modelos de **Sign Language Translation (SLT)** — tradução direta de língua de sinais contínua para texto em linguagem natural (Português).

Em vez de focar na manipulação pesada de pixels de vídeo brutos, o trabalho propõe uma abordagem de **Green AI** e redução de dimensionalidade: extrair coordenadas geométricas tridimensionais de um dataset de sinais isolados, aplicar um algoritmo inédito de interpolação temporal (*Keyframe Blending*) guiado por um corpus de texto paralelo, e treinar um modelo *End-to-End* (Encoder-Decoder).

O *paper* será submetido a conferências de alto impacto (como BRACIS/STIL ou workshops internacionais de acessibilidade), defendendo a tese de que pipelines de síntese de dados baseados em esqueletos diminuem o abismo entre dados artificiais e reais (*Domain Shift*), democratizando o treino de modelos de tradução para línguas de recursos escassos (*Low-Resource Languages*) como a Libras.

---

## 2. Ativos e Recursos Disponíveis

* **Dataset Textual (`pt-br2libras-gloss` / VLibrasBD):** Um corpus paralelo contendo mais de 127.000 sentenças alinhadas entre Português Brasileiro e a estrutura de Glosas da Libras.
* **Dataset de Vídeo (`V-LIBRASIL`):** Uma base de dados pública contendo vídeos em formato `.mp4` (gravados em chroma key) para cerca de 1.364 sinais/palavras isoladas em Libras.
* **Poder Computacional Alvo:** Treinamento viável em GPUs comerciais de consumo ou instâncias de nuvem padrão (ex: RTX 3090/4090 ou instâncias T4/A10G), viabilizado pela representação esquelética.

---

## 3. Plano de Implementação Detalhado

O agente de IA deve seguir rigorosamente os passos abaixo para construir o pipeline de código, engenharia de dados e arquitetura de rede do projeto.

### Passo 1: Preparação do Ambiente e Sanitização das Bases
1. Configurar o ambiente em Python 3.10+ utilizando `PyTorch`, `Hugging Face (transformers, accelerate, evaluate)`, `MediaPipe` e `OpenCV`.
2. Normalizar o texto do `pt-br2libras-gloss`: converter todas as glosas para CAIXA ALTA e remover pontuações textuais que não correspondam a marcações gramaticais da Libras, preservando apenas as *tags* de contexto essencial (como `?` ou `!`).
3. Indexar o `V-LIBRASIL`: criar um mapeamento chave-valor onde a chave é a palavra em glosa e o valor é o caminho do arquivo de vídeo correspondente.

### Passo 2: Extração Espacial (Landmark Extraction)
1. Construir um script utilizando `mediapipe.solutions.holistic` (`model_complexity=2`, `refine_face_landmarks=True`) para processar cada vídeo isolado do V-LIBRASIL frame a frame.
2. Extrair e filtrar apenas os seguintes pontos esqueléticos (totalizando 115 pontos por frame):
    * **Pose:** 33 pontos ($X, Y, Z$ + visibilidade).
    * **Mão Esquerda / Mão Direita:** 21 pontos cada ($X, Y, Z$).
    * **Face:** Selecionar manualmente uma máscara de **40 pontos** focada estritamente nos contornos dos olhos, sobrancelhas e lábios (responsáveis pelas expressões não-manuais).
3. Tratar omissões: frames onde o MediaPipe falhar em detectar as mãos devem ser preenchidos com vetores de zero `[0.0, 0.0, 0.0]` para evitar artefatos de congelamento.
4. Salvar as matrizes resultantes em arquivos compactados do NumPy (`.npy`) com o shape $(T \times 115 \times 3)$, indexados por palavra.

### Passo 3: Engenharia de Características (Normalização Geométrica)
Para garantir a invariância do modelo à escala da câmera e posição do sinalizador, aplicar as seguintes transformações matemáticas em cada frame:
1.  **Centralização:** Escolher o landmark do nariz ou o ponto médio entre os ombros como a origem $(0,0,0)$ e subtrair esse valor de todos os outros 114 pontos do frame.
2.  **Escalonamento:** Calcular a distância Euclidiana entre o ombro esquerdo e o direito e dividir todas as coordenadas do frame por essa distância.
3.  **Features Temporais (Dinâmica):** Calcular as derivadas temporais de primeira e segunda ordem para cada ponto: Velocidade ($\mathbf{V}_t = \mathbf{P}_t - \mathbf{P}_{t-1}$) e Aceleração ($\mathbf{A}_t = \mathbf{V}_t - \mathbf{V}_{t-1}$).
4. Concatenar os vetores gerando uma matriz de características final de dimensão 9 por ponto ($X, Y, Z, V_x, V_y, V_z, A_x, A_y, A_z$).

### Passo 4: Algoritmo de Síntese de Frases e Interpolação (Keyframe Blending)
Este módulo é o núcleo de inovação da pesquisa:
1. Ler uma sentença em glosa do `pt-br2libras-gloss` e quebrar em tokens (palavras isoladas).
2. Verificar a presença de todos os tokens no dicionário do V-LIBRASIL. Caso falte algum sinal, a frase deve ser descartada (documentando a taxa de descarte para o *paper*).
3. Para frases válidas, criar uma janela de transição fixa de $K = 5$ frames entre o final do sinal atual e o início do próximo sinal.
4. Aplicar **Interpolação Linear (LERP)** ou **Splines Cúbicas** (`scipy.interpolate`) nas coordenadas dos landmarks para suavizar a transição do movimento físico do esqueleto entre os sinais isolados, eliminando cortes abruptos.
5. Concatenar as sequências gerando um arquivo contínuo consolidado que represente a frase inteira sinalizada.

### Passo 5: Arquitetura e Treinamento do Modelo de SLT
A rede deve seguir uma arquitetura Encoder-Decoder baseada em Transformers:
1.  **Encoder Espacial-Temporal:**
    * Camada de projeção linear + Convolução 1D (kernel_size=3) para mapear os 115 pontos (com dimensão 9) para o tamanho oculto da rede ($D_{model} = 512$).
    * Aplicação de *Sine/Cosine Positional Encoding*.
    * Empilhamento de 4 a 6 camadas de `TransformerEncoderLayer` com *Multi-Head Self-Attention* (8 cabeças).
2.  **Decoder Textual (Transfer Learning):**
    * Carregar o modelo pré-treinado **PTT5-Base** (ou Large) do Hugging Face.
    * Conectar a saída do Encoder de Landmarks diretamente ao mecanismo de *Cross-Attention* do Decoder do PTT5. O Decoder será treinado para receber as features esqueléticas e cuspir o texto final traduzido em português escrito.
3.  **Hyperparâmetros e Regularização:**
    * Função de Perda: *CrossEntropyLoss* (ignorando padding tokens).
    * Otimizador: AdamW com taxa de aprendizado de $3\times10^{-5}$ e *linear decay*.
    * Aplicar *Dropout* de 0.2 nas camadas do encoder para mitigar o superajuste (*overfitting*) às transições sintéticas.

### Passo 6: Protocolo de Avaliação Experimental (Métricas do Paper)
1.  **Divisão Estrita de Dados:** Separar o corpus unificado em 80% treino, 10% validação e 10% teste. A separação deve ser feita a nível de sentenças completas para evitar vazamento de dados (*data leakage*).
2.  **Métricas de Tradução:** Avaliar o conjunto de teste utilizando **BLEU-4** (via SacreBLEU), **ROUGE-L** e **chrF**.
3.  **Estudo de Ablação (Ablation Study):** Rodar um experimento de controle comparando obrigatoriamente a performance do modelo treinado com a colagem de sinais "seca" (sem interpolação) versus o modelo treinado com o pipeline completo (com o algoritmo de *Keyframe Blending* e normalizações). Os resultados devem ser estruturados em tabelas comparativas.

---

## 4. Próximos Passos para o Agente de IA
1. Escrever o script Python focado no **Passo 2** e **Passo 3** para extração, limpeza e normalização geométrica dos vídeos do V-LIBRASIL.
2. Desenvolver o algoritmo matemático de interpolação do **Passo 4**.
3. Estruturar a classe `Dataset` do PyTorch e o loop de treino do Transformer do **Passo 5**.