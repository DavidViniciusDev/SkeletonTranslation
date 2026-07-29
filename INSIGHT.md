# A Ideia Central: Ensinar um Computador a Traduzir Libras Usando "Esqueletos" em vez de Vídeos

> Documento de intenção do projeto **SkelTrans** — tradução automática de Língua
> Brasileira de Sinais (Libras) contínua para Português escrito.
> Escrito para ser entendido por qualquer pessoa, mesmo sem conhecimento técnico
> de inteligência artificial.

---

## 1. O problema que queremos resolver

Milhões de brasileiros surdos se comunicam em **Libras**, uma língua completa e
independente do português — com gramática, ordem de palavras e expressões
próprias. Hoje, quase toda a informação escrita e falada do país está em
português, e a ponte entre os dois mundos depende de intérpretes humanos, que
são poucos e caros.

A pergunta que move este trabalho é:

> **É possível criar um "tradutor automático" que assista a alguém sinalizando em
> Libras e escreva, sozinho, a frase correspondente em português?**

Isso é o que a área chama de **Tradução de Língua de Sinais** (em inglês, *Sign
Language Translation* — SLT). É um problema difícil e pouco resolvido, porque
falta um ingrediente essencial: **dados**.

---

## 2. O grande obstáculo: faltam exemplos para o computador aprender

Uma inteligência artificial aprende por **exemplos**. Para ensinar um tradutor,
o ideal seria ter dezenas de milhares de vídeos de pessoas sinalizando **frases
inteiras**, cada um acompanhado da tradução escrita. Esse material praticamente
**não existe** para a Libras. Gravar tudo isso do zero custaria anos e milhões
de reais.

O que **existe** é diferente:

- **Um dicionário em vídeo** (a base *V-LIBRASIL*): cerca de **1.300 palavras
  isoladas**, cada uma gravada individualmente. É como ter um dicionário onde
  cada palavra tem seu vídeo, mas **nenhuma frase pronta**.
- **Um dicionário de frases em texto** (a base *pt-br2libras-gloss*): mais de
  **127.000 frases** já pareadas — a versão em português e a versão em Libras
  (escrita em "glosas", que são as palavras-sinal em ordem de Libras).

**A grande sacada do projeto é juntar essas duas bases.** Se eu tenho o vídeo de
cada palavra isolada, e tenho a receita de milhares de frases (quais palavras,
em que ordem), então eu posso **montar as frases artificialmente**, como quem
monta uma frase com peças de LEGO. Fabricamos os dados que não existem.

---

## 3. A decisão-chave: usar "esqueletos", não vídeos brutos

Aqui está o coração da nossa proposta, e o ponto que queremos defender.

Quando pensamos em ensinar um computador a "ver" alguém sinalizando, a ideia
óbvia é dar a ele o **vídeo bruto** — todos os pixels, todos os quadros. Mas isso
é extremamente pesado e ineficiente. Um vídeo carrega uma montanha de informação
que **não importa** para entender o sinal: a cor da camisa, a iluminação, o fundo,
o tom de pele, a qualidade da câmera. O computador gastaria um esforço gigantesco
só para aprender a **ignorar** tudo isso.

Nossa alternativa é converter cada vídeo em um **"esqueleto" de movimento**.

> **Analogia:** imagine trocar um filme por um **boneco de palito animado**.
> Em vez de guardar a imagem inteira, guardamos apenas alguns pontos-chave: onde
> estão as mãos, os dedos, os ombros, os olhos e a boca, quadro a quadro. É como
> aqueles pontinhos que os estúdios colam no corpo dos atores para criar
> animações digitais (*motion capture*).

Para uma língua de sinais, **isso é quase tudo o que importa**: o sinal está na
forma da mão, no movimento dos braços e na expressão do rosto — não na cor da
parede atrás. Ao reduzir o vídeo a **115 pontos por quadro**, jogamos fora o
ruído e ficamos com a essência.

### Por que isso é uma vantagem enorme

1. **Muito mais leve e barato (a tese de "IA Verde" / *Green AI*).**
   Um vídeo tem milhões de números por quadro; nosso esqueleto tem algumas
   centenas. Isso significa que o modelo pode ser treinado em uma **placa de
   vídeo comum de computador doméstico**, em vez de exigir supercomputadores
   caríssimos. Menos custo, menos energia, menos emissão de carbono.

2. **Foca no que interessa.** Ao remover fundo, roupa e iluminação, o modelo
   aprende o *movimento* diretamente, e tende a generalizar melhor.

3. **Facilita "fabricar" frases.** É muito mais fácil e natural **costurar
   esqueletos** de palavras diferentes para formar uma frase do que tentar
   colar pedaços de vídeos reais (que teriam cortes, saltos de iluminação e
   emendas visíveis).

---

## 4. O plano, em linguagem simples (os 6 passos)

Todo o projeto é uma linha de montagem que transforma **vídeos de palavras
soltas** em um **tradutor de frases**. Aqui está cada etapa sem jargão:

**Passo 1 — Organizar as bases.**
Arrumamos a "casa": padronizamos o texto das frases (tudo em maiúsculas, pontuação
limpa) e criamos um índice que liga cada palavra ao seu vídeo correspondente.
É a preparação dos ingredientes antes de cozinhar.

**Passo 2 — Transformar vídeo em esqueleto.**
Passamos cada vídeo do dicionário por um programa que detecta o corpo e as mãos
(a ferramenta *MediaPipe*, do Google) e anotamos, quadro a quadro, a posição dos
115 pontos-chave. O filme vira o boneco de palito.

**Passo 3 — "Limpar" e padronizar os esqueletos.**
Cada pessoa aparece em uma posição e distância diferente da câmera. Aqui nós
recentramos e reescalamos todos os esqueletos para um padrão único — como
alinhar todas as fotos no mesmo tamanho e enquadramento. Também calculamos a
**velocidade** de cada ponto, porque em língua de sinais o *movimento* diz tanto
quanto a *pose*.

**Passo 4 — Montar as frases (a inovação principal).**
Este é o passo mais original. Pegamos uma frase da base de texto (ex.: "QUALQUER
PESSOA"), buscamos o esqueleto de cada palavra e os **costuramos em sequência**.
O desafio: se apenas colássemos uma palavra na outra, o movimento daria um
"salto" artificial e feio entre os sinais. Então criamos uma **transição suave**
entre eles — a técnica que chamamos de *Keyframe Blending*.

> **Analogia:** é exatamente o que um desenho animado faz. O animador desenha
> algumas poses principais e o computador preenche os quadros do meio para o
> movimento parecer fluido. Nós fazemos o mesmo entre o fim de um sinal e o
> começo do próximo.

O resultado é um esqueleto contínuo que representa a **frase inteira sinalizada**,
mesmo que ela nunca tenha sido gravada de verdade.

**Passo 5 — O tradutor propriamente dito.**
Agora ensinamos o modelo. Ele tem duas partes:
- Um **"leitor" de esqueletos**, que assiste à sequência de movimentos e entende
  o que está sendo sinalizado.
- Um **"escritor" de português**, que produz a frase final.

O truque esperto aqui: para a parte que escreve português, **não começamos do
zero**. Reaproveitamos um modelo que **já sabe escrever português fluente** (o
*PTT5*, treinado por outros pesquisadores em milhões de textos). É como contratar
alguém que já é fluente no idioma e só precisa aprender a *nova tarefa* de ler os
esqueletos — em vez de alfabetizar alguém do nada. Isso economiza enormemente
dados e tempo.

**Passo 6 — Provar que funciona.**
Separamos parte das frases para **teste** (frases que o modelo nunca viu) e
medimos a qualidade da tradução com métricas reconhecidas na área (BLEU, ROUGE,
chrF). Além disso, fazemos um **experimento de controle**: comparamos o modelo
treinado com nossas transições suaves contra um treinado com a "colagem seca"
(sem suavização). Se o nosso for melhor, provamos que a inovação do Passo 4
realmente vale a pena.

---

## 5. Por que esta pesquisa importa (o argumento para os orientadores)

- **Resolve a escassez de dados de forma criativa.** Em vez de esperar por um
  dataset que não existe, nós o **sintetizamos** a partir de recursos que já
  temos. Essa é uma contribuição metodológica reaproveitável para **qualquer
  língua de sinais com poucos recursos** no mundo — não só a Libras.

- **É sustentável e acessível (*Green AI*).** Ao trabalhar com esqueletos em vez
  de vídeo, tornamos viável treinar tradutores de sinais **sem infraestrutura de
  ponta**. Democratiza a pesquisa: um laboratório modesto consegue reproduzir.

- **Ataca uma pergunta científica em aberto.** A grande dúvida é: *dados
  sintéticos (fabricados) conseguem treinar um modelo que funcione no mundo
  real?* Existe um abismo conhecido entre o artificial e o real (o chamado
  *domain shift*). Nosso trabalho investiga, mede e busca **reduzir esse abismo**
  — e essa é uma contribuição de valor mesmo que o resultado tenha limites.

- **Tem impacto social direto.** No fim da linha, isso é sobre **acessibilidade**
  e inclusão da comunidade surda, um tema de forte apelo em conferências de alto
  impacto (BRACIS, STIL e workshops internacionais de acessibilidade).

---

## 6. Em uma frase

> **Estamos ensinando um computador a traduzir Libras para português usando
> "esqueletos de movimento" no lugar de vídeos pesados — e fabricando as frases
> de treino a partir de um dicionário de palavras soltas, para superar a falta de
> dados de um jeito barato, sustentável e reaproveitável por outras línguas de
> sinais.**

Essa combinação — síntese de dados + representação esquelética leve +
reaproveitamento de um modelo de português já pronto — é o que torna a proposta
viável, original e defensável.
