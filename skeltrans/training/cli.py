"""Interface de linha de comando do treino/diagnóstico do modelo SLT.

Traduz os argumentos para ModelConfig/DataConfig/TrainConfig e despacha para o
smoke-test ou para o treino. O núcleo (models/trainer) não conhece o argparse.
"""

import argparse
import sys

from skeltrans.training.config import DataConfig, ModelConfig, TrainConfig
from skeltrans.training.diagnostics import smoke_test
from skeltrans.training.trainer import build_and_train

_MC, _DC, _TC = ModelConfig(), DataConfig(), TrainConfig()


def build_parser():
    ap = argparse.ArgumentParser(description="Passo 5 — Modelo SLT (LandmarkEncoder + PTT5 decoder).")
    ap.add_argument(
        "--smoke-test", action="store_true",
        help="Valida a arquitetura ponta-a-ponta (forward/backward/generate) com um T5 "
             "minusculo e dados aleatorios, na CPU, SEM baixar pesos nem exigir dados reais. "
             "Use para conferir a instalacao e a integracao encoder<->decoder rapidamente.")

    # ------------------------------ dados ------------------------------ #
    ap.add_argument(
        "--train-manifest",
        help="Caminho do manifesto JSON de TREINO. Aceita a lista plana "
             "[{features, text}, ...] ou o formato do Passo 4 "
             "{config, items:[{feature_file, pt_br}, ...]}. Cada item aponta para um .npy "
             "(T,115,9) de uma FRASE e seu texto-alvo em portugues. Obrigatorio para treinar "
             "(dispensado quando --smoke-test e usado).")
    ap.add_argument(
        "--val-manifest",
        help="Manifesto JSON de VALIDACAO (mesmo formato do treino). Ao fim de cada epoca "
             "mede a perda de validacao e salva o melhor checkpoint (best.pt). Opcional: sem "
             "ele, nao ha selecao do melhor modelo nem best.pt.")
    ap.add_argument(
        "--features-dir", default=_DC.features_dir,
        help="Pasta onde estao os .npy de features. Se informado, cada feature e buscada como "
             "<features-dir>/<basename>, ignorando o caminho gravado no manifesto. Util quando "
             "o manifesto e os .npy estao em pastas diferentes. Se omitido, caminhos relativos "
             "sao resolvidos em relacao ao proprio manifesto (e absolutos usados como estao).")

    # ------------------------------ modelo ----------------------------- #
    ap.add_argument(
        "--t5", default=_MC.t5_name,
        help="Identificador no Hugging Face (ou caminho local) do decoder pre-treinado a ser "
             "fine-tunado. Padrao: PTT5-base em portugues. O hidden desse modelo define o "
             "tamanho para o qual o adaptador do encoder projeta a memoria (cross-attention).")
    ap.add_argument(
        "--d-model", type=int, default=_MC.d_model,
        help="Dimensao oculta do Encoder de landmarks (projecao linear + Transformer). O plano "
             "usa 512. Se diferir do hidden do T5, um adaptador linear faz a ponte "
             "automaticamente. Deve ser divisivel por --nhead.")
    ap.add_argument(
        "--nhead", type=int, default=_MC.nhead,
        help="Numero de cabecas do Multi-Head Self-Attention em cada camada do encoder. "
             "Precisa dividir --d-model. Padrao: 8 (conforme o plano).")
    ap.add_argument(
        "--num-layers", type=int, default=_MC.num_layers,
        help="Quantidade de TransformerEncoderLayer empilhadas no encoder. O plano sugere de "
             "4 a 6; mais camadas = maior capacidade e custo. Padrao: 6.")
    ap.add_argument(
        "--dropout", type=float, default=_MC.dropout,
        help="Taxa de dropout no encoder (projecao, atencao e feed-forward), para regularizar "
             "contra overfitting as transicoes sinteticas. Padrao: 0.2 (conforme o plano).")
    ap.add_argument(
        "--use-ctc", action="store_true", default=_MC.use_ctc,
        help="Ativa a supervisao auxiliar de reconhecimento por CTC: um head sobre o encoder "
             "preve a sequencia de glosas (campo 'tokens' do manifesto), forcando o encoder a "
             "codificar identidade de sinal. Desligado por padrao (arquitetura original).")

    # ------------------------------ treino ----------------------------- #
    ap.add_argument(
        "--epochs", type=int, default=_TC.epochs,
        help="Numero de epocas de treino (varreduras completas do conjunto de treino). Padrao: 30.")
    ap.add_argument(
        "--batch-size", type=int, default=_DC.batch_size,
        help="Numero de frases por passo de otimizacao. Limitado pela memoria da GPU; frases "
             "longas encarecem o passo por causa do padding ao maior T do lote. Padrao: 8.")
    ap.add_argument(
        "--lr", type=float, default=_TC.lr,
        help="Taxa de aprendizado inicial do AdamW, aplicada a todos os parametros (encoder do "
             "zero e decoder pre-treinado). Padrao: 3e-5 (tipico de fine-tuning do T5).")
    ap.add_argument(
        "--warmup-steps", type=int, default=_TC.warmup_steps,
        help="Passos de aquecimento em que a taxa sobe linearmente de 0 ate --lr, antes do "
             "decaimento linear ate 0. Suaviza o inicio, quando o cross-attention ainda recebe "
             "features aleatorias do encoder. Padrao: 500.")
    ap.add_argument(
        "--grad-clip", type=float, default=_TC.grad_clip,
        help="Valor maximo da norma L2 dos gradientes (gradient clipping), para evitar explosao "
             "de gradiente e estabilizar o treino. Padrao: 1.0.")
    ap.add_argument(
        "--patience", type=int, default=_TC.patience,
        help="Early stopping: numero de epocas consecutivas sem melhora na val_loss antes de "
             "parar. 0 (padrao) desativa e treina todas as --epochs. Requer --val-manifest.")
    ap.add_argument(
        "--min-delta", type=float, default=_TC.min_delta,
        help="Melhora minima na val_loss para reiniciar a paciencia do early stopping. "
             "Padrao: 0.0 (qualquer melhora conta).")
    ap.add_argument(
        "--ctc-weight", type=float, default=_TC.ctc_weight,
        help="Peso da perda auxiliar de CTC na perda total (ce + peso*ctc). So tem efeito com "
             "--use-ctc. Padrao: 0.3.")
    ap.add_argument(
        "--max-text-len", type=int, default=_DC.max_text_len,
        help="Comprimento maximo (em tokens) do texto-alvo na tokenizacao; textos maiores sao "
             "truncados. Controla o tamanho dos labels e o custo do decoder. Padrao: 64.")
    ap.add_argument(
        "--num-workers", type=int, default=_DC.num_workers,
        help="Numero de processos do DataLoader para carregar/coletar os .npy em paralelo. "
             "Aumente se o I/O de disco for gargalo; 0 desativa o multiprocessing. Padrao: 4.")
    ap.add_argument(
        "--log-every", type=int, default=_TC.log_every,
        help="Intervalo, em passos, para imprimir a perda media e a taxa de aprendizado atual "
             "durante a epoca. Padrao: 50.")
    ap.add_argument(
        "--out-dir", default=_TC.out_dir,
        help="Diretorio onde salvar os checkpoints por epoca (epochN.pt), o melhor (best.pt) e "
             "o tokenizer. Criado se nao existir. Padrao: 'checkpoints'.")
    ap.add_argument(
        "--device", default=_TC.device,
        help="Dispositivo de execucao: 'cuda' ou 'cpu'. Se omitido, usa CUDA quando disponivel "
             "e cai para CPU caso contrario.")
    ap.add_argument(
        "--low-vram", action="store_true", default=_TC.low_vram,
        help="Ativa otimizacoes de memoria de GPU (desligadas por padrao): (1) move para a CPU "
             "os blocos do encoder do T5, que nunca executam nesta arquitetura; (2) autocast "
             "bfloat16 no forward (exige GPU Ampere+; senao mantem fp32); (3) otimizador AdamW "
             "8-bit do bitsandbytes, se instalado. Nao muda a arquitetura nem o formato dos "
             "checkpoints. Detalhes em LOW_VRAM.md.")
    ap.add_argument(
        "--grad-checkpoint", action="store_true", default=_TC.grad_checkpoint,
        help="Ativa gradient checkpointing no LandmarkEncoder e no decoder T5: as ativacoes "
             "deixam de ser guardadas e sao recomputadas no backward. Corta drasticamente a "
             "VRAM de ativacoes (o gargalo com sequencias longas de frames), ao custo de "
             "~25-30%% de velocidade. Combine com --low-vram. Detalhes em LOW_VRAM.md.")
    return ap


def args_to_configs(args):
    model_cfg = ModelConfig(t5_name=args.t5, d_model=args.d_model, nhead=args.nhead,
                            num_layers=args.num_layers, dropout=args.dropout,
                            use_ctc=args.use_ctc)
    data_cfg = DataConfig(train_manifest=args.train_manifest, val_manifest=args.val_manifest,
                          features_dir=args.features_dir, batch_size=args.batch_size,
                          num_workers=args.num_workers, max_text_len=args.max_text_len)
    train_cfg = TrainConfig(epochs=args.epochs, lr=args.lr, warmup_steps=args.warmup_steps,
                            grad_clip=args.grad_clip, log_every=args.log_every,
                            out_dir=args.out_dir, device=args.device,
                            patience=args.patience, min_delta=args.min_delta,
                            ctc_weight=args.ctc_weight, low_vram=args.low_vram,
                            grad_checkpoint=args.grad_checkpoint)
    return model_cfg, data_cfg, train_cfg


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.smoke_test:
        smoke_test(device=args.device, use_ctc=args.use_ctc)
        return
    if not args.train_manifest:
        sys.exit("Faltou --train-manifest (ou use --smoke-test). Veja o cabecalho do arquivo.")
    build_and_train(*args_to_configs(args))


if __name__ == "__main__":
    main()
