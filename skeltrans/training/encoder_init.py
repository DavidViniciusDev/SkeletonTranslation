"""Arquivo do encoder pré-treinado (``encoder.pt``) e o ``--init-encoder``.

O pré-treino (``pretrain_encoder.py``) grava só os pesos do
``LandmarkEncoder``, mais a configuração necessária para conferir que eles
cabem no encoder do treino de SLT. O treino de SLT os carrega com
``--init-encoder`` como ponto de partida; o resultado final fica inteiro no
``best.pt``, então avaliação e inferência não usam este arquivo.

Formato (um dict salvo com ``torch.save``)::

    {"format": "skeltrans-encoder-pretrain", "version": 1,
     "spec":    {input_dim, extra_features, input_norm, encoder_type,
                 d_model, nhead, num_layers, ds_steps},
     "encoder": state_dict do LandmarkEncoder (sem o adaptador),
     "info":    {como foi o pré-treino: épocas, máscara, perda, manifestos...}}

Só tensores e tipos simples: carrega com ``weights_only=True``, que não
executa código vindo do arquivo.
"""

import torch

from skeltrans.training.data.features import feature_dim, parse_groups

ENCODER_FORMAT = "skeltrans-encoder-pretrain"
ENCODER_VERSION = 1

# Campos que precisam ser iguais no pré-treino e no treino de SLT: mudam o
# formato dos pesos ou o significado da entrada. O dropout pode mudar.
SPEC_FIELDS = ("input_dim", "extra_features", "input_norm", "encoder_type",
               "d_model", "nhead", "num_layers", "ds_steps")

# Chaves do LandmarkEncoder que NÃO vêm do pré-treino: o adaptador (ponte para
# o hidden do T5) e o andar de conceitos (depende do CTC). Começam do zero.
_FRESH_PREFIXES = ("adapter.", "concept_pool.", "concept_norm.", "concept_transformer.")

# Flags da CLI correspondentes a cada campo, para as mensagens de erro.
_FLAG = {"input_dim": "--extra-features", "extra_features": "--extra-features",
         "input_norm": "--input-norm", "encoder_type": "--encoder-type",
         "d_model": "--d-model", "nhead": "--nhead", "num_layers": "--num-layers",
         "ds_steps": "--ds-steps"}


def encoder_spec(extra_features, input_norm, encoder_type, d_model, nhead,
                 num_layers, ds_steps):
    """Spec de um encoder a partir dos hiperparâmetros (forma canônica)."""
    groups = parse_groups(extra_features)
    return {"input_dim": feature_dim(groups), "extra_features": ",".join(groups),
            "input_norm": bool(input_norm), "encoder_type": str(encoder_type),
            "d_model": int(d_model), "nhead": int(nhead),
            "num_layers": int(num_layers), "ds_steps": int(ds_steps)}


def spec_from_model_config(model_cfg):
    """Spec do encoder que o treino de SLT vai montar (``ModelConfig``)."""
    return encoder_spec(model_cfg.extra_features, model_cfg.input_norm,
                        model_cfg.encoder_type, model_cfg.d_model, model_cfg.nhead,
                        model_cfg.num_layers, model_cfg.ds_steps)


def save_encoder(path, encoder, spec, info=None):
    """Grava os pesos do encoder (sem adaptador/conceitos) e a spec."""
    state = {k: v.detach().cpu() for k, v in encoder.state_dict().items()
             if not k.startswith(_FRESH_PREFIXES)}
    torch.save({"format": ENCODER_FORMAT, "version": ENCODER_VERSION,
                "spec": dict(spec), "encoder": state, "info": dict(info or {})}, path)


def read_encoder(path):
    """Lê um ``encoder.pt`` e confere o formato. Levanta ValueError se não for um."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("format") != ENCODER_FORMAT:
        raise ValueError(f"{path} nao e um encoder pre-treinado do SkeLTrans (gere-o com "
                         "pretrain_encoder.py; um best.pt de SLT nao serve aqui).")
    if ckpt.get("version") != ENCODER_VERSION:
        raise ValueError(f"{path}: versao {ckpt.get('version')} do formato nao suportada "
                         f"(esperada {ENCODER_VERSION}).")
    return ckpt


def spec_mismatches(saved, wanted):
    """Lista de diferenças (texto) entre a spec do arquivo e a do treino."""
    out = []
    for field in SPEC_FIELDS:
        a, b = saved.get(field), wanted.get(field)
        if a != b:
            out.append(f"{field}: pre-treino={a!r}, treino={b!r} ({_FLAG[field]})")
    return out


def load_encoder_weights(encoder, ckpt):
    """Copia os pesos pré-treinados para ``encoder`` (um ``LandmarkEncoder``).

    Exige que toda chave do arquivo exista no encoder e que só faltem as que
    começam do zero (adaptador e andar de conceitos). Qualquer outra
    diferença é erro: com ``strict=False`` puro, um peso de nome diferente
    ficaria aleatório sem aviso. Devolve as chaves que ficaram do zero.
    """
    missing, unexpected = encoder.load_state_dict(ckpt["encoder"], strict=False)
    bad_missing = [k for k in missing if not k.startswith(_FRESH_PREFIXES)]
    if unexpected or bad_missing:
        raise ValueError("encoder pre-treinado incompativel: "
                         f"chaves sobrando {unexpected[:4]}, faltando {bad_missing[:4]}.")
    return list(missing)


def init_encoder_from_file(slt_model, path, model_cfg):
    """``--init-encoder``: carrega ``path`` no encoder de um ``SLTModel``.

    Confere a spec contra ``model_cfg`` antes de tocar nos pesos. Com
    ``--input-norm``, a média e o desvio do pré-treino vêm junto (buffers) e
    devem ser mantidos: os pesos aprenderam com aquela escala. Devolve o
    ``info`` do arquivo (para o log).
    """
    ckpt = read_encoder(path)
    errors = spec_mismatches(ckpt["spec"], spec_from_model_config(model_cfg))
    if errors:
        raise ValueError(f"--init-encoder {path} nao combina com este treino:\n  "
                         + "\n  ".join(errors))
    load_encoder_weights(slt_model.encoder, ckpt)
    return ckpt.get("info", {})
