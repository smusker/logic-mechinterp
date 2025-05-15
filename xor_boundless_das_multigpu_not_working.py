# ================================================================
#  Boundless DAS on Meta-Llama-3-8B-Instruct  (multi-GPU friendly)
# ================================================================

import random, numpy as np, torch
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from pyvene import (
    IntervenableModel, RepresentationConfig, IntervenableConfig,
    BoundlessRotatedSpaceIntervention,
)

# ────────────────────────────────────────────────────────────────
#  hot-patch gather_neurons so its index tensors use the right GPU
# ────────────────────────────────────────────────────────────────
import pyvene.models.modeling_utils as pmu
import pyvene.models.intervenable_base as pib

if not hasattr(pmu, "_old_gather_neurons"):
    pmu._old_gather_neurons = pmu.gather_neurons

    def _to_tensor_if_numeric(x, dev):
        if x is None or isinstance(x, (str, slice)):
            return x
        if torch.is_tensor(x):
            return x.to(dev)
        if isinstance(x, (list, tuple)) and all(isinstance(i, int) for i in x):
            return torch.as_tensor(x, device=dev, dtype=torch.long)
        return x

    def _device_aware_gather(t_out, tokens=None, neurons=None, **kw):
        dev = t_out.device
        tokens  = _to_tensor_if_numeric(tokens,  dev)
        neurons = _to_tensor_if_numeric(neurons, dev)
        kw.pop("device", None)
        return pmu._old_gather_neurons(t_out, tokens, neurons, device=dev, **kw)

    pmu.gather_neurons = _device_aware_gather
    pib.gather_neurons = _device_aware_gather

# ────────────────────────────────────────────────────────────────
#  helpers
# ────────────────────────────────────────────────────────────────
def get_root_device(model):
    return next(model.parameters()).device


# ────────────────────────────────────────────────────────────────
#  load Llama-3-8B (sharded)
# ────────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
).eval()

print("root device =", get_root_device(model))

# ---------------------------------------------------------------
# crucial fix: place the *embedding matrix* on the root GPU so
# the index tensors that Accelerate copies there during forward
# and backward match the weight shard’s device.
# ---------------------------------------------------------------
model.model.embed_tokens = model.model.embed_tokens.to(get_root_device(model))

# ────────────────────────────────────────────────────────────────
#  data
# ────────────────────────────────────────────────────────────────
from xor_utils import build_loader, cf_pair_model_A, cf_pair_model_B

train_A = build_loader(tokenizer, cf_pair_model_A, 8000, batch_size=16)
test_A  = build_loader(tokenizer, cf_pair_model_A, 1000, batch_size=16, shuffle=False)

train_B = build_loader(tokenizer, cf_pair_model_B, 8000, batch_size=16)
test_B  = build_loader(tokenizer, cf_pair_model_B, 1000, batch_size=16, shuffle=False)

# ────────────────────────────────────────────────────────────────
#  Intervenable model
# ────────────────────────────────────────────────────────────────
LAYER_TO_PROBE = 12
TOKEN_TO_PROBE = 66      # prompt length 68 → penultimate token = 66

def make_intervenable():
    cfg = IntervenableConfig(
        model_type=type(model),
        representations=[RepresentationConfig(LAYER_TO_PROBE, "block_output")],
        intervention_types=BoundlessRotatedSpaceIntervention,
    )
    iv = IntervenableModel(cfg, model)
    iv.disable_model_gradients()

    # move the tiny intervention (~16 MB) onto the same GPU as layer-12
    layer_dev = next(model.model.layers[LAYER_TO_PROBE].parameters()).device
    for intv in iv.interventions.values():
        intv.to(layer_dev)
    return iv

# ────────────────────────────────────────────────────────────────
#  training loop
# ────────────────────────────────────────────────────────────────
def train_das(intervenable, loader, epochs=2):
    intv = list(intervenable.interventions.values())[0]
    optim = torch.optim.Adam(
        [{"params": intv.rotate_layer.parameters()},
         {"params": intv.intervention_boundaries, "lr": 1e-2}],
        lr=1e-3,
    )
    total_steps = epochs * len(loader)
    temp_sched  = torch.linspace(50., 0.1, total_steps, device=get_root_device(model))
    step = 0

    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"epoch {ep}")
        for batch in pbar:
            # NOTE: input_ids stay on CPU (embedding wrapper handles them)
            optim.zero_grad()

            _, cf = intervenable(
                {"input_ids": batch["input_ids"]},
                [{"input_ids": batch["source_input_ids"]}],
                {"sources->base": TOKEN_TO_PROBE},
            )

            logits_dev = cf.logits.device
            labels     = batch["labels"].to(logits_dev)

            ce = CrossEntropyLoss()(
                    cf.logits.view(-1, cf.logits.size(-1)),
                    labels.view(-1))
            l1 = intv.intervention_boundaries.sum().to(logits_dev)
            loss = ce + l1

            loss.backward(); optim.step()
            intv.set_temperature(temp_sched[step]); step += 1
            pbar.set_postfix(loss=float(loss))

# ────────────────────────────────────────────────────────────────
#  evaluation
# ────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_iia(intervenable, loader):
    tot = cor = 0
    for batch in loader:
        _, cf = intervenable(
            {"input_ids": batch["input_ids"]},
            [{"input_ids": batch["source_input_ids"]}],
            {"sources->base": TOKEN_TO_PROBE},
        )
        pred = cf.logits.argmax(-1).cpu()
        labels = batch["labels"]
        mask = labels != -100
        tot += mask.sum().item()
        cor += ((pred == labels) & mask).sum().item()
    return cor / tot

# ────────────────────────────────────────────────────────────────
#  run
# ────────────────────────────────────────────────────────────────
set_seed(42); random.seed(42); np.random.seed(42)

iv_A = make_intervenable()
print("\n⫸  training (Model A)")
train_das(iv_A, train_A, epochs=2)
print("IIA A =", compute_iia(iv_A, test_A))

iv_B = make_intervenable()
print("\n⫸  training (Model B)")
train_das(iv_B, train_B, epochs=2)
print("IIA B =", compute_iia(iv_B, test_B))