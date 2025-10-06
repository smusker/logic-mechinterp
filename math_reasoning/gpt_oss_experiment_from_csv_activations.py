# =========================
# GENERATION (PATCHED, single-position loop)
# =========================
def _get_return_token_id(tokenizer: AutoTokenizer, default_id: int = 200002) -> int:
    # Try to find <|return|> by token; fall back to known id
    try:
        rtok_id = tokenizer.convert_tokens_to_ids("<|return|>")
        if isinstance(rtok_id, int) and rtok_id > 0:
            return rtok_id
    except Exception:
        pass
    return default_id

def generate_with_activation_patch_single_pos(
    question: str,
    base_complete_analysis: str,
    source_complete_analysis: str,
    up_to_edit_text: str,
    layer: int = PATCH_LAYER,
    window: int = PATCH_WINDOW
) -> str:
    """
    Decode greedily while applying a single-position vanilla activation patch for the
    first K steps (one token pos per step) near the edit site. Then continue without patching.
    """
    init_model_and_tokenizer()
    intervenable = init_intervenable(layer=layer)

    # Render prompts
    base_prompt   = render_harmony_prompt_with_complete_analysis(question, base_complete_analysis)
    source_prompt = render_harmony_prompt_with_complete_analysis(question, source_complete_analysis)

    # Tokenize input contexts
    base_tokens   = _TOKENIZER(base_prompt, return_tensors="pt")["input_ids"].to(_MODEL.device)
    source_tokens = _TOKENIZER(source_prompt, return_tensors="pt")["input_ids"].to(_MODEL.device)

    # Compute edit center and small range
    center_idx = compute_edit_index(_TOKENIZER, question, up_to_edit_text, source_complete_analysis)
    patch_positions = build_patch_positions(center_idx, window, seq_len=base_tokens.shape[1])

    output_str = ""
    pred_str = ""
    end_token_id = _get_return_token_id(_TOKENIZER, default_id=200002)

    steps = 0
    K = len(patch_positions)

    while steps < MAX_NEW_TOKENS and pred_str != "<|return|>":
        if steps < K:
            pos = patch_positions[steps]
            # Some Pyvene builds require a LIST of positions:
            unit_loc = {"sources->base": [pos]}
            with torch.no_grad():
                _, cf_outputs = intervenable(
                    base={"input_ids": base_tokens},
                    sources=[{"input_ids": source_tokens}],
                    unit_locations=unit_loc,
                )
            next_id = cf_outputs.logits[0, -1].argmax(dim=-1)
        else:
            with torch.no_grad():
                logits = _MODEL(input_ids=base_tokens).logits
            next_id = logits[0, -1].argmax(dim=-1)

        if int(next_id.item()) == int(end_token_id):
            break

        pred_str = _TOKENIZER.decode(next_id)
        output_str += pred_str

        # Append to BOTH streams to keep them aligned
        next_id_2d = next_id.unsqueeze(0).unsqueeze(0)
        base_tokens = torch.cat([base_tokens, next_id_2d], dim=1)
        source_tokens = torch.cat([source_tokens, next_id_2d], dim=1)

        steps += 1

    full_text = _TOKENIZER.decode(base_tokens[0], skip_special_tokens=False) + output_str
    return parse_final_from_text(full_text)
