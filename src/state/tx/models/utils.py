from typing import Union

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, LlamaConfig, LlamaModel, PreTrainedModel, LlamaAttention

# LoRA / PEFT
try:
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    TaskType = None  # type: ignore


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    activation: nn.Module = nn.ReLU,  # default to nn.ReLU class
) -> nn.Sequential:
    """
    Build an MLP of `n_layers` from `in_dim` to `out_dim`.
    ...
    """
    layers = []
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")

    if n_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
    else:
        # First layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation())  # instantiate the class
        layers.append(nn.Dropout(dropout))

        # Intermediate layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())  # instantiate again
            layers.append(nn.Dropout(dropout))

        # Final layer
        layers.append(nn.Linear(hidden_dim, out_dim))

    return nn.Sequential(*layers)


def get_activation_class(name: str) -> nn.Module:
    """
    Given a string activation name, return the corresponding nn.Module class.

    Supported activation functions (add any more here):
    - ReLU
    - LeakyReLU
    - ELU
    - SELU
    - GELU
    """
    name = name.lower()

    if name == "relu":
        return nn.ReLU
    elif name == "leakyrelu":
        return nn.LeakyReLU
    elif name == "elu":
        return nn.ELU
    elif name == "selu":
        return nn.SELU
    elif name == "gelu":
        return nn.GELU
    # Add more as needed...
    else:
        raise ValueError(f"Unsupported activation function: {name}")


def get_loss_fn(loss: Union[str, nn.Module]) -> nn.Module:
    """
    Given a string loss function name, return the corresponding nn.Module class.

    Supported loss functions (add any more here):
    - MSELoss
    - L1Loss
    - SmoothL1Loss
    """
    if isinstance(loss, nn.Module):
        return loss

    loss = loss.lower()

    if loss == "mse":
        return nn.MSELoss()
    # Add more as needed...
    else:
        raise ValueError(f"Unsupported loss function: {loss}")


def get_transformer_backbone(key, kwargs) -> tuple[PreTrainedModel, int]:
    kwargs = dict(kwargs or {})

    if key == "GPT2":
        config = GPT2Config(**kwargs)
        model = GPT2BidirectionalModel(config)

        # Zero out position embeddings and freeze them
        model.wpe.weight.requires_grad = False
        model.wte.weight.requires_grad = False
        model.wpe.weight.zero_()
        model.wte.weight.zero_()

        model_dim = config.n_embd
    elif key == "llama":
        bidirectional_attention = bool(kwargs.pop("bidirectional_attention", False))
        use_qk_norm = bool(kwargs.pop("use_qk_norm", False))

        config = LlamaConfig(**kwargs)
        if bidirectional_attention:
            model = LlamaBidirectionalModel(config)
        else:
            model = LlamaModel(config)
        model_dim = config.hidden_size

        model.embed_tokens.weight.requires_grad = False
        model.embed_tokens.weight.zero_()
        
        # Apply QK normalization if requested
        if use_qk_norm:
            _replace_attention_with_qk_norm(model)
    else:
        raise ValueError(f"Unknown backbone key {key}")

    return model, model_dim


# -------------------------------
# LoRA utilities
# -------------------------------
def _default_lora_targets(backbone_key: str, adapt_mlp: bool) -> list[str]:
    """
    Choose target module names for LoRA injection based on backbone type.
    """
    k = backbone_key.lower()
    if k == "llama":
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if adapt_mlp:
            targets += ["gate_proj", "up_proj", "down_proj"]
        return targets
    if k == "gpt2":
        targets = ["c_attn", "c_proj"]
        if adapt_mlp:
            targets += ["mlp.c_fc", "mlp.c_proj"]
        return targets
    raise ValueError(f"Unsupported backbone for LoRA: {backbone_key}")


def apply_lora(model: PreTrainedModel, backbone_key: str, lora_cfg: dict | None) -> PreTrainedModel:
    """
    Apply LoRA adapters to a HuggingFace transformer model when enabled.
    If PEFT is unavailable or config is disabled, returns the original model.
    """
    if not lora_cfg or not lora_cfg.get("enable", False):
        return model

    if LoraConfig is None or get_peft_model is None:
        raise ImportError(
            "peft is not installed but `lora.enable` is True. Add `peft` to dependencies."
        )

    target = lora_cfg.get("target", "auto")
    adapt_mlp = bool(lora_cfg.get("adapt_mlp", False))
    target_modules = (
        lora_cfg.get("target_modules")
        if target != "auto"
        else _default_lora_targets(backbone_key, adapt_mlp)
    )

    # Build PEFT LoRA config
    task_type_key = lora_cfg.get("task_type", "FEATURE_EXTRACTION")
    task_type = TaskType[task_type_key] if isinstance(task_type_key, str) else task_type_key

    config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias=lora_cfg.get("bias", "none"),
        target_modules=target_modules,
        task_type=task_type,
    )

    peft_model = get_peft_model(model, config)

    # Optional: print trainable params summary if available
    try:
        peft_model.print_trainable_parameters()
    except Exception:
        pass

    return peft_model


class NoRoPE(nn.Module):
    """
    A drop-in replacement for LlamaRotaryEmbedding that always returns:
      cos = all ones, sin = all zeros
    of shape (batch_size, seq_len, head_dim), so rotary has no effect.
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor):
        # hidden_states: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _hidden_dim = hidden_states.shape

        # Create cos = ones, sin = zeros
        #   shape --> (batch_size, seq_len, head_dim)
        cos = hidden_states.new_ones(batch_size, seq_len, self.head_dim)
        sin = hidden_states.new_zeros(batch_size, seq_len, self.head_dim)
        return cos, sin


class LlamaAttentionWithQKNorm(LlamaAttention):
    """
    LlamaAttention with QK normalization.
    Normalizes query and key vectors before computing attention scores.
    This helps stabilize training and improve model performance.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: torch.Tensor | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, ...] | None]:
        """
        Forward pass with QK normalization.
        Normalizes Q and K vectors (L2 normalization) before computing attention scores.
        """
        bsz, q_len, _ = hidden_states.size()

        # Get Q, K, V projections (same as parent class)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings if provided
        cos, sin = None, None
        if position_embeddings is not None:
            cos, sin = position_embeddings
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Apply QK normalization: normalize Q and K vectors (L2 norm)
        # Shape: [bsz, num_heads, q_len, head_dim]
        query_norm = torch.norm(query_states, p=2, dim=-1, keepdim=True)
        key_norm = torch.norm(key_states, p=2, dim=-1, keepdim=True)
        
        # Avoid division by zero
        eps = 1e-8
        query_states = query_states / (query_norm + eps)
        key_states = key_states / (key_norm + eps)

        # Handle past_key_value for caching (same as parent)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat key/value heads if using GQA
        # Use repeat_kv function from transformers if available, otherwise use _repeat_kv method
        from transformers.models.llama.modeling_llama import repeat_kv
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores: (Q_normalized @ K_normalized^T) / sqrt(head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim**0.5)

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, key_states.size(2)):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, key_states.size(2))}, "
                    f"but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # Convert to float32 for numerical stability
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def _replace_attention_with_qk_norm(model: LlamaModel) -> None:
    """
    Replace all LlamaAttention layers in a LlamaModel (or LlamaBidirectionalModel) 
    with LlamaAttentionWithQKNorm.
    This modifies the model in-place.
    
    Args:
        model: The LlamaModel or LlamaBidirectionalModel instance to modify
    """
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    
    for layer_idx, layer in enumerate(model.layers):
        if isinstance(layer, LlamaDecoderLayer) and hasattr(layer, "self_attn"):
            # Create new attention layer with QK norm, copying config from existing
            old_attn = layer.self_attn
            # Get layer_idx from old attention if available, otherwise use loop index
            attn_layer_idx = getattr(old_attn, "layer_idx", layer_idx)
            new_attn = LlamaAttentionWithQKNorm(
                config=model.config,
                layer_idx=attn_layer_idx,
            )
            # Copy weights from old attention to new
            new_attn.load_state_dict(old_attn.state_dict(), strict=False)
            layer.self_attn = new_attn


class LlamaBidirectionalModel(LlamaModel):
    """
    A drop-in replacement for LlamaModel with bidirectional attention.
    By overriding _update_causal_mask to return None, all tokens attend to each other.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.rotary_emb = NoRoPE(
            head_dim=config.head_dim,
        )
        
        # Explicitly disable causal attention
        self.config.is_causal = False
        # force every layer to be non-causal
        for layer in self.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.is_causal = False   # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values,
        output_attentions: bool = False,
    ):
        # By returning None, we disable any causal‐(look‐ahead) masking.
        # The only mask that remains is whatever "attention_mask" the user has passed
        # (e.g. padding‐mask), which will be handled by Flash/SDPA internally as non‐causal.
        return None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        cache_position: torch.LongTensor = None,
        **flash_attn_kwargs,
    ):
        flash_attn_kwargs["is_causal"] = False
        
        # If no attention_mask is provided, create an all-ones mask (no masking)
        # This ensures bidirectional attention with correct device/dtype
        if attention_mask is None:
            # Get batch size (B) and sequence length (S) from input_embeds if available, else from input_ids.
            # If neither is available, fall back to attention_mask=None and log a warning.
            B = None
            S = None
            if inputs_embeds is not None:
                B, S = inputs_embeds.size(0), inputs_embeds.size(1)
            if B and S:
                attention_mask = torch.ones((B, 1, S, S), dtype=torch.float, device=inputs_embeds.device)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )


class GPT2BidirectionalModel(GPT2Model):
    """
    A thin wrapper around GPT2Model that disables the causal (unidirectional) mask,
    allowing full bidirectional attention—and prints the internal bias mask each forward pass.
    """

    def __init__(self, config: GPT2Config):
        # Mark as not‐a‐decoder (for downstream utilities).
        config.is_decoder = False
        super().__init__(config)

        # Overwrite each attention's bias so no triangular masking occurs.
        for block in self.h:
            # block.attn.bias is a bool‐tensor of shape (1, 1, max_pos, max_pos).
            block.attn.bias.data.fill_(True)
            block.attn.is_causal = False

        def _no_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values,
            output_attentions: bool,
        ):
            return None

        self._update_causal_mask = _no_causal_mask.__get__(self, GPT2Model)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Determine sequence length for printing the relevant slice of bias
        if input_ids is not None:
            seq_len = input_ids.size(1)
        elif inputs_embeds is not None:
            seq_len = inputs_embeds.size(1)
        else:
            seq_len = None  # If neither is given, we can’t infer seq_len

        if seq_len is not None:
            # Print the (1, 1, seq_len, seq_len) slice of the bias for the first block
            bias_mask = self.h[0].attn.bias[0, 0, :seq_len, :seq_len]
        #     print("Bias mask (block 0) slice [0,0,:seq_len,:seq_len]:")
        #     print(bias_mask)
        # else:
        #     print("Cannot infer sequence length to print bias mask.")

        # If a 2D attention_mask was provided, print its expanded 4D version:
        if attention_mask is not None:
            # Expand to (batch_size, 1, seq_len, seq_len)
            B, S = attention_mask.size()
            expanded = attention_mask.unsqueeze(1).unsqueeze(2).expand(B, 1, S, S)
            # Convert to float mask (1→0.0, 0→-inf) just like GPT2 does internally
            neg_inf = torch.finfo(self.dtype).min
            float_mask = (1.0 - expanded.to(self.dtype)) * neg_inf
            # print(f"Expanded attention_mask (shape {expanded.shape}) → float mask:")
            # print(float_mask)

        # Finally, call the parent forward method
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
