"""Design B: Lngram single-table memory (zyaaa-ux/Lngram, MIT, vendored unchanged in lngram/) on a frozen HF
Qwen3. The repo's patch_model_with_lngram replaces the decoder forward with a transformers-4 signature; here
the same injection (hidden + lngram(hidden) on the pre-attention residual) is applied with a forward pre-hook,
so it works with transformers 5. The module has no KV-cache path: generate with use_cache=False."""
import torch
from lngram.Single_table_Lngram import (LngramConfig, RouteNgramMemory, init_linear_weight_,
                                        pack_bits_to_route_codes, resolve_added_module_init_std)
from torch import nn

CFG = LngramConfig()  # repo defaults: layers (1, 11), 4 bits/route, n-grams (2, 3), memory_dim 16, surrogate on


class LngramLayer(nn.Module):
    def __init__(self, model, cfg=CFG):
        super().__init__()
        h, std = model.config.hidden_size, resolve_added_module_init_std(model.config)
        self.q_proj = nn.Linear(h, h, bias=False)
        init_linear_weight_(self.q_proj.weight, std=std * cfg.q_proj_init_std_scale)
        self.memory = RouteNgramMemory(
            hidden_size=h, bits_per_route=cfg.bits_per_route, memory_dim=cfg.memory_dim, ngrams=cfg.ngrams,
            dropout=cfg.dropout, initializer_range=std, table_init_mode=cfg.table_init_mode,
            table_init_std_scale=cfg.table_init_std_scale, output_proj_init_std_scale=cfg.output_proj_init_std_scale,
            rmsnorm_eps=model.config.rms_norm_eps, conv_kernel_size=cfg.conv_kernel_size,
            conv_dilation=cfg.conv_dilation or max(cfg.ngrams), conv_bias=cfg.conv_bias,
            conv_zero_init=cfg.conv_zero_init, proj_chunk_size=cfg.proj_chunk_size,
            q_surrogate_enable=cfg.q_surrogate_enable, q_surrogate_temp=cfg.q_surrogate_temp,
            q_surrogate_scale=cfg.q_surrogate_scale, q_surrogate_route_chunk_size=cfg.q_surrogate_route_chunk_size,
            q_surrogate_accum_fp32=cfg.q_surrogate_accum_fp32)
        self.enabled = True
        self.mode = "on"  # on | zero | shuffle (shuffle permutes route codes, the analogue of shuffled ids)

    def injection(self, norm, hidden):
        q_logits = self.q_proj(norm(hidden))
        codes = pack_bits_to_route_codes((q_logits > 0).to(torch.int32), CFG.bits_per_route)
        if self.mode == "shuffle":
            codes = (codes * 7 + 3) % self.memory.alphabet_size  # fixed bijection on route codes
        out = self.memory.inject_from_route_codes(hidden_states=hidden, route_codes_btr=codes,
                                                  q_logits=q_logits if CFG.q_surrogate_enable else None)
        return torch.zeros_like(out) if self.mode == "zero" else out


def attach(model, cfg=CFG):
    for p in model.parameters():
        p.requires_grad_(False)
    mods = []
    for idx in cfg.target_layers:
        layer = model.model.layers[idx]
        mod = LngramLayer(model, cfg).to(model.device, model.dtype)

        def pre(layer_, args, kwargs, mod=mod):
            if not mod.enabled:
                return None
            hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            hidden = hidden + mod.injection(layer_.input_layernorm, hidden).to(hidden.dtype)
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = hidden
            else:
                args = (hidden,) + tuple(args[1:])
            return args, kwargs

        layer.register_forward_pre_hook(pre, with_kwargs=True)
        layer.lngram = mod  # registers params under the layer for state_dict / optimizer discovery
        mods.append(mod)
    return mods


def table_param_names(model):
    return {n for n, p in model.named_parameters() if p.requires_grad and ".memory.tables." in n}
