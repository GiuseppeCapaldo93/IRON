#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Whisper-small transcription of one 30 s window on Phoenix/NPU1.

Phases, each kept within the NPU1 context cache and cleaned up in between:

1. frontend: conv1 / GELU / conv2 / GELU on the NPU at 3000 Mel frames.
2. encoder: 12 blocks at 1536 physical rows (1500 logical) with four-column
   GEMMs; LayerNorm, GELU and masked softmax on the CPU. The resident 768x768
   GEMM then computes the cross-attention K/V of every decoder block.
3. prefill: the four-token Whisper prompt.
4. decode: one token per step with four-column NPU GEMVs and resident BF16
   weights; LayerNorm, attention and GELU on the CPU.

The vocabulary projection and the greedy token selection run on the CPU.

Acceptance: the generated token ids must exactly match a CPU FP32 Hugging Face
greedy decode of the same log-Mel features under the same suppression policy.
The process exits non-zero on a mismatch or when no end-of-text token is
produced.
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import aie.utils as aie_utils
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from iron.operators.gelu.op import GELU
from iron.operators.gemm.op import GEMM
from iron.operators.gemv.op import GEMV

from whisper_audio import load_wav, log_mel_spectrogram
from whisper_common import (
    ARTIFACT_ROOT,
    BLOCKS,
    HEAD_DIM,
    HEADS,
    MELS,
    MLP,
    SCALE,
    STATE,
    whisper_checkpoint,
)
from whisper_decoder import (
    DECODER_PHYSICAL_SEQ,
    DecoderPrefillRuntime,
    LayerNorm,
    load_decoder_block_weights,
    load_decoder_embedding,
    load_decoder_final_layernorm,
    load_decoder_position_embedding,
    make_context,
    merge_heads,
    run_decoder_embedding,
    run_gemm,
    run_projection,
    split_heads,
)
from whisper_encoder import load_block_weights

FRAMES = 3000
FRAMES_PAD = 3072
SEQ = 1500
SEQ_PAD = 1536
K1_REAL = 3 * MELS
K1_PAD = 256
K2 = 3 * STATE

# Phoenix exposes four usable AIE columns.
COLUMNS = 4

# <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
PROMPT = [50258, 50259, 50359, 50363]
EOS = 50257
FIRST_NON_TEXT_SPECIAL = 50358
MAX_NEW_TOKENS = 224

runtime_backend = aie_utils.DefaultNPURuntime


def default_build_dir() -> Path:
    value = os.environ.get("IRON_WHISPER_BUILD_DIR")
    return Path(value) if value else ARTIFACT_ROOT / "pipeline-build"


def normalize_words(text):
    text = text.upper()
    for short, long in (("MR.", "MISTER"), ("MRS.", "MISSUS"), ("DR.", "DOCTOR")):
        text = text.replace(short, long)
    return re.sub(r"[^A-Z0-9' ]", " ", text).replace("'", "").split()


def word_error_rate(reference, hypothesis):
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1] / max(len(ref), 1)


def bf16(x):
    return x.to(torch.bfloat16).float()


def nrmse(actual, expected):
    actual = actual.float()
    expected = expected.float()
    return (
        torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
    ).item()


def cleanup(label):
    before = len(runtime_backend._context_cache)
    gc.collect()
    runtime_backend.cleanup()
    after = len(runtime_backend._context_cache)
    print(f"[cleanup] {label}: contexts {before} -> {after}")
    if runtime_backend._context_cache or runtime_backend._insts_cache:
        raise RuntimeError(f"{label}: NPU state survived cleanup")


def check_contexts(label, limit=5):
    count = len(runtime_backend._context_cache)
    print(f"[contexts] {label}: {count}")
    if count > limit:
        raise RuntimeError(f"{label}: {count} contexts exceed budget {limit}")


def gemm(M, K, N, build_root, name, tile_m=32, tile_k=64, tile_n=64, cols=COLUMNS):
    op = GEMM(
        M=M,
        K=K,
        N=N,
        num_aie_columns=cols,
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        prio_accuracy=True,
        emulate_bf16_mmul_with_bfp16=False,
        context=make_context(build_root, name),
    )
    op.compile()
    return op, op.get_callable()


def gelu_npu(x, build_root, name):
    tensor_class = aie_utils.DEFAULT_TENSOR_CLASS
    size = x.numel()
    op = GELU(
        size=size,
        num_aie_columns=1,
        num_channels=1,
        tile_size=8192,
        context=make_context(build_root, name),
    )
    op.compile()
    fn = op.get_callable()
    inp = tensor_class.from_torch(x.to(torch.bfloat16).flatten().contiguous())
    out = tensor_class((size,), dtype=np.dtype("bfloat16"))
    fn(inp, out)
    result = out.to_torch().float().clone().reshape(x.shape)
    del out, inp, fn, op
    return result


def pad_to(x, rows):
    out = torch.zeros((rows, x.shape[1]), dtype=x.dtype)
    out[: x.shape[0]] = x
    return out


# ============================================================================
# Phase 1: frontend
# ============================================================================


def run_frontend_full(mel, checkpoint, build_root):
    with safe_open(str(checkpoint), framework="pt", device="cpu") as h:
        w1 = h.get_tensor("model.encoder.conv1.weight").float()
        b1 = h.get_tensor("model.encoder.conv1.bias").float()
        w2 = h.get_tensor("model.encoder.conv2.weight").float()
        b2 = h.get_tensor("model.encoder.conv2.bias").float()
        positions = h.get_tensor("model.encoder.embed_positions.weight").float()

    # conv1 (k=3, s=1, p=1) as an im2col GEMM: [FRAMES, 3*MELS] x [3*MELS, STATE]
    xt = F.pad(bf16(mel[0]).T, (0, 0, 1, 1))
    a1 = torch.cat([xt[0:FRAMES], xt[1 : FRAMES + 1], xt[2 : FRAMES + 2]], dim=1)
    a1_pad = torch.zeros((FRAMES_PAD, K1_PAD))
    a1_pad[:FRAMES, :K1_REAL] = a1
    b1_gemm = torch.zeros((K1_PAD, STATE))
    b1_gemm[:K1_REAL] = w1.permute(2, 1, 0).reshape(K1_REAL, STATE)

    op, fn = gemm(FRAMES_PAD, K1_PAD, STATE, build_root, "conv1", tile_k=32)
    conv1 = run_gemm(fn, a1_pad, b1_gemm, (FRAMES_PAD, STATE)) + b1
    del fn, op

    gelu1 = gelu_npu(conv1, build_root, "gelu1")[:FRAMES]

    # conv2 (k=3, s=2, p=1): output o uses padded rows 2o, 2o+1, 2o+2
    xp = F.pad(bf16(gelu1), (0, 0, 1, 1))
    a2 = torch.cat(
        [xp[0 : 2 * SEQ : 2], xp[1 : 2 * SEQ + 1 : 2], xp[2 : 2 * SEQ + 2 : 2]],
        dim=1,
    )
    b2_gemm = w2.permute(2, 1, 0).reshape(K2, STATE)

    op, fn = gemm(SEQ_PAD, K2, STATE, build_root, "conv2")
    conv2 = run_gemm(fn, pad_to(a2, SEQ_PAD), b2_gemm, (SEQ_PAD, STATE)) + b2
    del fn, op

    gelu2 = gelu_npu(conv2, build_root, "gelu2")

    check_contexts("frontend", limit=6)

    x = torch.zeros((SEQ_PAD, STATE))
    x[:SEQ] = gelu2[:SEQ] + positions[:SEQ]
    return x


# ============================================================================
# Phase 2: encoder and cross-attention K/V
# ============================================================================


class FullWindowEncoderRuntime:
    def __init__(self, build_root):
        def make(K, N, name, cols=COLUMNS):
            return gemm(SEQ_PAD, K, N, build_root, name, tile_m=16, cols=cols)

        self.p768_op, self.p768 = make(STATE, STATE, "gemm-768-768")
        self.score_op, self.score = make(HEAD_DIM, SEQ_PAD, "score")
        # N=64 is not a multiple of COLUMNS * tile_n, so this one uses one column.
        self.value_op, self.value = make(SEQ_PAD, HEAD_DIM, "value", cols=1)
        self.fc1_op, self.fc1 = make(STATE, MLP, "gemm-768-3072")
        self.fc2_op, self.fc2 = make(MLP, STATE, "gemm-3072-768")
        self.key_mask = torch.zeros(SEQ_PAD, dtype=torch.bool)
        self.key_mask[SEQ:] = True

    @staticmethod
    def layer_norm(x, weight, bias):
        return F.layer_norm(bf16(x), (STATE,), weight.float(), bias.float())

    def projection(self, x, weight, bias):
        out = run_gemm(self.p768, x, weight.float().T.contiguous(), (SEQ_PAD, STATE))
        if bias is not None:
            out += bias.float()
        return out

    def block(self, x, w):
        h = self.layer_norm(x, w["attn_ln.weight"], w["attn_ln.bias"])
        q = self.projection(h, w["attn.query.weight"], w["attn.query.bias"])
        k = self.projection(h, w["attn.key.weight"], None)
        v = self.projection(h, w["attn.value.weight"], w["attn.value.bias"])
        qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)

        heads = []
        for head in range(HEADS):
            scores = run_gemm(
                self.score,
                qh[head] * SCALE,
                kh[head].T.contiguous(),
                (SEQ_PAD, SEQ_PAD),
            )
            scores.masked_fill_(self.key_mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            heads.append(run_gemm(self.value, probs, vh[head], (SEQ_PAD, HEAD_DIM)))

        x = x + self.projection(
            merge_heads(torch.stack(heads)), w["attn.out.weight"], w["attn.out.bias"]
        )

        h = self.layer_norm(x, w["mlp_ln.weight"], w["mlp_ln.bias"])
        up = run_gemm(
            self.fc1, h, w["mlp.0.weight"].float().T.contiguous(), (SEQ_PAD, MLP)
        )
        up += w["mlp.0.bias"].float()
        act = F.gelu(bf16(up))
        down = run_gemm(
            self.fc2, act, w["mlp.2.weight"].float().T.contiguous(), (SEQ_PAD, STATE)
        )
        down += w["mlp.2.bias"].float()
        x = x + down
        x[SEQ:] = 0.0
        return x

    def cross_kv(self, encoder_pad, decoder_weights):
        caches = []
        for w in decoder_weights:
            k = run_gemm(
                self.p768,
                encoder_pad,
                w["encoder_attn.k_proj.weight"].T.contiguous(),
                (SEQ_PAD, STATE),
            )[:SEQ]
            v = run_gemm(
                self.p768,
                encoder_pad,
                w["encoder_attn.v_proj.weight"].T.contiguous(),
                (SEQ_PAD, STATE),
            )[:SEQ]
            v += w["encoder_attn.v_proj.bias"]
            caches.append((split_heads(k), split_heads(v)))
        return caches


def run_encoder(mel, checkpoint, build_root, timings):
    start = time.perf_counter()
    x = run_frontend_full(mel, checkpoint, build_root / "frontend")
    timings["frontend"] = time.perf_counter() - start
    frontend = x[:SEQ].clone()
    cleanup("frontend -> encoder")

    start = time.perf_counter()
    runtime = FullWindowEncoderRuntime(build_root / "encoder")
    check_contexts("encoder resident")
    timings["encoder_setup"] = time.perf_counter() - start

    start = time.perf_counter()
    for block in range(BLOCKS):
        x = runtime.block(x, load_block_weights(checkpoint, block))
        if not torch.isfinite(x[:SEQ]).all():
            raise RuntimeError(f"Encoder block {block} produced non-finite values")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as h:
        ln_w = h.get_tensor("model.encoder.layer_norm.weight").float()
        ln_b = h.get_tensor("model.encoder.layer_norm.bias").float()
    encoder = F.layer_norm(bf16(x[:SEQ]), (STATE,), ln_w, ln_b)
    timings["encoder"] = time.perf_counter() - start
    return runtime, frontend, encoder


# ============================================================================
# Phase 3: prefill with precomputed cross-attention K/V
# ============================================================================


class FullWindowPrefillRuntime(DecoderPrefillRuntime):
    """Four-token prefill; the score GEMM is widened to N=SEQ_PAD."""

    def __init__(self, build_root, block_weights):
        self.build_root = Path(build_root)
        self.block_weights = block_weights
        self.logical_rows = len(PROMPT)

        self.layernorm = LayerNorm(
            size=len(PROMPT) * STATE,
            num_aie_columns=1,
            num_channels=1,
            tile_size=STATE,
            context=make_context(self.build_root, "layernorm-4"),
        )
        self.layernorm.compile()
        self.layernorm_fn = self.layernorm.get_callable()

        def make(K, N, name):
            P = DECODER_PHYSICAL_SEQ
            return gemm(P, K, N, self.build_root, name, tile_m=16)

        self.projection, self.projection_fn = make(STATE, STATE, "gemm-768-768")
        self.score, self.score_fn = make(HEAD_DIM, SEQ_PAD, "score-64-1536")
        self.fc1, self.fc1_fn = make(STATE, MLP, "gemm-768-3072")
        self.fc2, self.fc2_fn = make(MLP, STATE, "gemm-3072-768")

    def _project(self, x, weights, name, bias=True):
        return run_projection(
            self.projection_fn,
            x,
            weights[f"{name}.weight"],
            weights[f"{name}.bias"] if bias else None,
            self.logical_rows,
        )

    def _scores(self, qh_head, k_cols):
        rows = self.logical_rows
        q_pad = torch.zeros((DECODER_PHYSICAL_SEQ, HEAD_DIM))
        q_pad[:rows] = qh_head * SCALE
        k_pad = torch.zeros((HEAD_DIM, SEQ_PAD))
        k_pad[:, : k_cols.shape[1]] = k_cols
        out = run_gemm(self.score_fn, q_pad, k_pad, (DECODER_PHYSICAL_SEQ, SEQ_PAD))
        return out[:rows, : k_cols.shape[1]]

    def _attend(self, qh, kh, vh, mask=None):
        scores = torch.stack([self._scores(qh[i], kh[i].T) for i in range(HEADS)])
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        return merge_heads(torch.matmul(torch.softmax(scores, dim=-1), vh))

    def self_attention(self, x, weights):
        h = self.layer_norm(
            x,
            weights["self_attn_layer_norm.weight"],
            weights["self_attn_layer_norm.bias"],
        )
        qh = split_heads(self._project(h, weights, "self_attn.q_proj"))
        kh = split_heads(self._project(h, weights, "self_attn.k_proj", bias=False))
        vh = split_heads(self._project(h, weights, "self_attn.v_proj"))
        rows = self.logical_rows
        mask = torch.triu(torch.ones(rows, rows, dtype=torch.bool), diagonal=1)
        out = self._project(
            self._attend(qh, kh, vh, mask), weights, "self_attn.out_proj"
        )
        return x + out, kh, vh

    def cross_attention_cached(self, x, kh, vh, weights):
        h = self.layer_norm(
            x,
            weights["encoder_attn_layer_norm.weight"],
            weights["encoder_attn_layer_norm.bias"],
        )
        qh = split_heads(self._project(h, weights, "encoder_attn.q_proj"))
        out = self._project(self._attend(qh, kh, vh), weights, "encoder_attn.out_proj")
        return x + out

    def run(self, x, cross):
        caches = {}
        for block in range(BLOCKS):
            w = self.block_weights[block]
            x, self_k, self_v = self.self_attention(x, w)
            cross_k, cross_v = cross[block]
            x = self.cross_attention_cached(x, cross_k, cross_v, w)
            x = self.mlp(x, w)
            caches[block] = {
                "self_k": self_k.clone(),
                "self_v": self_v.clone(),
                "cross_k": cross_k,
                "cross_v": cross_v,
            }
        return x, caches


# ============================================================================
# Phase 4: incremental decoder on four-column GEMV
# ============================================================================


class GemvDecoderRuntime:
    """One-token decoder step with NPU GEMV projections and resident BF16 weights.

    LayerNorm, attention and GELU run on the CPU: an NPU call per 768-wide row
    costs far more in dispatch than the work itself. Four contexts are used:
    768->2304 (fused Q|K|V), 768->768, 768->3072 and 3072->768.
    """

    GEOMETRIES = {  # name: (M out, K in, tile_size_input)
        "qkv": (3 * STATE, STATE, 8),
        "p768": (STATE, STATE, 8),
        "fc1": (MLP, STATE, 8),
        "fc2": (STATE, MLP, 2),
    }

    def __init__(self, build_root, block_weights):
        tc = aie_utils.DEFAULT_TENSOR_CLASS
        self.weights = block_weights
        self.ops, self.fns, self.outputs = {}, {}, {}
        for name, (M, K, tile_in) in self.GEOMETRIES.items():
            per_col = M // COLUMNS
            op = GEMV(
                M=M,
                K=K,
                num_aie_columns=COLUMNS,
                tile_size_input=tile_in,
                tile_size_output=per_col // 2 if per_col > 128 else per_col,
                context=make_context(Path(build_root), f"gemv-{name}"),
            )
            op.compile()
            self.ops[name], self.fns[name] = op, op.get_callable()
            self.outputs[name] = tc((M,), dtype=np.dtype("bfloat16"))
        self.inputs = {
            size: tc((size,), dtype=np.dtype("bfloat16")) for size in (STATE, MLP)
        }

        def resident(t):
            return tc.from_torch(t.to(torch.bfloat16).contiguous())

        self.resident = []
        for w in block_weights:
            qkv = torch.cat(
                [
                    w["self_attn.q_proj.weight"],
                    w["self_attn.k_proj.weight"],
                    w["self_attn.v_proj.weight"],
                ]
            )
            self.resident.append(
                {
                    "qkv": resident(qkv),
                    "self_out": resident(w["self_attn.out_proj.weight"]),
                    "cross_q": resident(w["encoder_attn.q_proj.weight"]),
                    "cross_out": resident(w["encoder_attn.out_proj.weight"]),
                    "fc1": resident(w["fc1.weight"]),
                    "fc2": resident(w["fc2.weight"]),
                }
            )

    @staticmethod
    def layer_norm(x, weight, bias):
        # Same numerics as the NPU LayerNorm: BF16 in, BF16 out, FP32 affine.
        return bf16(F.layer_norm(bf16(x), (STATE,))) * weight.float() + bias.float()

    def gemv(self, geometry, block, key, x, bias=None):
        vec = self.inputs[x.shape[-1]]
        bits = x.reshape(-1).to(torch.bfloat16).contiguous().view(torch.uint16)
        with vec.overwrite() as buffer:
            buffer.view(np.uint16)[...] = bits.numpy()
        out = self.outputs[geometry]
        self.fns[geometry](self.resident[block][key], vec, out)
        y = out.to_torch().float().reshape(1, -1).clone()
        return y if bias is None else y + bias.float()

    @staticmethod
    def heads(t):
        return t.view(1, HEADS, HEAD_DIM).transpose(0, 1)

    def attend(self, q, k, v):
        scores = torch.matmul(self.heads(q) * SCALE, k.transpose(1, 2))
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        return torch.matmul(probs, v).transpose(0, 1).reshape(1, STATE)

    def run_block(self, block, x, cache):
        w = self.weights[block]
        h = self.layer_norm(
            x, w["self_attn_layer_norm.weight"], w["self_attn_layer_norm.bias"]
        )
        qkv = self.gemv("qkv", block, "qkv", h)
        q = qkv[:, :STATE] + w["self_attn.q_proj.bias"].float()
        k = qkv[:, STATE : 2 * STATE]
        v = qkv[:, 2 * STATE :] + w["self_attn.v_proj.bias"].float()
        self_k = torch.cat([cache["self_k"], self.heads(k)], dim=1)
        self_v = torch.cat([cache["self_v"], self.heads(v)], dim=1)
        ctx = self.attend(q, self_k, self_v)
        x = x + self.gemv("p768", block, "self_out", ctx, w["self_attn.out_proj.bias"])

        h = self.layer_norm(
            x, w["encoder_attn_layer_norm.weight"], w["encoder_attn_layer_norm.bias"]
        )
        q = self.gemv("p768", block, "cross_q", h, w["encoder_attn.q_proj.bias"])
        ctx = self.attend(q, cache["cross_k"], cache["cross_v"])
        x = x + self.gemv(
            "p768", block, "cross_out", ctx, w["encoder_attn.out_proj.bias"]
        )

        h = self.layer_norm(x, w["final_layer_norm.weight"], w["final_layer_norm.bias"])
        up = F.gelu(self.gemv("fc1", block, "fc1", h, w["fc1.bias"]))
        x = x + self.gemv("fc2", block, "fc2", up, w["fc2.bias"])
        return x, {**cache, "self_k": self_k, "self_v": self_v}

    def step(self, x, caches):
        new = {}
        for block in range(BLOCKS):
            x, new[block] = self.run_block(block, x, caches[block])
        return x, new


# ============================================================================
# Greedy decoding policy and CPU FP32 reference
# ============================================================================


def apply_policy(logits, step, suppress, begin_suppress):
    logits = logits.clone()
    logits[suppress] = float("-inf")
    logits[FIRST_NON_TEXT_SPECIAL:] = float("-inf")
    if step == 0:
        logits[begin_suppress] = float("-inf")
    return logits


def cpu_reference(snapshot, mel, suppress, begin_suppress):
    """Hugging Face FP32 frontend, encoder and greedy tokens for the same mel."""

    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.float32
    ).eval()
    encoder = model.model.encoder
    with torch.no_grad():
        conv = F.gelu(encoder.conv2(F.gelu(encoder.conv1(mel))))[0].T
        frontend = encoder.embed_positions.weight[:SEQ] + conv
        hidden = encoder(mel).last_hidden_state
        ids = list(PROMPT)
        tokens = []
        for step in range(MAX_NEW_TOKENS):
            logits = model(
                encoder_outputs=(hidden,), decoder_input_ids=torch.tensor([ids])
            ).logits[0, -1]
            token = int(apply_policy(logits, step, suppress, begin_suppress).argmax())
            tokens.append(token)
            ids.append(token)
            if token == EOS:
                break
    return frontend, hidden[0], tokens


def transcribe(mel, checkpoint, build_root, timings):
    """Run phases 1-4 and return (frontend, encoder, generated token ids)."""

    config = json.loads((checkpoint.parent / "config.json").read_text("utf-8"))
    suppress = config["suppress_tokens"]
    begin_suppress = config["begin_suppress_tokens"]

    runtime, frontend, encoder = run_encoder(mel, checkpoint, build_root, timings)

    start = time.perf_counter()
    decoder_weights = [load_decoder_block_weights(checkpoint, b) for b in range(BLOCKS)]
    cross = runtime.cross_kv(pad_to(encoder, SEQ_PAD), decoder_weights)
    timings["cross_kv"] = time.perf_counter() - start
    del runtime
    cleanup("encoder -> prefill")

    token_embedding = load_decoder_embedding(checkpoint)
    position_embedding = load_decoder_position_embedding(checkpoint)
    final_w, final_b = (t.float() for t in load_decoder_final_layernorm(checkpoint))

    def embed(ids, offset):
        return run_decoder_embedding(
            token_ids=torch.tensor(ids, dtype=torch.long),
            position_offset=offset,
            checkpoint=checkpoint,
            token_embedding=token_embedding,
            position_embedding=position_embedding,
        )

    def next_token(raw, step):
        final = F.layer_norm(raw[-1:].float(), (STATE,), final_w, final_b)
        logits = torch.matmul(final, token_embedding.T)[0]
        return int(apply_policy(logits, step, suppress, begin_suppress).argmax())

    start = time.perf_counter()
    prefill = FullWindowPrefillRuntime(build_root / "prefill", decoder_weights)
    check_contexts("prefill resident")
    timings["prefill_setup"] = time.perf_counter() - start

    start = time.perf_counter()
    raw, caches = prefill.run(embed(PROMPT, 0), cross)
    token = next_token(raw, 0)
    timings["prefill"] = time.perf_counter() - start
    del prefill
    cleanup("prefill -> decode")

    start = time.perf_counter()
    decoder = GemvDecoderRuntime(build_root / "decode", decoder_weights)
    check_contexts("decode resident")
    timings["decode_setup"] = time.perf_counter() - start

    generated = [token]
    start = time.perf_counter()
    while token != EOS and len(generated) < MAX_NEW_TOKENS:
        raw, caches = decoder.step(
            embed([token], len(PROMPT) + len(generated) - 1), caches
        )
        token = next_token(raw, len(generated))
        generated.append(token)
    timings["decode"] = time.perf_counter() - start
    del decoder
    cleanup("final")
    return frontend, encoder, generated


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe up to 30 s of 16-kHz mono PCM16 audio on Phoenix/NPU1."
    )
    parser.add_argument(
        "--wav",
        type=Path,
        default=os.environ.get("WHISPER_TEST_WAV"),
        help="input WAV (default: $WHISPER_TEST_WAV)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=default_build_dir(),
        help="compiled kernel directory (default: $IRON_WHISPER_BUILD_DIR or "
        "<artifact dir>/pipeline-build); keep it short on Windows",
    )
    parser.add_argument(
        "--reference-text",
        type=Path,
        help="optional transcript file; reports word error rate against it",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="skip the CPU FP32 reference; the result is then unverified",
    )
    args = parser.parse_args()
    if args.wav is None:
        parser.error("pass --wav or set WHISPER_TEST_WAV")
    wav = Path(args.wav)

    device = aie_utils.ensure_current_device()
    if device is None or device.resolve().name != "npu1":
        raise RuntimeError("Expected Phoenix/NPU1")

    checkpoint = whisper_checkpoint()
    snapshot = checkpoint.parent
    config = json.loads((snapshot / "config.json").read_text("utf-8"))

    from transformers import WhisperProcessor

    tokenizer = WhisperProcessor.from_pretrained(
        snapshot, local_files_only=True
    ).tokenizer

    waveform = load_wav(wav)
    duration = waveform.numel() / 16000
    if duration > 30.0:
        print(f"WARNING: {duration:.1f} s of audio; only the first 30 s is used")
    mel = log_mel_spectrogram(waveform, target_frames=FRAMES).float()
    print(f"Audio: {wav} ({duration:.3f} s), mel {tuple(mel.shape)}")

    reference = None
    if not args.no_reference:
        start = time.perf_counter()
        reference = cpu_reference(
            snapshot,
            mel,
            config["suppress_tokens"],
            config["begin_suppress_tokens"],
        )
        print(f"CPU FP32 reference: {time.perf_counter() - start:.3f} s")

    timings = {}
    frontend, encoder, generated = transcribe(mel, checkpoint, args.build_dir, timings)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    eos = generated[-1] == EOS

    print()
    print("Token IDs:", generated)
    print("Text:", text.strip())
    print("EOS reached:", eos)

    token_match = None
    if reference is not None:
        ref_frontend, ref_encoder, ref_tokens = reference
        cosine = F.cosine_similarity(
            encoder.flatten(), ref_encoder.flatten(), dim=0
        ).item()
        print(f"Frontend NRMSE vs FP32: {100 * nrmse(frontend, ref_frontend):.3f}%")
        print(f"Encoder NRMSE vs FP32: {100 * nrmse(encoder, ref_encoder):.3f}%")
        print(f"Encoder cosine vs FP32: {cosine:.6f}")
        token_match = generated == ref_tokens
        print(f"Exact CPU FP32 token match: {token_match}")
        if not token_match:
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(generated, ref_tokens)) if a != b),
                min(len(generated), len(ref_tokens)),
            )
            print(f"  first difference at token {first_diff}")
            print("  CPU FP32 token IDs:", ref_tokens)

    if args.reference_text:
        truth = args.reference_text.read_text("utf-8")
        print(f"WER vs reference text: {100 * word_error_rate(truth, text):.1f}%")

    compute = sum(v for k, v in timings.items() if not k.endswith("_setup"))
    for key, value in timings.items():
        print(f"  {key:16s} {value:8.3f} s")
    print(f"  compute total    {compute:8.3f} s  (excludes kernel setup)")
    print(f"  realtime factor  {compute / duration:8.3f}")
    print(f"  decode rate      {(len(generated) - 1) / timings['decode']:8.2f} token/s")

    if not eos:
        raise SystemExit("FAILED: no end-of-text token")
    if token_match is False:
        raise SystemExit("FAILED: token mismatch with the CPU FP32 reference")
    status = "PASSED" if token_match else "COMPLETE (unverified)"
    print(f"Whisper-small Phoenix transcription: {status}")


if __name__ == "__main__":
    main()
