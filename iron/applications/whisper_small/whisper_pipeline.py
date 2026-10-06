#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Whisper-small transcription on Phoenix/NPU1.

Each 30 s window runs three phases, each kept within the NPU1 context cache
and cleaned up before the next:

1. frontend: conv1 / GELU / conv2 / GELU on the NPU at 3000 Mel frames.
2. encoder: 12 blocks at 1536 physical rows (1500 logical) with four-column
   GEMMs; LayerNorm, GELU and masked softmax on the CPU. The resident 768x768
   GEMM then computes the cross-attention K/V of every decoder block.
3. decoder: the prompt, then one generated token per step, with four-column
   NPU GEMVs and resident BF16 weights; LayerNorm, attention and GELU on the
   CPU.

The vocabulary projection and the greedy token selection run on the CPU.

Audio up to 30 s is decoded as one window without timestamps. Longer audio is
decoded window by window with timestamps, following OpenAI Whisper's
sequential long-form algorithm (greedy, no temperature fallback, no
conditioning on previous text).

Acceptance: the generated token ids of every window must exactly match a CPU
FP32 Hugging Face greedy decode of the same log-Mel features under the same
token selection. The process exits non-zero on a mismatch or when a window
produces no end-of-text token.
"""

import argparse
import gc
import json
import os
import re
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

from whisper_audio import load_wav, log_mel_spectrogram, log_mel_spectrogram_long
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
    load_decoder_block_weights,
    load_decoder_embedding,
    load_decoder_final_layernorm,
    load_decoder_position_embedding,
    make_context,
    merge_heads,
    run_decoder_embedding,
    run_gemm,
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
# The same prompt without <|notimestamps|>, for long-form decoding.
TIMESTAMP_PROMPT = PROMPT[:3]
EOS = 50257
FIRST_NON_TEXT_SPECIAL = 50358
# <|0.00|>; each following token adds 0.02 s, i.e. two 10 ms Mel frames.
TIMESTAMP_BEGIN = 50364
TIMESTAMP_SECONDS = 0.02
HOP_SECONDS = 0.01
INPUT_STRIDE = 2
# The first timestamp may be at most 1.0 s.
MAX_INITIAL_TIMESTAMP = 50
# Accepted timestamp difference against the CPU reference, in 0.02 s steps.
TIMESTAMP_TOLERANCE = 2
MAX_NEW_TOKENS = 224
# CPU threads for torch during decoding; see PhoenixWhisper.decode.
DECODE_THREADS = 2

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
    return op, direct_callable(op)


def direct_callable(op):
    """Like ``op.get_callable()``, but reusing one XRT run object.

    The generic dispatch path re-validates the arguments and builds a new run
    for every call; reusing the run halves the per-call host cost (about
    0.4 -> 0.2 ms on Phoenix), which dominates the decoder's GEMVs. Falls
    back to the generic callable for kernels it does not cover.
    """

    import pyxrt
    from aie.utils.npukernel import NPUKernel

    handle = runtime_backend.load(
        NPUKernel(
            xclbin_path=op.xclbin_artifact.filename,
            kernel_name=op.xclbin_artifact.kernel_name,
            insts_path=op.insts_artifact.filename,
        )
    )
    if getattr(handle, "is_full_elf", False) or not handle.insts_bo:
        return op.get_callable()
    run = pyxrt.run(handle.kernel)
    for index, value in enumerate((3, handle.insts_bo, handle.insts.nbytes)):
        run.set_arg(index, value)
    completed = pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED

    def call(*tensors):
        for index, tensor in enumerate(tensors, 3):
            # Flush host writes; outputs stay device-current, so the next
            # read pulls them back.
            tensor.to("npu")
            run.set_arg(index, tensor.buffer_object())
        run.start()
        state = run.wait()
        if state != completed:
            raise RuntimeError(f"{op.xclbin_artifact.kernel_name}: kernel {state}")

    return call


def resident(t):
    """BF16 copy of ``t`` in an NPU-visible buffer, for repeated use."""

    return aie_utils.DEFAULT_TENSOR_CLASS.from_torch(t.to(torch.bfloat16).contiguous())


def write_bf16(buffer, x):
    bits = x.to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
    with buffer.overwrite() as data:
        data.view(np.uint16)[...] = bits.reshape(data.shape)


class BufferedGemm:
    """A compiled GEMM with its input and output buffers allocated once.

    Allocating XRT buffers costs more than the NPU work of most encoder
    GEMMs, so each call only copies data into the existing buffers.
    """

    def __init__(self, M, K, N, build_root, name, cols=COLUMNS):
        tc = aie_utils.DEFAULT_TENSOR_CLASS
        bf = np.dtype("bfloat16")
        self.op, self.fn = gemm(M, K, N, build_root, name, tile_m=16, cols=cols)
        self.a = tc((M, K), dtype=bf)
        self.b = tc((K, N), dtype=bf)
        self.c = tc((M, N), dtype=bf)

    def __call__(self, a, b):
        """``a @ b`` in FP32; ``b`` is a torch tensor or a resident buffer."""

        write_bf16(self.a, a)
        if isinstance(b, torch.Tensor):
            write_bf16(self.b, b)
            b = self.b
        self.fn(self.a, b, self.c)
        return self.c.to_torch().float()


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


def resident_encoder_weights(w):
    """Encoder block projection weights as resident (K, N) BF16 buffers."""

    return {
        name: resident(w[f"{name}.weight"].float().T)
        for name in (
            "attn.query",
            "attn.key",
            "attn.value",
            "attn.out",
            "mlp.0",
            "mlp.2",
        )
    }


def resident_cross_weights(w):
    return {
        name: resident(w[f"encoder_attn.{name}.weight"].float().T)
        for name in ("k_proj", "v_proj")
    }


class FullWindowEncoderRuntime:
    def __init__(self, build_root):
        def make(K, N, name, cols=COLUMNS):
            return BufferedGemm(SEQ_PAD, K, N, build_root, name, cols=cols)

        self.p768 = make(STATE, STATE, "gemm-768-768")
        self.score = make(HEAD_DIM, SEQ_PAD, "score")
        # N=64 is not a multiple of COLUMNS * tile_n, so this one uses one column.
        self.value = make(SEQ_PAD, HEAD_DIM, "value", cols=1)
        self.fc1 = make(STATE, MLP, "gemm-768-3072")
        self.fc2 = make(MLP, STATE, "gemm-3072-768")
        self.key_mask = torch.zeros(SEQ_PAD, dtype=torch.bool)
        self.key_mask[SEQ:] = True

    @staticmethod
    def layer_norm(x, weight, bias):
        return F.layer_norm(bf16(x), (STATE,), weight.float(), bias.float())

    def projection(self, x, weight, bias):
        out = self.p768(x, weight)
        if bias is not None:
            out += bias.float()
        return out

    def block(self, x, w, r):
        """One encoder block; ``w`` holds the FP32 weights and biases, ``r``
        the resident projection weights."""

        h = self.layer_norm(x, w["attn_ln.weight"], w["attn_ln.bias"])
        q = self.projection(h, r["attn.query"], w["attn.query.bias"])
        k = self.projection(h, r["attn.key"], None)
        v = self.projection(h, r["attn.value"], w["attn.value.bias"])
        qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)

        heads = []
        for head in range(HEADS):
            scores = self.score(qh[head] * SCALE, kh[head].T)
            scores.masked_fill_(self.key_mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            heads.append(self.value(probs, vh[head]))

        x = x + self.projection(
            merge_heads(torch.stack(heads)), r["attn.out"], w["attn.out.bias"]
        )

        h = self.layer_norm(x, w["mlp_ln.weight"], w["mlp_ln.bias"])
        up = self.fc1(h, r["mlp.0"])
        up += w["mlp.0.bias"].float()
        act = F.gelu(bf16(up))
        down = self.fc2(act, r["mlp.2"])
        down += w["mlp.2.bias"].float()
        x = x + down
        x[SEQ:] = 0.0
        return x

    def cross_kv(self, encoder_pad, decoder_weights, cross_weights):
        caches = []
        for w, r in zip(decoder_weights, cross_weights):
            k = self.p768(encoder_pad, r["k_proj"])[:SEQ]
            v = self.p768(encoder_pad, r["v_proj"])[:SEQ]
            v += w["encoder_attn.v_proj.bias"]
            caches.append((split_heads(k), split_heads(v)))
        return caches


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

    def __init__(self, build_root, block_weights, resident_weights):
        tc = aie_utils.DEFAULT_TENSOR_CLASS
        self.weights = block_weights
        self.resident = resident_weights
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
            self.ops[name], self.fns[name] = op, direct_callable(op)
            self.outputs[name] = tc((M,), dtype=np.dtype("bfloat16"))
        self.inputs = {
            size: tc((size,), dtype=np.dtype("bfloat16")) for size in (STATE, MLP)
        }

    @staticmethod
    def resident_weights(w):
        """The block's GEMV weights as resident BF16 (out, in) matrices."""

        qkv = torch.cat(
            [
                w["self_attn.q_proj.weight"],
                w["self_attn.k_proj.weight"],
                w["self_attn.v_proj.weight"],
            ]
        )
        return {
            "qkv": resident(qkv),
            "self_out": resident(w["self_attn.out_proj.weight"]),
            "cross_q": resident(w["encoder_attn.q_proj.weight"]),
            "cross_out": resident(w["encoder_attn.out_proj.weight"]),
            "fc1": resident(w["fc1.weight"]),
            "fc2": resident(w["fc2.weight"]),
        }

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
# Token selection
# ============================================================================


class TokenPolicy:
    """Whisper greedy-decoding logit filters.

    Without timestamps, all special tokens are suppressed. With timestamps,
    OpenAI Whisper's ApplyTimestampRules apply: timestamps come in pairs and
    never decrease, the first token is a timestamp of at most 1.0 s, and a
    timestamp is forced whenever the total timestamp probability exceeds that
    of the most likely text token.
    """

    def __init__(self, config, timestamps):
        self.suppress = config["suppress_tokens"]
        self.begin_suppress = config["begin_suppress_tokens"]
        self.timestamps = timestamps

    def __call__(self, logits, generated):
        logits = logits.float().clone()
        logits[self.suppress] = float("-inf")
        if not generated:
            logits[self.begin_suppress] = float("-inf")
        if not self.timestamps:
            logits[FIRST_NON_TEXT_SPECIAL:] = float("-inf")
            return logits

        logits[FIRST_NON_TEXT_SPECIAL:TIMESTAMP_BEGIN] = float("-inf")
        last = len(generated) >= 1 and generated[-1] >= TIMESTAMP_BEGIN
        penultimate = len(generated) < 2 or generated[-2] >= TIMESTAMP_BEGIN
        if last:
            if penultimate:
                logits[TIMESTAMP_BEGIN:] = float("-inf")
            else:
                logits[:EOS] = float("-inf")

        stamps = [t for t in generated if t >= TIMESTAMP_BEGIN]
        if stamps:
            floor = stamps[-1] if last and not penultimate else stamps[-1] + 1
            logits[TIMESTAMP_BEGIN:floor] = float("-inf")
        if not generated:
            logits[:TIMESTAMP_BEGIN] = float("-inf")
            logits[TIMESTAMP_BEGIN + MAX_INITIAL_TIMESTAMP + 1 :] = float("-inf")

        logprobs = torch.log_softmax(logits, dim=-1)
        timestamp_logprob = torch.logsumexp(logprobs[TIMESTAMP_BEGIN:], dim=-1)
        if timestamp_logprob > logprobs[:TIMESTAMP_BEGIN].max():
            logits[:TIMESTAMP_BEGIN] = float("-inf")
        return logits


def greedy_decode(step, prompt, policy):
    """Feed ``prompt``, then pick tokens until end-of-text or the length limit.

    ``step(token_ids)`` appends tokens to the decoder state and returns the
    logits of the last one.
    """

    logits = step(prompt)
    generated = []
    while True:
        token = int(policy(logits, generated).argmax())
        generated.append(token)
        if token == EOS or len(generated) >= MAX_NEW_TOKENS:
            return generated
        logits = step([token])


# ============================================================================
# NPU and CPU backends: one 30 s window of log-Mel features -> tokens
# ============================================================================


class PhoenixWhisper:
    """Phoenix/NPU1 backend; each phase releases its contexts before the next."""

    def __init__(self, checkpoint, build_root):
        self.checkpoint = checkpoint
        self.build_root = Path(build_root)
        self.encoder_weights = [
            load_block_weights(checkpoint, b) for b in range(BLOCKS)
        ]
        self.decoder_weights = [
            load_decoder_block_weights(checkpoint, b) for b in range(BLOCKS)
        ]
        # Converted to BF16 once and reused by every window.
        self.encoder_resident = [
            resident_encoder_weights(w) for w in self.encoder_weights
        ]
        self.cross_resident = [resident_cross_weights(w) for w in self.decoder_weights]
        self.decoder_resident = [
            GemvDecoderRuntime.resident_weights(w) for w in self.decoder_weights
        ]
        with safe_open(str(checkpoint), framework="pt", device="cpu") as h:
            self.encoder_ln = (
                h.get_tensor("model.encoder.layer_norm.weight").float(),
                h.get_tensor("model.encoder.layer_norm.bias").float(),
            )
        self.token_embedding = load_decoder_embedding(checkpoint)
        self.position_embedding = load_decoder_position_embedding(checkpoint)
        self.final_ln = tuple(
            t.float() for t in load_decoder_final_layernorm(checkpoint)
        )
        self.timings = {}

    def _time(self, key, start):
        self.timings[key] = self.timings.get(key, 0.0) + time.perf_counter() - start

    def encode(self, mel):
        """Return frontend output, encoder output and per-block cross K/V."""

        start = time.perf_counter()
        x = run_frontend_full(mel, self.checkpoint, self.build_root / "frontend")
        self._time("frontend", start)
        frontend = x[:SEQ].clone()
        cleanup("frontend -> encoder")

        start = time.perf_counter()
        runtime = FullWindowEncoderRuntime(self.build_root / "encoder")
        check_contexts("encoder resident")
        self._time("encoder_setup", start)

        start = time.perf_counter()
        for block in range(BLOCKS):
            x = runtime.block(
                x, self.encoder_weights[block], self.encoder_resident[block]
            )
            if not torch.isfinite(x[:SEQ]).all():
                raise RuntimeError(f"Encoder block {block} produced non-finite values")
        encoder = F.layer_norm(bf16(x[:SEQ]), (STATE,), *self.encoder_ln)
        self._time("encoder", start)

        start = time.perf_counter()
        cross = runtime.cross_kv(
            pad_to(encoder, SEQ_PAD), self.decoder_weights, self.cross_resident
        )
        self._time("cross_kv", start)
        del runtime
        cleanup("encoder -> decoder")
        return frontend, encoder, cross

    def decode(self, cross, prompt, policy):
        start = time.perf_counter()
        decoder = GemvDecoderRuntime(
            self.build_root / "decode", self.decoder_weights, self.decoder_resident
        )
        check_contexts("decoder resident")
        self._time("decode_setup", start)

        empty = torch.zeros((HEADS, 0, HEAD_DIM))
        state = {
            "position": 0,
            "caches": {
                block: {"self_k": empty, "self_v": empty, "cross_k": k, "cross_v": v}
                for block, (k, v) in enumerate(cross)
            },
        }

        def step(token_ids):
            for token in token_ids:
                x = run_decoder_embedding(
                    token_ids=torch.tensor([token], dtype=torch.long),
                    position_offset=state["position"],
                    checkpoint=self.checkpoint,
                    token_embedding=self.token_embedding,
                    position_embedding=self.position_embedding,
                )
                raw, state["caches"] = decoder.step(x, state["caches"])
                state["position"] += 1
            final = F.layer_norm(raw[-1:].float(), (STATE,), *self.final_ln)
            return torch.matmul(final, self.token_embedding.T)[0]

        start = time.perf_counter()
        # Each step alternates short NPU calls with small CPU ops; idle torch
        # worker threads spinning between them slow every NPU dispatch.
        threads = torch.get_num_threads()
        torch.set_num_threads(min(threads, DECODE_THREADS))
        try:
            tokens = greedy_decode(step, prompt, policy)
        finally:
            torch.set_num_threads(threads)
        self._time("decode", start)
        self.timings["decode_tokens"] = (
            self.timings.get("decode_tokens", 0) + len(prompt) + len(tokens) - 1
        )
        del decoder
        cleanup("decoder -> next")
        return tokens

    def window(self, mel, prompt, policy):
        frontend, encoder, cross = self.encode(mel)
        return self.decode(cross, prompt, policy), (frontend, encoder)


class CpuWhisper:
    """Hugging Face FP32 reference backend with the same token selection."""

    def __init__(self, snapshot):
        from transformers import WhisperForConditionalGeneration

        self.model = WhisperForConditionalGeneration.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.float32
        ).eval()

    @torch.no_grad()
    def window(self, mel, prompt, policy):
        encoder = self.model.model.encoder
        conv = F.gelu(encoder.conv2(F.gelu(encoder.conv1(mel))))[0].T
        frontend = encoder.embed_positions.weight[:SEQ] + conv
        hidden = encoder(mel).last_hidden_state
        state = {"past": None}

        def step(token_ids):
            out = self.model(
                encoder_outputs=(hidden,),
                decoder_input_ids=torch.tensor([token_ids]),
                past_key_values=state["past"],
                use_cache=True,
            )
            state["past"] = out.past_key_values
            return out.logits[0, -1]

        return greedy_decode(step, prompt, policy), (frontend, hidden[0])


# ============================================================================
# Long-form transcription
# ============================================================================


def decode_window(backend, mel, seek, content_frames, policy):
    size = min(FRAMES, content_frames - seek)
    segment = torch.zeros((1, MELS, FRAMES))
    segment[:, :, :size] = mel[:, :, seek : seek + size]
    tokens, _ = backend.window(segment, TIMESTAMP_PROMPT, policy)
    return size, tokens


def transcribe_long(backend, mel, content_frames, policy):
    """OpenAI Whisper's sequential long-form decoding, without temperature
    fallback or conditioning on previous text.

    Each window is decoded with timestamps. The next window starts at the last
    complete timestamp pair, or after the whole window when the decoder ends
    with a single timestamp or produces none.
    """

    seek = 0
    windows = []
    while seek < content_frames:
        size, tokens = decode_window(backend, mel, seek, content_frames, policy)
        windows.append({"seek": seek, "tokens": tokens})

        text_end = [t for t in tokens if t != EOS]
        is_stamp = [t >= TIMESTAMP_BEGIN for t in text_end]
        single_ending = is_stamp[-2:] == [False, True]
        pairs = [
            i + 1 for i in range(len(is_stamp) - 1) if is_stamp[i] and is_stamp[i + 1]
        ]
        advance = size
        if pairs and not single_ending:
            last_stamp = text_end[pairs[-1] - 1] - TIMESTAMP_BEGIN
            advance = last_stamp * INPUT_STRIDE
        seek += advance if advance > 0 else size
    return windows


def windows_match(actual, expected):
    """Same text tokens at the same positions, and every timestamp within
    TIMESTAMP_TOLERANCE steps.

    Adjacent timestamps are often near-ties, so a BF16 decoder may legitimately
    pick a neighbour of the FP32 choice; text tokens must still be identical.
    Returns (match, largest timestamp difference in steps).
    """

    if len(actual) != len(expected):
        return False, None
    worst = 0
    for a, e in zip(actual, expected):
        if a >= TIMESTAMP_BEGIN and e >= TIMESTAMP_BEGIN:
            worst = max(worst, abs(a - e))
        elif a != e:
            return False, None
    return worst <= TIMESTAMP_TOLERANCE, worst


def segments_of(windows, tokenizer):
    """(start s, end s, text) for every timestamp-delimited segment."""

    result = []
    for window in windows:
        offset = window["seek"] * HOP_SECONDS
        start, text = None, []
        for token in window["tokens"]:
            if token >= TIMESTAMP_BEGIN:
                time_s = offset + (token - TIMESTAMP_BEGIN) * TIMESTAMP_SECONDS
                if start is None:
                    start = time_s
                elif text:
                    result.append((start, time_s, tokenizer.decode(text).strip()))
                    start, text = time_s, []
                else:
                    start = time_s
            elif token < EOS:
                text.append(token)
        if text:
            result.append((start or offset, None, tokenizer.decode(text).strip()))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe 16-kHz mono PCM16 audio with Whisper-small on "
        "Phoenix/NPU1. Audio longer than 30 s is decoded window by window."
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
        "--long-form",
        action="store_true",
        help="use timestamp-based long-form decoding even for audio up to 30 s",
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
    long_form = args.long_form or duration > 30.0
    print(
        f"Audio: {wav} ({duration:.3f} s), {'long-form' if long_form else 'one window'}"
    )

    npu = PhoenixWhisper(checkpoint, args.build_dir)

    if long_form:
        mel, content_frames = log_mel_spectrogram_long(waveform)
        policy = TokenPolicy(config, timestamps=True)
        windows = transcribe_long(npu, mel, content_frames, policy)
        generated = [t for w in windows for t in w["tokens"]]
        for start_s, end_s, text in segments_of(windows, tokenizer):
            end = "   ?  " if end_s is None else f"{end_s:6.2f}"
            print(f"  [{start_s:6.2f} -> {end}] {text}")
        eos = all(w["tokens"][-1] == EOS for w in windows)
        print(f"Windows: {len(windows)} (seek frames {[w['seek'] for w in windows]})")
        token_match = None
        if not args.no_reference:
            # The reference decodes the same windows, so that a timestamp
            # near-tie in one window does not shift every later window. It is
            # loaded after the NPU run, which it would otherwise slow down.
            cpu = CpuWhisper(snapshot)
            start = time.perf_counter()
            token_match, worst = True, 0
            for i, window in enumerate(windows):
                _, expected = decode_window(
                    cpu, mel, window["seek"], content_frames, policy
                )
                match, diff = windows_match(window["tokens"], expected)
                if not match:
                    token_match = False
                    print(f"  window {i} (seek {window['seek']}) differs")
                    print("    NPU:", window["tokens"])
                    print("    CPU:", expected)
                elif diff:
                    worst = max(worst, diff)
            print(f"CPU FP32 reference: {time.perf_counter() - start:.3f} s")
            if token_match:
                print(
                    f"Largest timestamp difference: {worst} "
                    f"({worst * TIMESTAMP_SECONDS:.2f} s)"
                )
    else:
        mel = log_mel_spectrogram(waveform, target_frames=FRAMES).float()
        policy = TokenPolicy(config, timestamps=False)
        generated, (frontend, encoder) = npu.window(mel, PROMPT, policy)
        eos = generated[-1] == EOS
        token_match = None
        if not args.no_reference:
            cpu = CpuWhisper(snapshot)
            start = time.perf_counter()
            ref_tokens, (ref_frontend, ref_encoder) = cpu.window(mel, PROMPT, policy)
            print(f"CPU FP32 reference: {time.perf_counter() - start:.3f} s")
            cosine = F.cosine_similarity(
                encoder.flatten(), ref_encoder.flatten(), dim=0
            ).item()
            print(f"Frontend NRMSE vs FP32: {100 * nrmse(frontend, ref_frontend):.3f}%")
            print(f"Encoder NRMSE vs FP32: {100 * nrmse(encoder, ref_encoder):.3f}%")
            print(f"Encoder cosine vs FP32: {cosine:.6f}")
            token_match = generated == ref_tokens
            if not token_match:
                print("  CPU FP32 token IDs:", ref_tokens)

    text = tokenizer.decode(
        [t for t in generated if t < EOS], skip_special_tokens=True
    ).strip()
    print()
    print("Token IDs:", generated)
    print("Text:", text)
    print("EOS reached:", eos)
    if token_match is not None:
        if long_form:
            print(f"CPU FP32 match (text exact, timestamps +-2): {token_match}")
        else:
            print(f"Exact CPU FP32 token match: {token_match}")

    if args.reference_text:
        truth = args.reference_text.read_text("utf-8")
        print(f"WER vs reference text: {100 * word_error_rate(truth, text):.1f}%")

    timings = npu.timings
    decode_tokens = timings.pop("decode_tokens")
    compute = sum(v for k, v in timings.items() if not k.endswith("_setup"))
    for key, value in timings.items():
        print(f"  {key:16s} {value:8.3f} s")
    print(f"  compute total    {compute:8.3f} s  (excludes kernel setup)")
    print(f"  realtime factor  {compute / duration:8.3f}")
    print(f"  decode rate      {decode_tokens / timings['decode']:8.2f} token/s")

    if not eos:
        raise SystemExit("FAILED: no end-of-text token")
    if token_match is False:
        raise SystemExit("FAILED: token mismatch with the CPU FP32 reference")
    status = "PASSED" if token_match else "COMPLETE (unverified)"
    print(f"Whisper-small Phoenix transcription: {status}")


if __name__ == "__main__":
    main()
