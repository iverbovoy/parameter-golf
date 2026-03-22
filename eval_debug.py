"""Quick eval-only script to debug sliding window issue."""
import io, math, os, sys, zlib, time
import torch
import torch.distributed as dist

# Reuse everything from train_gpt
from train_gpt import (
    Hyperparameters, GPT,
    build_sentencepiece_luts, load_validation_tokens,
    eval_val_sliding, dequantize_state_dict_int8
)
import sentencepiece as spm

def main():
    args = Hyperparameters()
    # Override eval params from env
    args.eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 1024))
    args.eval_stride = int(os.environ.get("EVAL_STRIDE", 256))

    device = torch.device("cuda:0")

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    # Load val tokens
    val_tokens = load_validation_tokens(args.val_files, args.eval_seq_len)
    print(f"val tokens: {val_tokens.numel():,}")

    # Build model
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_repeats=args.num_repeats,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        model_dim=args.model_dim,
        mlp_mult=args.mlp_mult,
        rope_base=args.rope_base,
        tie_embeddings=args.tie_embeddings,
        qk_gain_init=args.qk_gain_init,
        logit_softcap=args.logit_softcap,
        num_value_embeds=args.num_value_embeds,
        tied_embed_init_std=args.tied_embed_init_std,
    ).to(device)

    # Load int8 checkpoint
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob)), map_location="cpu")
    model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    print(f"Loaded int8 checkpoint, params: {sum(p.numel() for p in model.parameters()):,}")

    # Run sliding window eval
    print(f"Running sliding window eval: seq_len={args.eval_seq_len} stride={args.eval_stride}")
    t0 = time.perf_counter()
    sw_loss, sw_bpb = eval_val_sliding(
        args, model, rank=0, world_size=1, device=device,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )
    elapsed = time.perf_counter() - t0
    print(f"sliding_window val_loss:{sw_loss:.4f} val_bpb:{sw_bpb:.4f} seq_len:{args.eval_seq_len} stride:{args.eval_stride} time:{elapsed:.1f}s")
    print(f"exact val_loss:{sw_loss:.8f} val_bpb:{sw_bpb:.8f}")

if __name__ == "__main__":
    main()
