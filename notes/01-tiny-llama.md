```
原始 Transformer                          现代 Decoder-only LLM

Input / Output Embedding                  embed_tokens
        │                                      │
Positional Encoding                      RoPE 等位置编码
        │                                      │
        ▼                                      ▼
┌─────────────────────┐               ┌─────────────────────┐
│ Multi-Head Attention│               │ Self Attention      │
│ Add & Norm          │               │ + residual + norm   │
├─────────────────────┤               ├─────────────────────┤
│ Feed Forward        │       ≈       │ MLP                 │
│ Add & Norm          │               │ + residual + norm   │
└─────────────────────┘               └─────────────────────┘
        │                                      │
       × N                                    × N
        │                                      │
        ▼                                      ▼
Linear                                  lm_head (Linear)
        │                                      │
Softmax                                 logits → softmax/sampling
```


Old-fashioned MLP
```
输入 768 维
   ↓
Linear(768 → 3072)
   ↓
ReLU
   ↓
Linear(3072 → 768)
   ↓
输出 768 维
```
<=>
```
mlp = nn.Sequential(
    nn.Linear(768, 3072),
    nn.ReLU(),
    nn.Linear(3072, 768),
)
```

Gated MLP
```
              ┌─ Linear ─ SiLU ─┐
x ────────────┤                  × ── Linear ──► output
              └─ Linear ────────┘
```
