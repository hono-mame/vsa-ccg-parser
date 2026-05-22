import torch
import torchhd

d = 10000  # VSAは高次元が前提

# ランダムな基底ベクトルを3つ作る（item memory の素）
cat   = torchhd.random(1, d).squeeze(0)
chase = torchhd.random(1, d).squeeze(0)
mouse = torchhd.random(1, d).squeeze(0)

# 別々のランダムベクトルは「ほぼ直交」= 類似度ほぼ0
print("cat vs mouse:", torchhd.cosine_similarity(cat, mouse).item())   # ≈ 0
print("cat vs cat  :", torchhd.cosine_similarity(cat, cat).item())     # = 1

# --- bind（結合, ⊛, 各要素の掛け算）: 2つを結びつける。結果は両方と非類似になる ---
# https://torchhd.readthedocs.io/en/stable/generated/torchhd.bind.html
bound = torchhd.bind(cat, chase)
print("bind結果 vs cat:", torchhd.cosine_similarity(bound, cat).item())  # ≈ 0
print("次元:", bound.shape)  # (10000,) ← ★何度bindしても増えない！

# --- bundle（重ね合わせ, ⊕, 各要素の足し算）: 束ねる。結果は構成要素すべてと類似 ---
# 重ね合わせた場合、比較する要素が重ね合わせたものに入っている場合は類似度が高くなり、そうでない場合はほぼ0になる
# https://torchhd.readthedocs.io/en/stable/generated/torchhd.bundle.html
bundled = torchhd.bundle(cat, mouse)
print("bundle結果 vs cat  :", torchhd.cosine_similarity(bundled, cat).item())    # > 0
print("bundle結果 vs mouse:", torchhd.cosine_similarity(bundled, mouse).item())  # > 0
print("bundle結果 vs chase:", torchhd.cosine_similarity(bundled, chase).item())  # ≈ 0（入れてない）


# 役割ベクトル（スロット）を用意
r_subj = torchhd.random(1, d).squeeze(0) # 要素が +1 か -1 のランダムな値を、1行 d 列（例：1万列）作成
r_obj  = torchhd.random(1, d).squeeze(0)

# 文「猫がネズミを追う」を1本のベクトルに畳み込む
#   (主語=猫) と (目的語=ネズミ) を bind して bundle
sentence = torchhd.bundle(
    torchhd.bind(r_subj, cat),
    torchhd.bind(r_obj, mouse),
)
print("文ベクトルの次元:", sentence.shape)  # (10000,) ← ずっとd次元のまま

# デコード: 「主語は何？」→ r_subj で unbind して item memory と照合
query = torchhd.bind(sentence, r_subj)   # bindは自己逆（MAP系）なので unbind になる
memory = torch.stack([cat, mouse, chase])
sims = torchhd.cosine_similarity(query, memory)
names = ["cat", "mouse", "chase"]
print(dict(zip(names, sims.tolist())))
# → cat が突出して高い = 「主語は猫」と正しく取り出せた