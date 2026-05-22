import numpy as np

d = 500  # 標準的な分散表現の次元の半分

subj = np.random.randn(d)          # 主語ベクトル  : d
verb = np.random.randn(d, d, d)    # 他動詞テンソル: d×d×d ← ここが既に重い
obj  = np.random.randn(d)          # 目的語ベクトル: d

print("他動詞テンソルの要素数:", verb.size)              # 1.25億
print("メモリ:", verb.nbytes / 1e9, "GB")               # 1GB

sentence = np.einsum('i,ijk,k->j', subj, verb, obj)  # 主語ベクトル、他動詞テンソル、目的語ベクトルを縮約して文ベクトルを得る
print("文ベクトルの形状:", sentence.shape)             # (d,) =