import numpy as np

# ==========================================
# データの準備 (目で追える小さなサイズ)
# ==========================================
# 2次元配列 (行列)
A = np.arange(1, 7).reshape(2, 3)  # 2行3列
B = np.arange(1, 7).reshape(3, 2)  # 3行2列
C = np.arange(1, 7).reshape(2, 3)  # 2行3列 (Aと同じ形)
S = np.arange(1, 10).reshape(3, 3) # 3行3列 (正方行列)

# 3次元配列 (テンソル: バッチサイズが含まれるイメージ)
# (バッチサイズ=2, 行=2, 列=3)
T1 = np.arange(1, 13).reshape(2, 2, 3) 
# (バッチサイズ=2, 行=3, 列=2)
T2 = np.arange(1, 13).reshape(2, 3, 2) 

print("=== 1. 単一の配列に対する操作 (変形・集約) ===")

# ① 転置 (Transpose)
# i(行)とj(列)を入れ替える
print("\n[元の行列 A]")
print(A)
res1 = np.einsum('ij->ji', A)
print("\n[転置] 'ij->ji' A(2,3) -> (3,2)")
print(res1)

print("--------------------------------------------------------")

# ② 全要素の合計 (Sum)
# 出力に添字を書かない(->の右が空) = 全ての次元(i, j)を潰して足し合わせる
res2 = np.einsum('ij->', A)
print("\n[全合計] 'ij->' A(2,3) -> スカラー")
print(res2) # = np.sum(A)

print("--------------------------------------------------------")

# ③ 特定の軸に沿った合計 (Axis Sum)
# j(列)を出力に書かない = 列方向を潰して足す (各行の合計が出る)
res3 = np.einsum('ij->i', A)
print("\n[行ごとの合計] 'ij->i' A(2,3) -> (2,)")
print(res3) # = np.sum(A, axis=1)

print("--------------------------------------------------------")

# ④ 対角成分の抽出 (Diagonal)
# 同じ文字'ii'を使うと、行と列のインデックスが同じ場所(対角)だけを拾う
print("\n[元の行列 S]")
print(S)
res4 = np.einsum('ii->i', S)
print("\n[対角成分] 'ii->i' S(3,3) -> (3,)")
print(res4) # = np.diag(S)

print("--------------------------------------------------------")

# ⑤ トレース (Trace: 対角成分の和)
# 対角成分'ii'を拾い、出力に書かないことでそれを足し合わせる
res5 = np.einsum('ii->', S)
print("\n[トレース] 'ii->' S(3,3) -> スカラー")
print(res5) # = np.trace(S)


print("\n=== 2. 複数の配列に対する操作 (掛け算・縮約) ===")

# ⑥ 要素ごとの掛け算 (Hadamard Product)
# 同じ添字のまま出力する = 掛け算だけして足し合わせ(潰す)は行わない
print("\n[元の行列 A]")
print(A)
print("\n[元の行列 C]")
print(C)
res6 = np.einsum('ij,ij->ij', A, C)
print("\n[要素ごとの積] 'ij,ij->ij', A(2,3), C(2,3) -> (2,3)")
print(res6) # = A * C

print("--------------------------------------------------------")

# ⑦ 行列の積 (Matrix Multiplication)
# jが共通している＆出力に無い = jの次元で掛け合わせて足す(縮約する)
print("\n[元の行列 A]")
print(A)
print("\n[元の行列 B]")
print(B)
res7 = np.einsum('ij,jk->ik', A, B)
print("\n[行列の積] 'ij,jk->ik', A(2,3), B(3,2) -> (2,2)")
print(res7) # = np.dot(A, B) または A @ B

print("--------------------------------------------------------")

# ⑧ バッチ行列積 (Batched Matrix Multiplication) ★ディープラーニングで超頻出
# b(バッチ)はそのまま残し、各バッチの中で ij,jk->ik の行列積を行う
print("\n[元の行列 T1]")
print(T1)
print("\n[元の行列 T2]")
print(T2)
res8 = np.einsum('bij,bjk->bik', T1, T2)
print("\n[バッチ行列積] 'bij,bjk->bik', T1(2,2,3), T2(2,3,2) -> (2,2,2)")
print(res8.shape) # = np.matmul(T1, T2) または T1 @ T2
print(res8)

print("--------------------------------------------------------")

# ⑨ テンソル縮約 (Tensor Contraction)
# 3次元データと2次元データなど、複雑な次元同士を特定の軸(k)で掛け合わせて潰す
# T1の最後の軸(k)と、Bの最初の軸(k)で縮約する
print("\n[元の行列 T1]")
print(T1)
print("\n[元の行列 B]")
print(B)
res9 = np.einsum('ijk,kl->ijl', T1, B)
print("\n[テンソル縮約] 'ijk,kl->ijl', T1(2,2,3), B(3,2) -> (2,2,2)")
print(res9.shape) # = np.tensordot(T1, B, axes=1)
print(res9)

print("--------------------------------------------------------")

# ⑩ テンソル外積 (Outer Product)
# 共通の文字を持たせず、全ての添字を出力に並べる = 全組み合わせの掛け算(次元が増える)
res10 = np.einsum('ij,kl->ijkl', A, B)
print("\n[テンソル外積] 'ij,kl->ijkl', A(2,3), B(3,2) -> (2,3,3,2)")
print(res10.shape)
print(res10)