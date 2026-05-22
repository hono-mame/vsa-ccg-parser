# Research Log

VSA による CCG パーサの重ね合わせに関する研究の週次進捗・思考メモ。

書き方の方針:
- **日付は新しいものを上に**追記する（逆時系列）
- 各エントリは「何をやった / 何が分かった / 次に何をやるか」の3点を最低限カバーする
- 失敗した実験・捨てた仮説も残す（後で「なぜそれをやらなかったか」を思い出せる）
- 参考文献は本文中に `[Author Year]` で引用し、最下部の References に詳細を書く

---

## 2026-05-22

### やったこと
- リポジトリ `vsa-ccg` をスキャフォールド（`src/vsa_ccg/`, `sandbox/`, `experiments/`, etc.）
- `pyproject.toml` で依存（torch, torch-hd, numpy, scipy）を宣言
- 学習用スクリプト 3 本を `sandbox/` に整理
  - `einsum.py`: 主語ベクトル × 他動詞テンソル × 目的語ベクトル の縮約（メモリ問題の可視化）
  - `einsum_test.py`: `np.einsum` の各種記法を網羅
  - `torchhd_demo.py`: `bind` / `bundle` / `unbind` と役割ベクトルによる文表現

### 分かったこと
- 他動詞を $d \times d \times d$ のテンソルで持つナイーブな方法は $d=500$ で約 1GB。これでは実用にならない。
  → VSA の `bind` で次元を増やさず役割と値を結びつけるアプローチが必要な理由を体感。
- `torchhd.bind` は MAP 系では自己逆元（bind を 2 回かけると元に戻る）。これにより役割ベクトルでクエリすると元の値が取り出せる。

### 次にやること
- [ ] CCG の基本コンビネータ（forward/backward application, composition）を VSA でどう表現するか調査
- [ ] 既存研究のサーベイ: 「VSA + 構文解析」「holographic reduced representations + parsing」
- [ ] `src/vsa_ccg/vsa/` に bind/bundle のラッパを書き始める（torchhd のどの VSA モデル — MAP, HRR, FHRR — を採用するか決める）

---

## References

<!-- 例:
- Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. *Cognitive Computation*, 1(2), 139–159.
- Steedman, M. (2000). *The Syntactic Process*. MIT Press.
-->
