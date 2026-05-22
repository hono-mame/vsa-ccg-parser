# sandbox

ライブラリの挙動を理解するための**学習用スクリプト**置き場です。研究本体のコードではありません。

| ファイル | 内容 |
|----------|------|
| `einsum.py` | `np.einsum` を使った主語・動詞・目的語テンソルの縮約デモ |
| `einsum_test.py` | `np.einsum` の各種記法（転置・合計・行列積・テンソル縮約など）の動作確認 |
| `torchhd_demo.py` | `torchhd` の `bind` / `bundle` / `unbind` の基本操作と、役割ベクトルによる文表現の例 |

実行例:

```bash
python3 sandbox/torchhd_demo.py
```
