# vsa-ccg

VSA（Vector Symbolic Architectures, 超高次元表現）による CCG（Combinatory Categorial Grammar）パーサの**重ね合わせ**に関する研究用リポジトリです。

複数の構文解析候補を高次元ベクトル空間上で同時に表現・操作することで、曖昧性のある自然言語文を効率的に並列処理する手法を探求しています。

## ディレクトリ構成

```
.
├── src/vsa_ccg/      # 本体コード（パッケージ）
├── experiments/      # 実験スクリプト（exp001_*, exp002_* のように番号付き）
├── notebooks/        # 試行錯誤・可視化用 Jupyter notebook
├── sandbox/          # ライブラリ理解のための学習用スクリプト（研究本体ではない）
├── tests/            # src/ に対する単体テスト
├── docs/             # 設計メモ・発表資料・週報など
├── pyproject.toml
└── README.md
```

## セットアップ

Python 3.11 以上を想定。`uv` を推奨しますが、`pip` でも動きます。

### uv を使う場合（推奨）

```bash
uv venv
uv pip install -e ".[dev]"
```

### pip を使う場合

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 主要な依存ライブラリ

- [PyTorch](https://pytorch.org/) — テンソル演算基盤
- [torchhd](https://torchhd.readthedocs.io/) — VSA 演算（bind, bundle, etc.）
- NumPy / SciPy — 数値計算

## 参考文献

（随時追加）

- Kanerva, P. (2009). *Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors.*
- Steedman, M. (2000). *The Syntactic Process.* （CCG）

## ライセンス

未定（研究用途）。
