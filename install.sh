#!/bin/bash
# =========================================================
# 動画文字起こしシステム セットアップスクリプト (macOS / Linux)
# =========================================================
# 実行方法: bash install.sh

set -e

echo "========================================"
echo "  動画文字起こしシステム セットアップ"
echo "========================================"

# ── 1. ffmpeg の確認・インストール ────────────────────────
echo ""
echo "[1/3] ffmpeg を確認中..."

if command -v ffmpeg &>/dev/null; then
    echo "  ✓ ffmpeg は既にインストール済みです: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "  ffmpeg が見つかりません。インストールします..."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS (Homebrew)
        if ! command -v brew &>/dev/null; then
            echo "  Homebrew が必要です: https://brew.sh"
            exit 1
        fi
        brew install ffmpeg
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux (apt)
        sudo apt-get update && sudo apt-get install -y ffmpeg
    else
        echo "  お使いの OS に合わせて ffmpeg を手動でインストールしてください。"
        echo "  https://ffmpeg.org/download.html"
        exit 1
    fi
    echo "  ✓ ffmpeg をインストールしました"
fi

# ── 2. Python バージョン確認 ───────────────────────────────
echo ""
echo "[2/3] Python を確認中..."

if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version)
    echo "  ✓ $PY_VER"
elif command -v python &>/dev/null; then
    PY_VER=$(python --version)
    echo "  ✓ $PY_VER"
    alias python3=python
else
    echo "  Python が見つかりません。Python 3.8 以上をインストールしてください。"
    exit 1
fi

# ── 3. Python パッケージのインストール ─────────────────────
echo ""
echo "[3/3] Python パッケージをインストール中..."

pip3 install -r requirements.txt

echo ""
echo "========================================"
echo "  セットアップ完了！"
echo "========================================"
echo ""
echo "使い方:"
echo "  python3 transcribe.py 動画ファイル.mp4"
echo ""
echo "オプション例:"
echo "  --engine whisper   # Whisper に切り替え（オフライン）"
echo "  --model medium     # Whisper モデルサイズ (tiny/base/small/medium/large)"
echo "  --lines 2          # テロップを 2 行表示（1 行最大 15 文字固定）"
echo "  --gap 0.3          # 文節ギャップ閾値（秒）"
echo ""
