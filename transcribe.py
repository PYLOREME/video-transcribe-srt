#!/usr/bin/env python3
"""
動画文字起こしシステム
====================
音声文字起こし（ElevenLabs / Whisper） → SRT生成

使い方:
  # ElevenLabs（推奨・高精度）
  python transcribe.py 動画.mp4 --api-key YOUR_KEY

  # API キーを環境変数で渡す場合
  export ELEVENLABS_API_KEY=YOUR_KEY
  python transcribe.py 動画.mp4

  # Whisper（ローカル・オフライン）
  python transcribe.py 動画.mp4 --engine whisper

オプション一覧:
  --api-key       ElevenLabs API キー (または環境変数 ELEVENLABS_API_KEY)
  --engine        transcription エンジン: elevenlabs / whisper (default: elevenlabs)
  --model         Whisper モデルサイズ (tiny/base/small/medium/large, default: large)
  --lines         テロップ 1 枚の行数: 1 または 2 (default: 1)
                  1 行の最大文字数は 15 文字固定
  --gap           自然なポーズとみなすギャップ秒数 (default: 0.4)
  --language      音声言語コード (default: ja)
"""

import sys
import os
import re
import subprocess
import argparse
import tempfile
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# .env ファイル読み込み
# ─────────────────────────────────────────────────────────────

def load_dotenv(dotenv_path: str = None):
    """
    .env ファイルを読み込んで環境変数にセットする。
    外部ライブラリ不要のシンプルな実装。
    dotenv_path を省略した場合はスクリプトと同じフォルダの .env を探す。
    """
    if dotenv_path is None:
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

    if not os.path.exists(dotenv_path):
        return  # .env がなくてもエラーにしない

    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 空行・コメント行をスキップ
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")   # クォートを除去
            # すでに環境変数にあればそちらを優先
            os.environ.setdefault(key, val)

# スクリプト起動時に自動ロード
load_dotenv()


# ─────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────

def format_time_srt(seconds: float) -> str:
    """秒を SRT タイムコード形式 (HH:MM:SS,mmm) に変換"""
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def run_ffmpeg(cmd: list, label: str = "") -> subprocess.CompletedProcess:
    """ffmpeg/ffprobe コマンドを実行して結果を返す"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and label:
        print(f"  [警告/{label}] {result.stderr[-300:]}")
    return result


def extract_audio(video_path: str, tmpdir: str) -> str:
    """
    動画から音声だけを抽出して mp3 として保存する。
    ElevenLabs API への送信ファイルサイズを小さくするため 16kHz・モノラルに変換。
    """
    audio_path = os.path.join(tmpdir, "audio.mp3")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",                       # 映像除外
        "-acodec", "libmp3lame",
        "-ar", "16000",              # 16kHz（音声認識に十分）
        "-ac", "1",                  # モノラル
        "-b:a", "64k",               # ファイルサイズ削減
        "-y", audio_path,
    ]
    run_ffmpeg(cmd, "extract_audio")
    return audio_path


# ─────────────────────────────────────────────────────────────
# ステップ 1a : ElevenLabs 文字起こし
# ─────────────────────────────────────────────────────────────

def transcribe_elevenlabs(filepath: str,
                           api_key: str,
                           language: str = "ja") -> dict:
    """
    ElevenLabs Speech-to-Text API で文字起こしを行う。
    動画ファイルは自動で音声 (mp3) に変換してから送信する。
    返り値: {"text": "...", "words": [...]}
    """
    try:
        import requests
    except ImportError:
        print("エラー: requests がインストールされていません。")
        print("  → pip install requests")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        print("  音声を抽出中...")
        audio_path = extract_audio(filepath, tmpdir)
        audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"  音声ファイル: {audio_size_mb:.1f} MB")

        print("  ElevenLabs API に送信中...")
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {"xi-api-key": api_key}
        data = {
            "model_id": "scribe_v1",
            "timestamps_granularity": "word",
        }
        if language:
            data["language_code"] = language

        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
            response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code != 200:
        print(f"  エラー: ElevenLabs API エラー ({response.status_code})")
        print(f"  {response.text}")
        sys.exit(1)

    result = response.json()
    word_count = len([w for w in result.get("words", []) if w.get("type") == "word"])
    print(f"  文字起こし完了: {word_count} 単語")
    return result


# ─────────────────────────────────────────────────────────────
# ステップ 1b : Whisper 文字起こし（オフライン）
# ─────────────────────────────────────────────────────────────

def transcribe_whisper(filepath: str,
                        model_name: str = "large",
                        language: str   = "ja") -> dict:
    """Whisper でファイルを文字起こし（word_timestamps=True）"""
    try:
        import whisper
    except ImportError:
        print("エラー: openai-whisper がインストールされていません。")
        print("  → pip install openai-whisper")
        sys.exit(1)

    print(f"  Whisper モデル '{model_name}' をロード中...")
    model = whisper.load_model(model_name)
    print("  文字起こし実行中...")
    result = model.transcribe(
        filepath,
        language=language,
        word_timestamps=True,
        verbose=False,
        fp16=False,
    )
    return result


# ─────────────────────────────────────────────────────────────
# ステップ 2 : テロップ分割（文節・意味区切り最適化）
# ─────────────────────────────────────────────────────────────

PUNCTUATION = set("。、！？\n")

# 1 行に入る最大文字数（固定）
MAX_CHARS_PER_LINE = 15


# ─────────────────────────────────────────────────────────────
# 日本語文節分割ルール
# 「怖い | から」「な | んだけど」「て | ねって」のような
# 助詞/助動詞を途中で分断する不自然な切り方が起きないよう、以下の 4 段構えで防ぐ。
# ─────────────────────────────────────────────────────────────

# ① NEVER_SPLIT_PAIRS
# 「前の字 + 次の字」がこの集合に入っている場所では絶対に改行/分割しない。
# 助詞連体、助動詞、促音・拗音の内側などを漏れなく守る。
NEVER_SPLIT_PAIRS = frozenset({
    # 接続助詞・複合助詞
    "から", "けど", "けれ", "まで", "など", "って", "たら", "たり", "れば",
    "とか", "とも",                                            # 「〜と | か」を防ぐ
    "ので", "のに", "のが", "のを", "のは", "のも", "のと", "のか", "のよ",
    "のね", "のだ", "のみ",
    # 「〜にする / にせず / にして」等のイディオム
    "にす", "にせ", "にし", "にさ", "にな",
    # よく分断される複合表現
    "もう",                                                     # 「みたらも | う」を防ぐ
    "すか",                                                     # 「ないです | か」「ます | か」を防ぐ
    "がと", "とう", "おう",                                     # 「ありが | とう」「おはよ | う」を防ぐ
    "とい", "いう",                                             # 「と | いう」「とい | う」を防ぐ
    "ごい", "しい", "たい",                                     # i 形容詞末端「すご | い」等を防ぐ
    "よね", "もね", "わね", "のね", "かね", "だね", "たね",     # 終助詞連鎖の内部保護
    "よな", "もな",
    "ても", "でも",                                             # 「して | も」「〜で | も」の内部保護
    "なな", "なき",                                             # 「死な | なきゃ」「〜な | き」の内部保護
    "やる", "やり", "やっ",                                     # 「や | る」「や | りたい」を防ぐ
    "てみ", "でみ",                                             # 「やって | みたら」を防ぐ
    "すご",                                                     # 「す | ごい」を防ぐ
    "とで",                                                     # 「ということ | で」を防ぐ
    "おじ", "おば", "おに", "おと", "おか",                     # 「お | じちゃん」等、親族/敬称接頭辞
    "いい", "こい",                                             # 「かっこ | いい」「かっこい | い」を防ぐ
    "はば",                                                     # 「は | ばかる」を防ぐ
    "てほ", "でほ",                                             # 「やって | ほしい」を防ぐ
    "と思", "と言", "と考", "と信", "と決", "と感",             # 「〜と | 思う/言う/…」を防ぐ
    # 「思う」「考える」「言う」等の漢字動詞語幹＋活用語尾の内部保護
    "思う", "思っ", "思わ", "思い", "思え", "思お",
    "考え", "考わ", "考ら", "考ろ",
    "言う", "言っ", "言わ", "言い", "言え", "言お",
    # 形式名詞・副助詞との境界
    "とす",                                                     # 「こと | すら」等
    "のこ",                                                     # 「〜 | のこと」
    # ん + X （なんだ / なんで / なんじゃ …）
    "んだ", "んで", "んじ", "んが", "んの", "んな", "んと", "んか",
    # な + X （なん- / なの / なが（ら）/ なる）
    "なん", "なの", "なが", "なけ", "なる",
    # だ + X （だと / だけ- / だが / だし / だっ / だな / だよ / だね / だから）
    "だと", "だけ", "だが", "だし", "だっ", "だな", "だよ", "だね", "だか",
    # で + X （でも / でき / でし / です / でな）
    "でも", "でき", "でし", "です", "でな",
    # です・ます関連
    "ます", "でし", "まし", "ませ", "です",
    # て/で + 終助詞 → 「変えて | ね」を防ぐ
    "てね", "でね", "てよ", "でよ", "てさ", "てわ", "でわ",
    # 連体形・様態・伝聞・比況の「よう / そう」→「言ったよ | うな」を防ぐ
    "よう", "そう",
    # 意向・希望・様子の助動詞語尾 → 「見た | い」「したく | ない」を防ぐ
    "たい", "たく", "たか", "ない", "なく", "なか", "しい", "しく",
    # 形式名詞 → 「〜こと」「〜とき」「〜はず」「〜わけ」「〜ため」を単体で守る
    "こと", "とき", "はず", "わけ", "ため", "もの", "ところ",
    # 副助詞 → 「〜ばかり」「〜くらい」「〜ぐらい」を内部で切らない
    "ばか", "くら", "ぐら",
    # 促音直後（「かっ | こいい」等を防ぐため hiragana 子音を網羅）
    "って", "った", "っと", "っち", "っぱ", "っし", "っか", "っき",
    "っく", "っす", "っさ", "っこ", "っけ", "っせ", "っそ", "っつ",
    "っぴ", "っぷ", "っぺ", "っぽ", "っは", "っひ", "っふ", "っへ",
    "っほ", "っま", "っみ", "っむ", "っめ", "っも",
    # 拗音（小書き ゃゅょ の直前は絶対に切らない）
    "ちゃ", "しゃ", "じゃ", "きゃ", "にゃ", "ひゃ", "みゃ", "りゃ", "ぎゃ", "びゃ", "ぴゃ",
    "ちゅ", "しゅ", "じゅ", "きゅ", "にゅ", "ひゅ", "みゅ", "りゅ", "ぎゅ", "びゅ", "ぴゅ",
    "ちょ", "しょ", "じょ", "きょ", "にょ", "ひょ", "みょ", "りょ", "ぎょ", "びょ", "ぴょ",
    # 動詞語尾（一段/使役/受身の直前）
    "せる", "れる", "える", "きる", "しる", "せた", "れた", "せて", "れて",
    "した", "して", "せず", "しな",
    # ある / いる
    "ある", "いる",
})

# ② NEVER_LINE_HEAD
# これらの文字は「行/テロップの先頭」に絶対に来られない。
# 小書き仮名・促音・撥音・長音・句読点はどれも直前の音節にぶら下がる。
# 「て」もほぼ確実に前文節の連用形/接続の続き。
# 「ね」「よ」は終助詞であり文頭に来るのはほぼ前文節の続き。
NEVER_LINE_HEAD = frozenset("ぁぃぅぇぉゃゅょっんー、。！？てねよ")

# ②-B NEVER_LINE_HEAD_PATTERNS
# 「先頭数文字がこのパターンで始まる」なら前文節の続き扱いで分割禁止。
# 「気に | せず」「〜 | って」「〜 | から」等を防ぐ。
NEVER_LINE_HEAD_PATTERNS = (
    # 助動詞・活用語尾（前文節の続き）
    "せず", "せぬ", "せる", "せて", "せた", "せよ",
    "して", "した", "しな", "しま", "しよ", "しろ",
    "れて", "れる", "れば", "れた", "れな",
    "って", "った", "っと", "っか", "った",
    "たら", "たり", "たい", "たく",
    "ない", "なく", "なか", "なけ",
    # 接続助詞（後続節へ強く結びつく）
    "など", "けど", "から", "ので", "のに", "まで", "ながら", "なので",
    "だから", "でも",
    # 動詞連用形の続き（「言って | らっしゃった」「〜って | いう」等）
    "らっ", "らし", "ちゃ", "しゃ",
    "いう", "いっ", "いる", "いて", "いた", "いま",
    "ませ", "まし", "です",                                    # 「〜ない | ですか」対策
    "なきゃ", "なけ", "なな",                                  # 「〜 | なきゃ」「〜死 | ななきゃ」対策
    # 定型挨拶の途中（「あり | がとう」「ありがと | うござい」等）
    "がと", "とう", "ざい", "ござ",
    # 動詞・形容詞の連用形/語幹の続き（「や | りたい」「あ | りがとう」「す | ごい」等）
    "りた", "りま", "りが", "りに", "りな", "りの", "りて",
    "ごい", "しぶ",
    "みた", "みて", "みる", "みよ", "みま",                    # 「やって | みたら」対策
    "じち",                                                    # 「お | じちゃん」対策
)

# ②-C FORBIDDEN_END_SUFFIXES
# これらで終わる位置は「テロップの末尾」に置いてはいけない。
# 「では」「でも」は談話マーカー扱いで、後続節に付ける方が自然。
FORBIDDEN_END_SUFFIXES = (
    "では",
)

# ③ NATURAL_END_SUFFIXES
# ここに入っている接尾辞で終わる位置は「自然な区切り」候補（強）。
# ※ 意図的に外したもの:
#     から / ので / のに / たら / れば / ながら / なので
#       → 後続節と強く結び付く原因/条件節。切ると「怖いから | 逃げ回って」の
#         ような不自然な orphan が生まれるので、区切り候補にしない。
#     しかし / そして / または
#       → 文頭の接続詞。「文末」候補としては不適切。
NATURAL_END_SUFFIXES = (
    # 4 文字
    "みたいな", "みたいに",
    # 3 文字（長い方から先にマッチさせる）
    "ましょう", "ましたか", "ですかね", "ますかね",
    "ました", "でした", "ません", "ですか", "ますか",
    "ですね", "ますね", "ですよ", "ますよ", "でしょ",
    "ような", "ように", "ようだ", "ようで",   # 連体形・比況
    "そうな", "そうに", "そうだ", "そうで",   # 様態・伝聞
    "たいな", "たいに",                        # 願望+終助詞
    # 2 文字
    "けど", "って", "だと", "など", "けれ", "とか",
    "ですが", "ますが",
    "ます", "です",
)

# ④ NATURAL_END_CHARS
# 単一文字でも「そこで区切ってよい」と判断できるもの。
# 助詞（は/が/を/に/へ/と/も/や）、終助詞（ね/よ/の）、句読点だけに絞る。
# ※ 意図的に外したもの:
#     て / で        → 「変えて | ねって」のような分断を起こす
#     か             → 「怖いか | ら」のような分断を起こす（NEVER_SPLIT_PAIRS の
#                       "から" で 2-gram 段階でも弾かれるが、二重に守る）
#     な             → 「アグリーな | んだけど」のような分断を起こす
#     き / り / し / ず → 動詞連用形の途中で切れるケースが多い
NATURAL_END_CHARS = frozenset("。？！、はがをにへともやのねよ")


def _flush(buf_words: list, key_text: str = "text") -> dict:
    """バッファの単語リストをサブタイトル辞書に変換する"""
    return {
        "start": buf_words[0]["start"],
        "end":   buf_words[-1]["end"],
        "text":  "".join(w[key_text] for w in buf_words).strip(),
    }


def _pair_forbidden(text: str, pos: int) -> bool:
    """text[pos-1:pos+1] が NEVER_SPLIT_PAIRS なら True"""
    if pos <= 0 or pos >= len(text):
        return True
    return (text[pos - 1] + text[pos]) in NEVER_SPLIT_PAIRS


def _head_forbidden(text: str, pos: int) -> bool:
    """text[pos] を行頭に置けない（単一文字 or 複数字パターン）なら True"""
    if pos <= 0 or pos >= len(text):
        return True
    if text[pos] in NEVER_LINE_HEAD:
        return True
    for pat in NEVER_LINE_HEAD_PATTERNS:
        if text[pos: pos + len(pat)] == pat:
            return True
    return False


def _tail_forbidden(text: str, pos: int) -> bool:
    """text[:pos] の末尾がテロップ末尾に置けないパターンなら True"""
    if pos <= 0 or pos > len(text):
        return True
    for suffix in FORBIDDEN_END_SUFFIXES:
        if pos >= len(suffix) and text[pos - len(suffix): pos] == suffix:
            return True
    return False


def _is_strong_split(text: str, pos: int) -> bool:
    """
    text を text[:pos] / text[pos:] に切る位置が「自然な区切り」かを判定する（強判定）。
    NEVER_SPLIT_PAIRS / NEVER_LINE_HEAD / FORBIDDEN_END_SUFFIXES を満たしたうえで、
    NATURAL_END_SUFFIXES または NATURAL_END_CHARS のいずれかにマッチする必要がある。
    """
    if pos <= 0 or pos >= len(text):
        return False
    if _pair_forbidden(text, pos) or _head_forbidden(text, pos) or _tail_forbidden(text, pos):
        return False
    # 長い接尾辞から順に確認
    for suffix in NATURAL_END_SUFFIXES:
        if pos >= len(suffix) and text[pos - len(suffix): pos] == suffix:
            return True
    return text[pos - 1] in NATURAL_END_CHARS


def _is_safe_split(text: str, pos: int) -> bool:
    """
    フォールバック用の弱判定。
    「絶対に切れてはいけない禁止条件」だけを満たせば OK とする。
    """
    if pos <= 0 or pos >= len(text):
        return False
    if _pair_forbidden(text, pos) or _head_forbidden(text, pos) or _tail_forbidden(text, pos):
        return False
    return True


def _find_best_split(buf_words: list, max_chars: int,
                      key_text: str = "text", min_chars: int = 4) -> int:
    """
    buf_words を 2 つに割るためのインデックス (buf[:i] と buf[i:] の i) を返す。

    戦略:
      1. [min_chars, max_chars] の範囲で「強判定」を満たす最後方の位置を採用する。
         → テロップ枠を可能な限り埋めつつ、自然な区切りで切る。
      2. 見つからなければ「弱判定（禁止 2-gram だけ回避）」で最後方の位置を採用する。
      3. それでもダメなら max_chars に収まる最後の単語境界に落とす。
    """
    if len(buf_words) <= 1:
        return len(buf_words)

    text = ""
    boundaries = []  # buf_words[:i+1] を連結したときの文字数
    for w in buf_words:
        text += w[key_text]
        boundaries.append(len(text))

    # 1) 強判定: 最も後ろの自然な区切りを探す
    best = None
    for i, char_pos in enumerate(boundaries):
        if char_pos > max_chars:
            break
        if char_pos < min_chars:
            continue
        if _is_strong_split(text, char_pos):
            best = i + 1
    if best is not None:
        return best

    # 2) 弱判定: 禁止条件だけ回避して最後方の単語境界
    for i in range(len(boundaries) - 1, -1, -1):
        char_pos = boundaries[i]
        if char_pos == 0 or char_pos > max_chars:
            continue
        if _is_safe_split(text, char_pos):
            return i + 1

    # 3) 最終フォールバック: max_chars 以内の最後方の単語境界
    for i, char_pos in enumerate(boundaries):
        if char_pos > max_chars:
            return i if i > 0 else 1

    return len(buf_words)


def _process_word_stream(words: list,
                          max_chars: int,
                          gap_threshold: float,
                          key_text: str = "text") -> list:
    """
    単語リスト（タイムスタンプ付き）からサブタイトルを生成する共通ロジック。

    処理順序:
      ① 単語間のギャップ >= gap_threshold 秒  → 自然なポーズ → 現バッファをフラッシュ
      ─ 単語をバッファに追加 ─
      ③ バッファが max_chars 文字を超えた      → バッファを遡って意味的区切りで分割
                                               （残りは次のバッファへ繰り越す）
      ② 句点・読点（。、！？）で終わる         → バッファをフラッシュ
      ※ ③ を ② より先にチェックすることで「長い文が句点で終わる場合も
         途中で意味的に分割してから句点で閉じる」動作を実現する
    """
    subtitles = []
    buf      = []      # {"text":..., "start":..., "end":...} のリスト
    last_end = None

    for word in words:
        w_text = word[key_text]
        w_end  = word["end"]

        if not w_text:
            continue

        gap = (word["start"] - last_end) if last_end is not None else 0.0

        # ─ ① 自然なポーズで分割（新しい単語を追加する前に）────
        if gap >= gap_threshold and buf:
            subtitles.append(_flush(buf, key_text))
            buf      = []
            last_end = None

        # 単語をバッファに追加
        buf.append(word)
        last_end = w_end

        buf_text = "".join(w[key_text] for w in buf)

        # ─ ③ max_chars 超過 → 意味的区切りで分割（while で繰り返す）
        # NEVER_LINE_HEAD_PATTERNS（最長 3 文字）を確実に照合するため、
        # 少し先読みバッファ（+3 文字）が揃うまで分割を遅延する。
        # 句読点で終わっているときは即分割してよい。
        split_slack = 3
        while len(buf) > 1 and len(buf_text) > max_chars and (
            len(buf_text) > max_chars + split_slack
            or buf_text[-1] in PUNCTUATION
        ):
            split_pos = _find_best_split(buf, max_chars, key_text)
            part1     = buf[:split_pos]
            buf       = buf[split_pos:]
            if part1:
                subtitles.append(_flush(part1, key_text))
            buf_text = "".join(w[key_text] for w in buf)

        # ─ ② 句点・読点で分割（超過対応の後にチェック）────────
        if buf_text and buf_text[-1] in PUNCTUATION:
            subtitles.append(_flush(buf, key_text))
            buf      = []
            last_end = None

    # 残りをフラッシュ（末尾に溜まったバッファも上限で分割してから出す）
    while len(buf) > 1 and len("".join(w[key_text] for w in buf)) > max_chars:
        split_pos = _find_best_split(buf, max_chars, key_text)
        part1     = buf[:split_pos]
        buf       = buf[split_pos:]
        if part1:
            subtitles.append(_flush(part1, key_text))
    if buf:
        subtitles.append(_flush(buf, key_text))

    return subtitles


# 単独テロップに残ってしまいがちな 1〜2 文字の助詞・接続詞
TINY_MERGE_CHARS = frozenset("でてはがをにへともやのねよかなさわぞ")


def _merge_tiny_subtitles(subtitles: list, max_chars_total: int) -> list:
    """
    1〜2 文字だけの助詞テロップ（「で」「て」「が」等）は
    直前のテロップ末尾に併合する。前が満杯なら次に併合する。
    """
    if not subtitles:
        return subtitles

    merged: list = []
    pending_prepend = None  # 次のテロップの先頭に付ける文字列

    for sub in subtitles:
        sub = dict(sub)
        if pending_prepend is not None:
            sub["text"] = pending_prepend["text"] + sub["text"]
            sub["start"] = pending_prepend["start"]
            pending_prepend = None

        text = sub["text"]
        # 読点・句点を除いた本体で助詞かどうかを判定する
        content = "".join(c for c in text if c not in "。、")
        is_tiny_particle = (
            1 <= len(content) <= 2
            and all(c in TINY_MERGE_CHARS for c in content)
        )

        if is_tiny_particle and merged:
            prev_text    = merged[-1]["text"]
            prev_content = "".join(c for c in prev_text if c not in "。、")
            # 句読点を除いた実文字数で判定（読点付きで満杯扱いにならないように）
            if len(prev_content) + len(content) <= max_chars_total:
                merged[-1]["text"] += text
                merged[-1]["end"]   = sub["end"]
                continue
            # 前が満杯 → 次のテロップに繰り越して先頭に付ける
            pending_prepend = sub
            continue

        merged.append(sub)

    # 最後に持ち越しが残っていたら独立テロップとして戻す
    if pending_prepend is not None:
        merged.append(pending_prepend)

    return merged


def _line_head_ok(text: str) -> bool:
    """text が新しいテロップの先頭として来られるかを判定する"""
    if not text:
        return True
    if text[0] in NEVER_LINE_HEAD:
        return False
    for pat in NEVER_LINE_HEAD_PATTERNS:
        if text.startswith(pat):
            return False
    return True


def _resolve_forbidden_pair_boundaries(subtitles: list, max_chars_total: int) -> list:
    """
    テロップ N の末字 + テロップ N+1 の先頭字が NEVER_SPLIT_PAIRS に該当する境界を
    以下の優先順で解消する。
      1) 合計が上限内なら結合
      2) 末尾 1 文字を次の先頭に送る（新境界が safe になるなら）
      3) 先頭 1 文字を前の末尾に持ってくる（新境界が safe になるなら）
    """
    if not subtitles:
        return subtitles
    subs = [dict(s) for s in subtitles]

    def core(text):
        stripped = text.rstrip("。、")
        head_stripped = stripped.lstrip("。、")
        return head_stripped

    i = 0
    max_iter = len(subs) * 3  # 無限ループ防止
    steps = 0
    while i < len(subs) - 1 and steps < max_iter:
        steps += 1
        prev = subs[i]
        curr = subs[i + 1]
        prev_text = prev["text"]
        curr_text = curr["text"]
        prev_head_trim = prev_text.lstrip("。、")
        prev_core     = prev_head_trim.rstrip("。、")
        prev_lead     = prev_text[: len(prev_text) - len(prev_head_trim)]
        prev_trail    = prev_head_trim[len(prev_core):]
        curr_head_trim = curr_text.lstrip("。、")
        curr_core     = curr_head_trim.rstrip("。、")
        curr_lead     = curr_text[: len(curr_text) - len(curr_head_trim)]
        curr_trail    = curr_head_trim[len(curr_core):]

        if not prev_core or not curr_core:
            i += 1
            continue

        pair = prev_core[-1] + curr_core[0]
        if pair not in NEVER_SPLIT_PAIRS:
            i += 1
            continue

        # 1) 結合
        if len(prev_core) + len(curr_core) <= max_chars_total:
            prev["text"] = prev_text + curr_text
            prev["end"]  = curr["end"]
            subs.pop(i + 1)
            # 同じ i を再チェック（結合後にさらに次との境界を見る）
            continue

        # 2) 末尾 k 文字を次に送る（1..min(prev-1, 3) まで試す）
        applied = False
        for k in range(1, min(len(prev_core), 4)):
            if len(curr_core) + k > max_chars_total:
                break
            moved         = prev_core[-k:]
            new_prev_core = prev_core[:-k]
            if not new_prev_core:
                break
            new_curr_core = moved + curr_core
            new_pair      = new_prev_core[-1] + moved[0]
            if new_pair in NEVER_SPLIT_PAIRS:
                continue
            if not _line_head_ok(new_curr_core):
                continue
            prev["text"] = prev_lead + new_prev_core + prev_trail
            curr["text"] = curr_lead + new_curr_core + curr_trail
            applied = True
            break
        if applied:
            if i > 0:
                i -= 1
            continue

        # 3) 先頭 k 文字を前に持ってくる
        for k in range(1, min(len(curr_core), 4)):
            if len(prev_core) + k > max_chars_total:
                break
            moved         = curr_core[:k]
            new_curr_core = curr_core[k:]
            if not new_curr_core:
                break
            new_prev_core = prev_core + moved
            new_pair      = moved[-1] + new_curr_core[0]
            if new_pair in NEVER_SPLIT_PAIRS:
                continue
            if not _line_head_ok(new_curr_core):
                continue
            prev["text"] = prev_lead + new_prev_core + prev_trail
            curr["text"] = curr_lead + new_curr_core + curr_trail
            applied = True
            break
        if applied:
            if i > 0:
                i -= 1
            continue

        # どれもダメ → そのまま
        i += 1

    return subs


# 後方互換: 旧名からも呼べるようにする
def _merge_forbidden_pair_boundaries(subtitles: list, max_chars_total: int) -> list:
    return _resolve_forbidden_pair_boundaries(subtitles, max_chars_total)


def _shift_forbidden_end(subtitles: list) -> list:
    """
    テロップ末尾が FORBIDDEN_END_SUFFIXES（例: 「では」）で終わっている場合、
    その接尾辞を次テロップの先頭に移動する。
    句読点区切りでフラッシュされた場合の救済策。
    """
    if not subtitles:
        return subtitles
    out = [dict(s) for s in subtitles]
    for i in range(len(out) - 1):
        text = out[i]["text"]
        core = text.rstrip("。、！？")
        matched = None
        for suffix in FORBIDDEN_END_SUFFIXES:
            if core.endswith(suffix):
                matched = suffix
                break
        if matched:
            core_len = len(core)
            trailing_punct = text[core_len:]
            out[i]["text"]   = text[: core_len - len(matched)] + trailing_punct
            out[i + 1]["text"] = matched + out[i + 1]["text"]
    return [s for s in out if s["text"].strip("。、 ")]


def build_subtitles_elevenlabs(el_result: dict,
                                max_chars: int       = 20,
                                gap_threshold: float = 0.4) -> list:
    """ElevenLabs の文字起こし結果からサブタイトルを生成する"""
    raw_words = [
        {"text": w["text"], "start": w["start"], "end": w["end"]}
        for w in el_result.get("words", [])
        if w.get("type") == "word" and w.get("text", "").strip()
    ]
    if not raw_words:
        return []
    subs = _process_word_stream(raw_words, max_chars, gap_threshold, key_text="text")
    subs = _shift_forbidden_end(subs)
    subs = _merge_forbidden_pair_boundaries(subs, max_chars)
    return _merge_tiny_subtitles(subs, max_chars)


def build_subtitles_whisper(whisper_result: dict,
                             max_chars: int       = 20,
                             gap_threshold: float = 0.4) -> list:
    """Whisper の文字起こし結果からサブタイトルを生成する"""
    subtitles = []

    for seg in whisper_result.get("segments", []):
        text      = seg["text"].strip()
        seg_start = seg["start"]
        seg_end   = seg["end"]
        words     = seg.get("words", [])

        if not text:
            continue

        if words:
            # 単語レベルのタイムスタンプがある場合
            raw_words = [
                {"text": w.get("word", "").strip(),
                 "start": w.get("start", seg_start),
                 "end":   w.get("end",   seg_end)}
                for w in words
                if w.get("word", "").strip()
            ]
            seg_subs = _process_word_stream(raw_words, max_chars, gap_threshold, key_text="text")
            seg_subs = _shift_forbidden_end(seg_subs)
            seg_subs = _merge_forbidden_pair_boundaries(seg_subs, max_chars)
            subtitles.extend(_merge_tiny_subtitles(seg_subs, max_chars))
        else:
            # セグメントレベルのみ → 文字数に応じて線形補間
            parts     = _split_text_fallback(text, max_chars)
            total_len = sum(len(p) for p in parts) or 1
            duration  = seg_end - seg_start
            cur_time  = seg_start
            for part in parts:
                part_dur = (len(part) / total_len) * duration
                subtitles.append({"start": cur_time, "end": cur_time + part_dur, "text": part})
                cur_time += part_dur

    return subtitles


def _find_best_split_in_text(text: str, max_chars: int, min_chars: int = 4) -> int:
    """
    text 内で「max_chars 以内かつ自然な区切り」となる最も後ろの分割位置を返す。
    _find_best_split の文字列版（単語境界情報がない場合に使う）。
    """
    if len(text) <= max_chars:
        return len(text)

    upper = min(max_chars, len(text) - 1)

    # 1) 強判定
    for pos in range(upper, min_chars - 1, -1):
        if _is_strong_split(text, pos):
            return pos

    # 2) 弱判定（禁止条件だけ回避）
    for pos in range(upper, 0, -1):
        if _is_safe_split(text, pos):
            return pos

    # 3) 最終フォールバック
    return max_chars


def _split_text_fallback(text: str, max_chars: int) -> list:
    """
    タイムスタンプ情報がない場合のフォールバック分割。
    句点・読点で予備分割 → 自然な区切りで再分割する。
    """
    parts     = []
    raw_parts = re.split(r"(?<=[。、！？])", text)

    for seg in raw_parts:
        seg = seg.strip()
        if not seg:
            continue

        while len(seg) > max_chars:
            cut = _find_best_split_in_text(seg, max_chars)
            parts.append(seg[:cut])
            seg = seg[cut:].strip()

        if seg:
            parts.append(seg)

    return parts if parts else [text]


# ─────────────────────────────────────────────────────────────
# ステップ 3 : SRT ファイル出力
# ─────────────────────────────────────────────────────────────

def _wrap_two_lines(text: str, max_chars_per_line: int = MAX_CHARS_PER_LINE) -> str:
    """
    テキストを最大 2 行・各行 max_chars_per_line 以内に折り返す。
    _find_best_split_in_text と同じルール（NEVER_SPLIT_PAIRS / NEVER_LINE_HEAD /
    NATURAL_END_SUFFIXES / NATURAL_END_CHARS）で改行位置を決定する。
    """
    text = text.strip()
    if len(text) <= max_chars_per_line:
        return text

    pos = _find_best_split_in_text(text, max_chars_per_line, min_chars=1)
    line1 = text[:pos]
    line2 = text[pos:]
    return f"{line1}\n{line2}"


# 出力から除去する句読点（分割判定には引き続き使用する）
STRIP_PUNCT = str.maketrans("", "", "。、")


def write_srt(subtitles: list, output_path: str, lines: int = 1):
    """SRT ファイルを書き出す (lines=2 の場合は 2 行に折り返す)"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sub in enumerate(subtitles, 1):
            text = sub["text"]
            # 2 行折り返し（句読点を分割ヒントとして使うため、除去より前に実行）
            if lines == 2:
                text = _wrap_two_lines(text)
            # 句読点（。、）を出力から除去
            text = text.translate(STRIP_PUNCT)
            f.write(f"{i}\n")
            f.write(f"{format_time_srt(sub['start'])} --> {format_time_srt(sub['end'])}\n")
            f.write(f"{text}\n\n")
    print(f"  {len(subtitles)} 件のテロップを書き出しました → {output_path}")


# ─────────────────────────────────────────────────────────────
# メイン処理
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="動画文字起こしシステム: 文字起こし → SRT生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python transcribe.py video.mp4 --api-key YOUR_KEY
  python transcribe.py video.mp4 --engine whisper --model medium
  python transcribe.py video.mp4 --api-key YOUR_KEY --gap 0.3 --lines 2

テロップギャップ調整の目安:
  --gap 0.3  : 0.3秒以上の間を文節境界とみなす（分割多め）
  --gap 0.4  : デフォルト（バランス型）
  --gap 0.6  : 0.6秒以上の間のみ分割（分割少なめ）

テロップ行数:
  --lines 1  : 1 行テロップ（最大 15 文字 / 枚）
  --lines 2  : 2 行テロップ（各行最大 15 文字 = 最大 30 文字 / 枚）
        """,
    )
    parser.add_argument("input",
                        help="入力動画ファイル (例: video.mp4)")
    parser.add_argument("--api-key",     default=None,
                        help="ElevenLabs API キー (または環境変数 ELEVENLABS_API_KEY)")
    parser.add_argument("--engine",      default="elevenlabs",
                        choices=["elevenlabs", "whisper"],
                        help="文字起こしエンジン (default: elevenlabs)")
    parser.add_argument("--model",       default="large",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper モデルサイズ (default: large)")
    parser.add_argument("--lines",       default=1, type=int,
                        choices=[1, 2],
                        help="テロップ 1 枚の行数: 1 または 2 (default: 1)")
    parser.add_argument("--gap",         default=0.4, type=float,
                        help="文節区切りとみなすギャップ秒数 (default: 0.4)")
    parser.add_argument("--language",    default="ja",
                        help="音声言語コード (default: ja)")
    args = parser.parse_args()

    # ── テロップ 1 枚あたりの合計最大文字数 ────────────────────
    max_chars_total = MAX_CHARS_PER_LINE * args.lines

    # ── API キーの解決 ────────────────────────────────────────
    api_key = args.api_key or os.environ.get("ELEVENLABS_API_KEY")
    if args.engine == "elevenlabs" and not api_key:
        print("エラー: ElevenLabs API キーが必要です。")
        print("  --api-key YOUR_KEY  または")
        print("  export ELEVENLABS_API_KEY=YOUR_KEY")
        sys.exit(1)

    # ── 入力ファイル確認 ───────────────────────────────────────
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"エラー: ファイルが見つかりません → {input_path}")
        sys.exit(1)

    stem     = input_path.stem
    out_dir  = input_path.parent
    srt_path = out_dir / f"{stem}.srt"

    BAR = "=" * 60
    print(f"\n{BAR}")
    print("  動画文字起こしシステム")
    print(BAR)
    print(f"  入力ファイル    : {input_path.name}")
    print(f"  エンジン        : {args.engine}")
    print(f"  テロップ行数    : {args.lines} 行")
    print(f"  最大文字数      : {MAX_CHARS_PER_LINE} 文字/行 (合計 {max_chars_total} 文字)")
    print(f"  文節ギャップ閾値: {args.gap} 秒")
    print(BAR)

    # ── Step 1 : 文字起こし ────────────────────────────────────
    if args.engine == "elevenlabs":
        print(f"\n[1/2] 文字起こし中 (ElevenLabs)...")
        result   = transcribe_elevenlabs(str(input_path), api_key, args.language)
        subtitles = build_subtitles_elevenlabs(result, max_chars_total, args.gap)
    else:
        print(f"\n[1/2] 文字起こし中 (Whisper / {args.model})...")
        result   = transcribe_whisper(str(input_path), args.model, args.language)
        subtitles = build_subtitles_whisper(result, max_chars_total, args.gap)

    # ── Step 2 : SRT 生成 ──────────────────────────────────────
    print(f"\n[2/2] SRT ファイルを生成中... → {srt_path.name}")
    write_srt(subtitles, str(srt_path), lines=args.lines)

    # ── 完了 ───────────────────────────────────────────────────
    print(f"\n{BAR}")
    print("  完了！")
    print(f"  SRT ファイル : {srt_path}")
    print(BAR + "\n")


if __name__ == "__main__":
    main()
