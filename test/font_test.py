import streamlit as st

st.set_page_config(page_title="フォント一覧テスト", layout="wide")

# 🔤 試したいフォント一覧
font_list = [
    "Noto Sans JP",
    "M PLUS Rounded 1c",
    "Kosugi Maru",
    "Meiryo",
    "Roboto",
    "Roboto Rounded",
    "Arial",
    "Helvetica",
    "monospace",
    "sans-serif",
]

# 🎛️ サイドバー設定
selected_font = st.sidebar.selectbox("📑 フォントを選んでください", font_list)
font_size = st.sidebar.slider("🔠 フォントサイズ (px)", 12, 30, 18)
bold = st.sidebar.checkbox("🅱️ 太字にする")
bg_mode = st.sidebar.radio("🎨 背景モード", ["Dark", "Light"])

# 🎨 スタイル設定
font_weight = "bold" if bold else "normal"
bg_color = "#222" if bg_mode == "Dark" else "#f4f4f4"
text_color = "#ffffff" if bg_mode == "Dark" else "#222"

# 🌐 Google Fonts 読み込み
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP&family=Roboto&display=swap" rel="stylesheet">
""",
    unsafe_allow_html=True,
)

# 💄 グローバルCSS適用（タイトル等にも影響）
st.markdown(
    f"""
    <style>
    html, body, h1, h2, h3, h4, h5, h6, p, span, div {{
        font-family: '{selected_font}', sans-serif !important;
    }}
    .custom-font {{
        font-family: '{selected_font}', sans-serif !important;
        font-size: {font_size}px;
        font-weight: {font_weight};
        line-height: 1.7;
        color: {text_color};
        background-color: {bg_color};
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }}
    </style>
""",
    unsafe_allow_html=True,
)

# ✅ タイトル
st.markdown(
    f"<h1 style='color:#00BFFF;'>📘 フォント確認モード：{selected_font}</h1>",
    unsafe_allow_html=True,
)

# 📄 表示サンプル（Kouさんのトップページ文）
st.markdown(
    """
<div class="custom-font">

### WEB版 参謀くんへようこそ！

このアプリは、**現場の業務効率化**と**帳票作成の自動化**を目的とした、社内専用の業務支援ツールです。  
主に、以下のような機能を提供しています：

### 🛠 管理業務
- 工場日報の出力  
- 工場搬出入収支表の集計・出力  
- 管理票の自動生成  

### 📈 機能１（今後追加予定）
- 品目別の統計グラフ表示  
- AIによる異常検知アラート  

### ⚙️ 機能２（今後追加予定）
- ユーザーごとの履歴管理  
- ダッシュボード機能の追加

### 👇 ご利用方法
左の **サイドバー** から使用したい機能を選んでください。  
各機能の画面では、必要なCSVファイルをアップロードすると、所定のフォーマットでExcelが出力されます。

### 💡 サポート
操作に不明点がある場合は、サイドバー下部の「📄 マニュアル・操作説明」欄をご確認ください。  
それでも解決しない場合は、社内のサポートチームまでご連絡ください。

※ このアプリは定期的に機能追加・改善が行われます。お知らせ欄も随時ご確認ください。

</div>
""",
    unsafe_allow_html=True,
)
