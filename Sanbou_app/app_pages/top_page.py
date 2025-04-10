import streamlit as st
import time
import random


# タイプライター風チャット（プレーンテキスト用）
def typewriter_chat(message: str, delay=0.03):
    placeholder = st.empty()
    displayed = ""
    for char in message:
        displayed += char
        placeholder.markdown(displayed)
        time.sleep(delay)


# メイントップページ表示
def show_top_page():
    st.title("💬 参謀くん Chat Guide")

    # チャット本文（初回のみタイプライターで出す）
    chat = [
        "こんにちは！私は **WEB版 参謀くん** です 🧠✨",
        "このツールは、**現場業務の効率化**と**帳票作成の自動化**をサポートする、社内専用の業務支援アプリです。",
        "現在ご利用いただける機能は以下のとおりです：\n\n- 工場日報の作成\n- 工場搬出入収支表の集計\n- 管理票の自動生成",
        "ご利用の際は、👈 左側の **サイドバー** から出力したい項目を選び、対象のCSVファイルをアップロードしてください。",
        # "📄 アップロードが完了すると、所定のフォーマットに沿って **Excelファイルが自動で生成** されます。"
    ]

    # 初回表示フラグ（セッションで管理）
    if "top_page_viewed" not in st.session_state:
        st.session_state.top_page_viewed = False

    if not st.session_state.top_page_viewed:
        # ✅ 初回はアニメーション付きで1つずつ表示（自然な間隔）
        for msg in chat:
            with st.chat_message("assistant"):
                typewriter_chat(msg)
            time.sleep(random.uniform(0.4, 0.6))  # 吹き出し間隔にゆらぎ

        # ✅ サイドバー誘導チャット（typewriter）
        sidebar_msg = (
            "では、左の **サイドバー** にあるメニューから  \n"
            "出力したい帳票を選んでみてくださいね。CSVファイルのアップロードもそちらから行えます！  \n"
            "何を選べばいいか迷ったら、操作マニュアルも見てみてください📄"
        )
        with st.chat_message("assistant"):
            typewriter_chat(sidebar_msg)
        time.sleep(random.uniform(0.2, 0.6))

        # ✅ サポート案内リンク（HTMLは一括表示！）
        with st.chat_message("assistant"):
            st.markdown("""
                <div style="font-size: 16px;">
                💬 困ったときはこちらをご確認ください👇<br><br>
                📄 <a href="https://your-manual-link.com" target="_blank">操作マニュアルを見る</a><br>
                📧 <a href="mailto:support@example.com">サポートチームにメールする</a>
                </div>
            """, unsafe_allow_html=True)

        st.session_state.top_page_viewed = True  # 一度表示したら次回は一括へ

    else:
        # ✅ 2回目以降は全メッセージを一括表示（サクサク）
        for msg in chat:
            with st.chat_message("assistant"):
                st.markdown(msg)

        with st.chat_message("assistant"):
            st.markdown("""
                ✅ では、左の **サイドバー** にあるメニューから  
                出力したい帳票を選んでみてくださいね。CSVファイルのアップロードもそちらから行えます！  
                何を選べばいいか迷ったら、操作マニュアルも見てみてください📄
            """)

        with st.chat_message("assistant"):
            st.markdown("""
                <div style="font-size: 16px;">
                💬 困ったときはこちらをご確認ください👇<br><br>
                📄 <a href="https://your-manual-link.com" target="_blank">操作マニュアルを見る</a><br>
                📧 <a href="mailto:support@example.com">サポートチームにメールする</a>
                </div>
            """, unsafe_allow_html=True)
