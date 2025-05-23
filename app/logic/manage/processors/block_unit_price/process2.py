import streamlit as st
import pandas as pd


def confirm_transport_selection(df_after: pd.DataFrame) -> None:
    """運搬業者の選択内容を確認するダイアログを表示する

    処理の流れ:
        1. 選択された運搬業者の一覧を表示
        2. 確認用のYes/Noボタンを表示
        3. ユーザーの選択に応じて処理を分岐
            - Yes: 次のステップへ進む（process_mini_step = 2）
            - No: Step1（選択画面）に戻る（process_mini_step = 1）

    Args:
        df_after (pd.DataFrame): 運搬業者が選択された出荷データ
    """
    # セッション状態の初期化
    if "transport_selection_confirmed" not in st.session_state:
        st.session_state.transport_selection_confirmed = False

    def _create_confirmation_view(df: pd.DataFrame) -> pd.DataFrame:
        """確認用の表示データを作成"""
        filtered_df = df[df["運搬業者"].notna()]
        return filtered_df[["業者名", "品名", "明細備考", "運搬業者"]]

    def _show_confirmation_buttons() -> tuple[bool, bool]:
        """確認用のYes/Noボタンを表示"""
        st.write("この運搬業者選択で確定しますか？")
        col1, col2 = st.columns([1, 1])

        with col1:
            yes_clicked = st.button("✅ はい（この内容で確定）", key="yes_button")
        with col2:
            no_clicked = st.button("🔁 いいえ（やり直す）", key="no_button")

        return yes_clicked, no_clicked

    def _handle_user_selection(yes_clicked: bool, no_clicked: bool) -> None:
        """ユーザーの選択結果を処理"""
        if yes_clicked:
            st.success("✅ 確定されました。次に進みます。")
            st.session_state.transport_selection_confirmed = True
            st.session_state.process_mini_step = 2
            st.rerun()

        if no_clicked:
            st.warning("🔁 選択をやり直します（Step1に戻ります）")
            st.session_state.transport_selection_confirmed = False
            st.session_state.process_mini_step = 1
            st.rerun()

    # すでに確認済みの場合はスキップ
    if st.session_state.transport_selection_confirmed:
        return

    # メイン処理の実行
    st.title("運搬業者の確認")

    # 1. 確認用データの表示
    df_view = _create_confirmation_view(df_after)
    st.dataframe(df_view)

    # 2. 確認ボタンの表示と選択結果の取得
    yes_clicked, no_clicked = _show_confirmation_buttons()

    # 3. 選択結果の処理
    _handle_user_selection(yes_clicked, no_clicked)

    # 4. ユーザーの操作待ち
    st.stop()
