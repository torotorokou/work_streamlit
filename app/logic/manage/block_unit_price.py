from utils.logger import app_logger
import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.utils.load_template import load_master_and_template
from config.loader.main_path import MainPath
from logic.readers.read_transport_discount import ReadTransportDiscount
import streamlit as st

import time

# デバッグ用
from utils.debug_tools import debug_pause


def process(dfs):
    import streamlit as st

    logger = app_logger()

    # --- 内部ミニステップ管理 ---
    mini_step = st.session_state.get("process_mini_step", 0)

    # --- テンプレート設定の取得 ---
    template_key = "block_unit_price"
    template_config = get_template_config()[template_key]
    template_name = template_config["key"]
    csv_keys = template_config["required_files"]
    logger.info(f"[テンプレート設定読込] key={template_key}, files={csv_keys}")

    # --- コンフィグとマスター読み込み ---
    config = get_template_config()["block_unit_price"]
    master_path = config["master_csv_path"]["vendor_code"]
    master_csv = load_master_and_template(master_path)

    # 運搬費の読込
    mainpath = MainPath()
    reader = ReadTransportDiscount(mainpath)
    df_transport = reader.load_discounted_df()

    # --- CSVの読み込み ---
    df_dict = load_all_filtered_dataframes(dfs, csv_keys, template_name)
    df_shipping = df_dict.get("shipping")

    # 各処理の実行
    if mini_step == 0:
        logger.info("▶️ Step0: フィルタリング・単価追加・固定運搬費")
        df_shipping = make_df_shipping_after_use(master_csv, df_shipping)
        df_shipping = apply_unit_price_addition(master_csv, df_shipping)
        df_shipping = process1(df_shipping, df_transport)
        st.session_state.df_shipping_first = df_shipping
        st.session_state.process_mini_step = 1
        st.rerun()
        return None

    elif mini_step == 1:
        logger.info("▶️ Step1: 選択式運搬費（process2）")
        df_after = st.session_state.df_shipping_first
        if not st.session_state.get("block_unit_price_confirmed", False):
            df_after = process2(df_after, df_transport)
            st.session_state.df_shipping = df_after
            st.rerun()
            return None
        else:
            logger.info("▶️ 選択済みなのでスキップ")
            st.session_state.process_mini_step = 2
            st.rerun()
            return None

    elif mini_step == 2:
        logger.info("▶️ Step2: 加算処理実行中")
        df_after = st.session_state.df_shipping

        # YESNOの選択
        yes_no_box(df_after)

        # 運搬費の計算
        df_after = process3(df_after, df_transport)
        df_after = process4(df_after, df_transport)

        # ブロック単価の計算
        df_after = process5(df_after)

        # 整形・セル記入欄追加
        df_after = eksc(df_after)

        # セル記入用df作成
        master_csv = ekuserubunkai(df_after)

        # 合計行の追加
        master_csv = goukei(master_csv, df_shipping)

        # ステートの初期化
        st.session_state.process_mini_step = 0

    return master_csv


def make_df_shipping_after_use(master_csv, df_shipping):
    # --- 業者CDでフィルタ ---
    df_after = df_shipping[df_shipping["業者CD"].isin(master_csv["業者CD"])].copy()

    # --- 品名指定があるものをマージしてフィルタリング ---
    item_filter_df = master_csv[master_csv["品名"].notna()][
        ["業者CD", "品名"]
    ].drop_duplicates()

    # 丸源処理。品名でソートする
    if not item_filter_df.empty:
        # 「業者CDと品名のペア」が一致する行だけ残す（外積フィルタ）
        df_after = df_after.merge(
            item_filter_df, on=["業者CD", "品名"], how="left", indicator=True
        )
        df_after = df_after[
            (df_after["_merge"] == "both")
            | (~df_after["業者CD"].isin(item_filter_df["業者CD"]))
        ]
        df_after = df_after.drop(columns=["_merge"])

    # 正味重量が0を除外
    df_after = df_after[df_after["正味重量"].fillna(0) != 0]

    # 運搬費をmaster_csvから追加
    # 業者CDごとに1件に絞ってからマージ
    unique_master = master_csv.drop_duplicates(subset=["業者CD"])[
        ["業者CD", "運搬社数"]
    ]
    df_after = df_after.merge(unique_master, on="業者CD", how="left")

    # 運搬費カラムを作成
    df_after["運搬費"] = 0

    # 業者CDで並び替え
    df_after = df_after.sort_values(by="業者CD").reset_index(drop=True)

    return df_after


def apply_unit_price_addition(master_csv, df_shipping: pd.DataFrame) -> pd.DataFrame:
    """
    出荷データ（df）に対して、手数料情報を業者CD単位でマスターと照合し、
    対象業者の単価に加算を行う処理。
    """
    from logic.manage.utils.column_utils import apply_column_addition_by_keys

    # --- 単価への手数料処理（業者CDで結合） ---
    df_after = apply_column_addition_by_keys(
        base_df=df_shipping,
        addition_df=master_csv,
        join_keys=["業者CD"],
        value_col_to_add="手数料",
        update_target_col="単価",
    )

    return df_after


def process1(df_shipping, df_transport):
    from logic.manage.utils.column_utils import apply_column_addition_by_keys

    # --- ① 運搬社数 = 1 の行だけを抽出（対象行）
    target_rows = df_shipping[df_shipping["運搬社数"] == 1].copy()

    # --- ② 加算処理を適用
    updated_target_rows = apply_column_addition_by_keys(
        base_df=target_rows,
        addition_df=df_transport,
        join_keys=["業者CD"],
        value_col_to_add="運搬費",
        update_target_col="運搬費",
    )

    # --- ③ 運搬社数 != 1 の行をそのまま残す（非対象行）
    other_rows = df_shipping[df_shipping["運搬社数"] != 1].copy()

    # --- ④ 両方を結合（行順は変更される可能性あり）
    df_after = pd.concat([updated_target_rows, other_rows], ignore_index=True)

    # 業者CDで並び替え
    df_after = df_after.sort_values(by="業者CD").reset_index(drop=True)

    return df_after


def process2(df_after, df_transport):
    import streamlit as st
    import pandas as pd
    import re

    target_rows = df_after[df_after["運搬社数"] != 1].copy()

    if "block_unit_price_confirmed" not in st.session_state:
        st.session_state.block_unit_price_confirmed = False
    if "block_unit_price_transport_map" not in st.session_state:
        st.session_state.block_unit_price_transport_map = {}

    st.title("運搬業者の選択")

    st.markdown(
        """
    <style>
    h3 {
        border: none !important;
        margin-bottom: 0.5rem !important;
    }

    /* ✅ selectbox 白黒両対応 */
    div[data-baseweb="select"] > div {
        border-width: 1.5px !important;
        border-color: #999999 !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }

    div[data-baseweb="select"]:focus-within {
        box-shadow: 0 0 0 2px #cbd5e1 !important;
    }

    div[data-baseweb="select"] span {
        color: #f1f5f9 !important;
        font-weight: 600;
    }

    /* ✅ ラベルの色も調整（明暗両方見やすく） */
    label[data-testid="stWidgetLabel"] {
        color: #e5e7eb !important;
        font-size: 14px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    if not st.session_state.block_unit_price_confirmed:
        with st.form("transport_selection_form"):
            selected_map = {}

            for idx, row in target_rows.iterrows():
                gyousha_cd = row["業者CD"]
                gyousha_name = str(row.get("業者名", gyousha_cd))
                hinmei = str(row.get("品名", "")).strip()
                meisai = str(row.get("明細備考", "")).strip()

                gyousha_name_clean = re.sub(r"（\s*\d+\s*）", "", gyousha_name)
                hinmei_display = hinmei if hinmei else "-"
                meisai_display = meisai if meisai else "-"

                options = df_transport[df_transport["業者CD"] == gyousha_cd][
                    "運搬業者"
                ].tolist()
                if not options:
                    st.warning(
                        f"{gyousha_name_clean} に対応する運搬業者が見つかりません。"
                    )
                    continue

                select_key = f"select_block_unit_price_row_{idx}"
                if select_key not in st.session_state:
                    st.session_state[select_key] = options[0]

                st.markdown(
                    f"""
                    <div style='
                        background-color:#1e293b;
                        padding:1px 4px;
                        margin-bottom:6px;
                        border-radius:2px;
                        border:0.3px solid #3b4252;
                    '>
                    """,
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns([2, 3])

                with col1:
                    st.markdown(
                        f"""
                        <div style='padding-right:10px;'>
                            <div style='
                                font-size:18px;
                                font-weight:600;
                                color:#38bdf8;
                            '>
                                🗑️ {gyousha_name_clean}
                            </div>
                            <div style='
                                font-size:15px;
                                color:inherit;
                                margin-top: 2px;
                            '>
                                品名：{hinmei_display}
                            </div>
                            <div style='
                                font-size:14.5px;
                                color:inherit;
                                margin-top: 2px;
                            '>
                                明細備考：{meisai_display}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ✅ selectbox ラベルの色を事前に定義しておく
                st.markdown(
                    """
                <style>
                label[data-testid="stWidgetLabel"] {
                    color: #555555 !important;
                    font-size: 14.5px;
                }
                </style>
                """,
                    unsafe_allow_html=True,
                )

                # ✅ その後に selectbox を通常通り書く
                with col2:
                    selected = st.selectbox(
                        label="🚚 運搬業者を選択してください",
                        options=options,
                        key=select_key,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

                selected_map[idx] = selected

            submitted = st.form_submit_button("✅ 選択を確定して次へ進む")
            if submitted:
                if len(selected_map) < len(target_rows):
                    st.warning("未選択の行があります。すべての行を選択してください。")
                else:
                    st.session_state.block_unit_price_transport_map = selected_map
                    st.session_state.block_unit_price_confirmed = True
                    selected_df = pd.DataFrame.from_dict(
                        st.session_state.block_unit_price_transport_map,
                        orient="index",
                        columns=["運搬業者"],
                    )
                    selected_df.index.name = df_after.index.name
                    df_after = df_after.merge(
                        selected_df, how="left", left_index=True, right_index=True
                    )
                    st.success("✅ 選択が確定されました。")
                    return df_after

        st.stop()

    return df_after


def yes_no_box(df_after: pd.DataFrame) -> None:
    # --- ① 表示処理 ---

    filtered_df = df_after[df_after["運搬業者"].notna()]
    df_view = filtered_df[["業者名", "品名", "明細備考", "運搬業者"]]

    st.title("運搬業者の確認")
    st.dataframe(df_view)

    # --- ② Yes/No ボタン形式UI ---
    st.write("この運搬業者選択で確定しますか？")
    col1, col2 = st.columns([1, 1])

    with col1:
        yes_clicked = st.button("✅ はい（この内容で確定）", key="yes_button")
    with col2:
        no_clicked = st.button("🔁 いいえ（やり直す）", key="no_button")

    # --- ③ 処理分岐 ---
    if yes_clicked:
        st.success("✅ 確定されました。次に進みます。")
        return

    if no_clicked:
        st.warning("🔁 選択をやり直します（Step1に戻ります）")
        st.session_state.block_unit_price_confirmed = False
        st.session_state.process_mini_step = 1
        st.rerun()

    # --- ④ ユーザー操作を待機（中断） ---
    st.stop()


def process3(df_after, df_transport):
    from logic.manage.utils.column_utils import apply_column_addition_by_keys

    # --- ① 運搬業者が入っている行を抽出（対象行）
    target_rows = df_after[df_after["運搬業者"].notna()].copy()

    # --- 単価への手数料処理（業者CDで結合） ---
    updated_target_rows = apply_column_addition_by_keys(
        base_df=target_rows,
        addition_df=df_transport,
        join_keys=["業者CD", "運搬業者"],
        value_col_to_add="運搬費",
        update_target_col="運搬費",
    )

    # --- ③ 運搬社数 != 1 の行をそのまま残す（非対象行）
    other_rows = df_after[df_after["運搬業者"].isna()].copy()

    # --- ④ 両方を結合（行順は変更される可能性あり）
    df_after = pd.concat([updated_target_rows, other_rows], ignore_index=True)

    return df_after


def process4(df_after: pd.DataFrame, df_transport: pd.DataFrame) -> pd.DataFrame:
    # --- ① df_transport 側で "数字 * weight" 形式の行だけ抽出 ---
    運搬費_col = df_transport["運搬費"].astype(str).str.replace(r"\s+", "", regex=True)
    mask = 運搬費_col.str.fullmatch(r"\d+\*weight", na=False)

    df_transport_filtered = df_transport[mask].copy()

    # --- ② 数字部分だけを抽出して float に変換（計算係数）---
    df_transport_filtered["運搬費係数"] = (
        df_transport_filtered["運搬費"].str.extract(r"^(\d+)")[0].astype(float)
    )

    # --- ③ 必要な列だけにして、業者CD + 運搬業者でユニーク化 ---
    df_transport_filtered = df_transport_filtered.drop_duplicates(
        subset=["業者CD", "運搬業者"]
    )
    df_transport_filtered = df_transport_filtered[["業者CD", "運搬業者", "運搬費係数"]]

    # --- ④ df_after にマージ（業者CD＋運搬業者） ---
    df_target = df_after.merge(
        df_transport_filtered,
        how="left",
        on=["業者CD", "運搬業者"],
        suffixes=("", "_formula"),
    )

    # --- ⑤ 係数が存在する行だけ掛け算して反映 ---
    calc_mask = df_target["運搬費係数"].notna()
    df_target.loc[calc_mask, "運搬費"] = (
        df_target.loc[calc_mask, "運搬費係数"] * df_target.loc[calc_mask, "正味重量"]
    ).astype(float)

    # --- ⑥ マージ済み df_target を返す or 元の df_after に反映して返す ---
    return df_target


def process5(df):

    # 総額
    df["総額"] = df["単価"] * df["正味重量"] + df["運搬費"]
    df["ブロック単価"] = (df["総額"] / df["正味重量"].replace(0, pd.NA)).round(2)
    return df


def eksc(df):
    import pandas as pd
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, Border, Side, PatternFill
    from openpyxl.utils import get_column_letter

    # dfカラムのフィルタリング
    df = df[["業者名", "明細備考", "正味重量", "総額", "ブロック単価"]]

    #     # カラム名の変更
    #     df = df.rename(columns={
    #     # "業者名": "取引先名",
    #     "明細備考": "明細備考",
    #     "正味重量": "数量",
    #     "総額": "金額",
    #     "ブロック単価": "単価"
    # })
    return df


def ekuserubunkai(df):

    start_row = 7
    full_col_to_cell = {
        "業者名": "B",
        "明細備考": "C",
        "正味重量": "D",
        "総額": "E",
        "ブロック単価": "F",
    }

    # セル情報を再構築
    full_cell_info = []

    for idx, row in df.iterrows():
        for col, col_letter in full_col_to_cell.items():
            cell = f"{col_letter}{start_row + idx}"
            value = row[col]
            full_cell_info.append({"大項目": col, "セル": cell, "値": value})

    full_cell_df = pd.DataFrame(full_cell_info)

    return full_cell_df


def goukei(df, df_shipping):
    from utils.date_tools import to_reiwa_format

    # 日付を令和表記に変換（例: "令和6年5月16日"）
    date = to_reiwa_format(df_shipping["伝票日付"].iloc[0])

    # 追加行を定義
    new_row = pd.DataFrame([{"大項目": "日付", "セル": "E4", "値": date}])

    # df に行を追加
    df = pd.concat([df, new_row], ignore_index=True)

    return df
