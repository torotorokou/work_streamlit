from utils.logger import app_logger
import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.utils.load_template import load_master_and_template
from config.loader.main_path import MainPath
from logic.readers.read_transport_discount import ReadTransportDiscount
import streamlit as st

def process(dfs):
    logger = app_logger()

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
    mainpath = MainPath()  # YAMLからtransport_costsを含むパス群を取得
    reader = ReadTransportDiscount(mainpath)
    df_transport = reader.load_discounted_df()

    # --- CSVの読み込み ---
    df_dict = load_all_filtered_dataframes(dfs, csv_keys, template_name)
    df_shipping = df_dict.get("shipping")

    # --- 個別処理 ---
    logger.info("▶️ フィルタリング")
    df_after = make_df_shipping_after_use(master_csv, df_shipping)

    logger.info("▶️ 単価1円追加")
    df_after = apply_unit_price_addition(master_csv, df_after)


    # 固定運搬費の算出
    logger.info("▶️ 運搬費（固定）")
    df_after = process1(df_after,df_transport)


    # 選択式運搬費の算出
    logger.info("▶️ 運搬費（選択式）")
    df_after = process2(df_after, df_transport)

    # --- 選択に基づく加算処理 ---
    df_after = apply_selected_transport_cost(df_after, df_transport)



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
    unique_master = master_csv.drop_duplicates(subset=["業者CD"])[["業者CD", "運搬社数"]]
    df_after = df_after.merge(unique_master, on="業者CD", how="left")

    # 運搬費カラムを作成
    df_after["運搬費"] = 0

    # 業者CDで並び替え
    df_after = df_after.sort_values(by="業者CD").reset_index(drop=True)


    return df_after


def apply_unit_price_addition(master_csv, df_shipping: pd.DataFrame) -> pd.DataFrame:
    """
    出荷データ（df）に対して、1円追加情報を業者CD単位でマスターと照合し、
    対象業者の単価に加算を行う処理。
    """
    from logic.manage.utils.column_utils import apply_column_addition_by_keys

    # --- 単価への1円追加処理（業者CDで結合） ---
    df_after = apply_column_addition_by_keys(
        base_df=df_shipping,
        addition_df=master_csv,
        join_keys=["業者CD"],
        value_col_to_add="1円追加",
        update_target_col="単価",
    )

    return df_after


def process1(df_shipping,df_transport):
    from logic.manage.utils.column_utils import apply_column_addition_by_keys

 
    # --- ① 運搬社数 = 1 の行だけを抽出（対象行）
    target_rows = df_shipping[df_shipping["運搬社数"] == 1].copy()

    # --- ② 加算処理を適用
    updated_target_rows = apply_column_addition_by_keys(
        base_df=target_rows,
        addition_df=df_transport,
        join_keys=["業者CD"],
        value_col_to_add="運搬費",
        update_target_col="運搬費"
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

    # --- ① 対象行の抽出 ---
    target_rows = df_after[df_after["運搬社数"] != 1].copy()

    # --- ② 状態初期化 ---
    if "transport_confirmed" not in st.session_state:
        st.session_state.transport_confirmed = False
    if "selected_transport_map" not in st.session_state:
        st.session_state.selected_transport_map = {}

    # --- ③ タイトル・スタイル調整 ---
    st.title("🚚 行ごとの運搬業者選択")

    st.markdown("""
        <style>
        /* h3の下線を消す */
        h3 {
            border: none !important;
            border-bottom: none !important;
            margin-bottom: 0.5rem !important;
        }

        /* selectboxのフォーカス枠線を細く・柔らかく */
        div[data-baseweb="select"] > div {
            border-width: 1px !important;
            border-color: #475569 !important;
        }

        div[data-baseweb="select"]:focus-within {
            box-shadow: 0 0 0 1px #3b82f6 !important; /* 控えめな青色 */
        }
        </style>
    """, unsafe_allow_html=True)

    # --- ④ UI構築（未確定時） ---
    if not st.session_state.transport_confirmed:
        with st.form("transport_selection_form"):
            st.markdown("### 行ごとに適切な運搬業者を選択してください。")

            selected_map = {}

            for idx, row in target_rows.iterrows():
                gyousha_cd = row["業者CD"]
                gyousha_name = str(row.get("業者名", gyousha_cd))
                meisai = str(row.get("明細備考", "")).strip()

                # 表示整形
                gyousha_name_clean = re.sub(r"（\s*\d+\s*）", "", gyousha_name)
                meisai_display = meisai if meisai else "-"

                # 運搬業者候補を取得
                options = df_transport[df_transport["業者CD"] == gyousha_cd]["運搬業者"].tolist()
                if not options:
                    st.warning(f"{gyousha_name_clean} に対応する運搬業者が見つかりません。")
                    continue

                select_key = f"select_row_{idx}"
                if select_key not in st.session_state:
                    st.session_state[select_key] = options[0]

                # --- 枠で囲むブロック（細い線） ---
                st.markdown(f"""
                    <div style='
                        background-color:#1e293b;
                        padding:1px 4px;
                        margin-bottom:6px;
                        border-radius:2px;
                        border:0.3px solid #3b4252;
                    '>
                """, unsafe_allow_html=True)


                col1, col2 = st.columns([2, 3])

                with col1:
                    st.markdown(f"""
                        <div style='padding-right:10px;'>
                            <div style='font-size:18px; font-weight:600; color:#1e293b;'>
                                🚚 {gyousha_name_clean}
                            </div>
                            <div style='font-size:16px; color:#334155;'>
                                備考：{meisai_display}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)


                with col2:
                    selected = st.selectbox(
                        label="運搬業者を選択してください",
                        options=options,
                        key=select_key,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

                selected_map[idx] = selected

            # --- 確定ボタン ---
            submitted = st.form_submit_button("✅ 選択を確定して次へ進む")
            if submitted:
                st.session_state.selected_transport_map = selected_map
                st.session_state.transport_confirmed = True
                st.success("✅ 選択が確定されました。")

    # --- ⑤ 確定後の表示とマージ ---
    if st.session_state.transport_confirmed:
        st.success("以下の行で選択された運搬業者：")
        st.json(st.session_state.selected_transport_map)

        df_after["選択運搬業者"] = df_after.index.map(st.session_state.selected_transport_map)

    return df_after


def apply_selected_transport_cost(df_after: pd.DataFrame, cost_master_df: pd.DataFrame) -> pd.DataFrame:
    import streamlit as st

    if "選択運搬業者" not in df_after.columns:
        st.warning("運搬業者が選択されていません。")
        return df_after

    # 運搬費を加算
    # df_after = add_transport_cost(df_after, cost_master_df)

    # 表示
    st.write("✅ 運搬費加算後データ")
    st.dataframe(df_after)

    return df_after