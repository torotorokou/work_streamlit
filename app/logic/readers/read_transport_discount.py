import pandas as pd
from config.loader.main_path import MainPath


class ReadTransportDiscount:
    """
    運搬費CSVを読み込み、合積割引を自動適用したDataFrameを返すクラス。
    """
    def __init__(self, mainpath: MainPath):
        self.mainpath = mainpath
        self.discount_service = TransportDiscountService()

    def load_discounted_df(self) -> pd.DataFrame:
        """
        CSVファイルを読み込み、合積に基づいた割引を適用したDataFrameを返す。
        """
        # YAML経由でCSVパスを取得
        csv_path = self.mainpath.get_path("transport_costs", section="csv")

        # CSV読み込み
        df = pd.read_csv(csv_path)

        # 割引適用
        df = self.discount_service.apply_discount(df)

        return df


class TransportDiscountService:
    """
    合積（合積 == 1）に基づいて、運搬費を割引するサービスクラス。
    """
    def __init__(self, discount_rate: float = 0.5):
        self.discount_rate = discount_rate  # デフォルトで50%割引

    def apply_discount(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        合積列に基づいて、該当行の運搬費に割引を適用する。
        """
        df = df.copy()

        # 必須列の存在チェック
        if "合積" not in df.columns:
            raise KeyError("入力データに '合積' 列が存在しません")

        # 合積 == 1 の行に割引を適用
        mask = df["合積"] == 1
        df.loc[mask, "運搬費"] = df.loc[mask, "運搬費"].astype(float) * self.discount_rate

        return df
