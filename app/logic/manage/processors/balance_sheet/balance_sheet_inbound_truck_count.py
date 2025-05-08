def inbound_truck_count(df_receive):
    total = df_receive["受入番号"].nunique()
    return total
