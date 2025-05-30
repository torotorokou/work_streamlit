# --- 表のスタイル指定用（ラベル別色） ---
def style_label(val):
    if val == "警告":
        return "color: red; font-weight: bold"
    elif val == "注意":
        return "color: orange"
    return ""
