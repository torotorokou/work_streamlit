# 表示名 → URL ID
page_dict = {
    "トップページ": "home",
    "管理業務": "manage_work",
    "やよい会計": "yayoi",
    "機能２": "feature2",
}

# URL ID → 表示名（逆変換用）
page_dict_reverse = {v: k for k, v in page_dict.items()}

# 表示用のページ名リスト（UI用）
page_labels = list(page_dict.keys())