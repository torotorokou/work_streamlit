def resolve_dtype(dtype_str):
    import numpy as np

    dtype_map = {
        "int": int,
        "float": float,
        "str": str,
        "datetime": "datetime64[ns]",
        "datetime64": "datetime64[ns]",
        "np.int64": np.int64,
        "np.float64": np.float64,
    }

    return dtype_map.get(dtype_str, str)  # デフォルトはstrにしておく
