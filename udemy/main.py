import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image

st.title("Streamlit 超入門")

st.write("プログレスバーの表示")
'start!!'

latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    latest_iteration.text(f'Iteration {i+1}')
    bar.progress(i + 1)
    time.sleep(0.01)

"done!!!"

st.write("DataFrame")

df = pd.DataFrame({
    "1列目": [1, 2, 3, 4],
    "2列目": [10, 20, 30, 40]
})

st.write(df)


# st.dataframe(df.style.highlight_max(axis=0), width=300, height=300)

# st.table(df.style.highlight_max(axis=0))

# """
# # 章
# ## 節
# ### 項

# ``` python
# import streamlit as st
# import numpy as np
# import pandas as pd
# ```
# """

df = pd.DataFrame(
    np.random.rand(20,3),
    columns=['a', 'b', 'c']
)

st.write(df)
st.line_chart(df)
st.area_chart(df)
st.bar_chart(df)


df = pd.DataFrame(
    np.random.rand(100, 2)/[50, 50]+[35.69, 139.70],
    columns=['lat', 'lon']
)

st.map(df)


st.write("Display Image")
image = Image.open("/work/10.work_env/work_streamlit/sample.jpg")
st.image(image, caption="sample", use_container_width=True)