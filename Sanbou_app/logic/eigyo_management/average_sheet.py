import streamlit as st
import pandas as pd
# ABC平均表
def process(dfs, label_map):
    # 工場日報の処理を書く
    print("📄 ABCの処理..")

    df = []
    print("CSV読込")