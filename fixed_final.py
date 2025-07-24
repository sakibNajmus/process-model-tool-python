# Smart Process Miner App with Proper Image Rendering for Heuristics Miner

from __future__ import annotations
import io
import os
import tempfile
from typing import Tuple, Optional

import pandas as pd
import numpy as np

try:
    import streamlit as st
except ImportError as e:
    raise ImportError(
        "streamlit is required to run this application. "
        "Install it via 'pip install streamlit' and try again."
    ) from e

try:
    import pm4py
    from pm4py.objects.log.util import dataframe_utils
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.algo.discovery.alpha import algorithm as alpha_miner
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
    from pm4py.objects.conversion.process_tree import converter as pt_converter
except ImportError:
    pm4py = None


def clean_event_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    invalid_values = ["", "NaN", "nan", "null", "unknown", np.nan]
    df.replace(invalid_values, np.nan, inplace=True)
    df.drop_duplicates(inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    if 'session_id' not in df.columns:
        if {'type', 'content'}.issubset(df.columns):
            df['session_id'] = df.groupby(['type', 'content']).ngroup().astype(str)
        else:
            df['session_id'] = df.groupby(df.columns.tolist()).ngroup().astype(str)

    if 'semantic_label' not in df.columns:
        if 'type' in df.columns and 'content' in df.columns:
            df['semantic_label'] = df['type'].astype(str) + " → " + df['content'].astype(str)
        elif 'concept:name' in df.columns:
            df['semantic_label'] = df['concept:name']
        else:
            first_obj_col = df.select_dtypes(include='object').columns[0]
            df['semantic_label'] = df[first_obj_col]

    if 'timestamp' not in df.columns:
        candidate_cols = [c for c in df.columns if 'time' in c or 'date' in c]
        if candidate_cols:
            df.rename(columns={candidate_cols[0]: 'timestamp'}, inplace=True)
        else:
            raise ValueError("No timestamp column found. Please include a 'timestamp' column.")

    df = df.sort_values(by=['session_id', 'timestamp'])
    df['prev_action'] = df.groupby('session_id')['semantic_label'].shift()
    df = df[df['semantic_label'] != df['prev_action']].copy()
    df.drop(columns=['prev_action'], inplace=True)

    df['case:concept:name'] = df['session_id']
    df['concept:name'] = df['semantic_label']
    df['time:timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['time:timestamp'], inplace=True)
    return df


def discover_process_model(df: pd.DataFrame) -> Tuple[str, Optional[str], Optional[bytes]]:
    if pm4py is None:
        raise RuntimeError("pm4py is not installed.")

    df_pm4py = dataframe_utils.convert_timestamp_columns_in_df(df)
    event_log = log_converter.apply(df_pm4py, variant=log_converter.Variants.TO_EVENT_LOG)
    trace_count = len(event_log)

    try:
        if trace_count <= 100:
            algorithm_name = "Alpha Miner"
            net, im, fm = alpha_miner.apply(event_log)
            gviz = pn_visualizer.apply(net, im, fm)
            dot = gviz.source
            png_bytes = gviz.pipe(format='png')
            return algorithm_name, dot, png_bytes

        elif 100 < trace_count <= 500:
            algorithm_name = "Inductive Miner"
            process_tree = inductive_miner.apply(event_log)
            net, im, fm = pt_converter.apply(process_tree)
            gviz = pn_visualizer.apply(net, im, fm)
            dot = gviz.source
            png_bytes = gviz.pipe(format='png')
            return algorithm_name, dot, png_bytes

        else:
            algorithm_name = "Heuristics Miner"
            heu_net = heuristics_miner.apply_heu(event_log)
            gviz = hn_visualizer.apply(heu_net, parameters={"format": "png"})

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_path = tmp_file.name

            hn_visualizer.save(gviz, tmp_path)

            with open(tmp_path, "rb") as f:
                png_bytes = f.read()
            os.remove(tmp_path)

            return algorithm_name, None, png_bytes

    except Exception as exc:
        return f"Failed to discover model: {exc}", None, None


def main() -> None:
    st.set_page_config(page_title="Smart Process Miner", layout="wide")
    st.title("📈 Smart Process Mining Web App")

    st.markdown("Upload an Excel or CSV event log. Cleaned data will be shown below. Then discover a process model.")

    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                df_raw = pd.read_excel(uploaded_file)
            elif uploaded_file.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                st.error("Unsupported file type.")
                return
        except Exception as exc:
            st.error(f"Failed to read file: {exc}")
            return

        try:
            df_clean = clean_event_log(df_raw)
        except Exception as exc:
            st.error(f"Error during data cleaning: {exc}")
            return

        st.subheader("Cleaned Event Log (Preview)")
        st.dataframe(df_clean.head(100))
        st.write(f"Total events: {len(df_clean)}")
        st.write(f"Unique sessions: {df_clean['session_id'].nunique()}")

        if st.button("Proceed to Process Modelling"):
            if pm4py is None:
                st.error("PM4Py not installed.")
                return

            with st.spinner("Generating process model..."):
                algo_name, dot, image_bytes = discover_process_model(df_clean)

            if "Failed" in algo_name:
                st.error(algo_name)
            else:
                st.success(f"Model discovered using {algo_name}.")
                if dot:
                    st.graphviz_chart(dot)

                if image_bytes:
                    st.subheader("Process Model Image")
                    st.image(image_bytes, use_column_width=True)
                    st.download_button(
                        label="📥 Download PNG",
                        data=image_bytes,
                        file_name="process_model.png",
                        mime="image/png"
                    )


if __name__ == "__main__":
    main()
