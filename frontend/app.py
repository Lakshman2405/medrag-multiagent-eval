import streamlit as st
import json
import pandas as pd
import os
import requests


# Auto-detect backend URL
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000"   # default for local
)



def split_logs_by_agent(logs: str):
    sections = {}

    current_agent = "General"
    sections[current_agent] = []

    for line in logs.splitlines():

        if "Agent:" in line:
            current_agent = line.split("Agent:")[-1].strip()
            if current_agent not in sections:
                sections[current_agent] = []

        sections[current_agent].append(line)

    return sections




st.set_page_config(layout="wide", page_title="MedQA Multi-Agent Dashboard")

# ---------------- LOAD STATIC JSON ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "dashboard_medqa.json")

if not os.path.exists(file_path):
    file_path = os.path.join(BASE_DIR, "backend", "dashboard_medqa.json")

with open(file_path) as f:
    static_data = json.load(f)

# =========================================================
# 🔧 FUNCTION: RENDER DASHBOARD (GLOBAL, NOT INSIDE IF)
# =========================================================
import streamlit as st
import pandas as pd
import altair as alt


def render_dashboard(data, dynamic=False):

    if not data.get("results"):
        st.warning("No results returned from backend")
        return

    summary = data.get("summary", {})
    results = pd.DataFrame(data.get("results", []))

    acc_type = pd.DataFrame([
        {"Type": k, "Accuracy": v}
        for k, v in data.get("accuracy_by_type", {}).items()
    ])

    fail_dist = pd.DataFrame([
        {"Failure": k, "Count": v}
        for k, v in data.get("failure_distribution", {}).items()
    ])

    iterations = pd.DataFrame([
        {"Iterations": str(k), "Count": v}
        for k, v in data.get("iteration_distribution", {}).items()
    ])

    calibration = pd.DataFrame(data.get("calibration", []))

    sub_tab = st.radio(
        "Select Analysis",
        ["Overview", "Performance", "Failure Analysis", "Case Explorer"],
        horizontal=True
    )

    # ================= OVERVIEW =================
    if sub_tab == "Overview":

        st.title("Multi-Agent Evaluation Dashboard")

        total_failures = fail_dist["Count"].sum() if not fail_dist.empty else 0
        total_cases = summary.get("total_cases", 1)
        failure_rate = (total_failures / total_cases) * 100

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{summary.get('accuracy', 0):.2f}%")
        col2.metric("Failure Rate", f"{failure_rate:.2f}%")
        col3.metric("Avg Time", f"{summary.get('avg_time', 0):.2f}s")
        col4.metric("Iterations", f"{summary.get('avg_iters', 0):.2f}")

        st.divider()

        if not acc_type.empty:
            best = acc_type.loc[acc_type["Accuracy"].idxmax()]
            worst = acc_type.loc[acc_type["Accuracy"].idxmin()]
        else:
            best = {"Type": "N/A", "Accuracy": 0}
            worst = {"Type": "N/A", "Accuracy": 0}

        dominant_text = "N/A"
        if not fail_dist.empty:
            dominant = fail_dist.loc[fail_dist["Count"].idxmax()]
            dominant_text = f"{dominant['Failure']} ({dominant['Count']} cases)"

        if dynamic:
            st.subheader("🧠 Auto-Generated Insights")
        else:
            st.subheader("📊 Summary Insights")

        st.markdown(f"""
        - Accuracy: **{summary.get('accuracy', 0):.2f}%**
        - Total Cases: **{total_cases}**
        - Avg Time: **{summary.get('avg_time', 0):.2f}s**
        - Avg Iterations: **{summary.get('avg_iters', 0):.2f}**

        **Best Category:** {best['Type']} ({best['Accuracy']:.1f}%)  
        **Worst Category:** {worst['Type']} ({worst['Accuracy']:.1f}%)  

        **Top Failure:** {dominant_text}
        """)

    # ================= PERFORMANCE =================
    elif sub_tab == "Performance":

        col1, col2 = st.columns(2)

        with col1:
            if not acc_type.empty:
                chart = alt.Chart(acc_type).mark_bar().encode(
                    x=alt.X("Type:N", title="Question Type"),
                    y=alt.Y("Accuracy:Q", title="Accuracy (%)"),
                    tooltip=["Type", "Accuracy"]
                ).properties(title="Accuracy by Question Type")
                st.altair_chart(chart, use_container_width=True)

        with col2:
            if not iterations.empty:
                chart = alt.Chart(iterations).mark_bar().encode(
                    x=alt.X("Iterations:N", title="Number of Iterations"),
                    y=alt.Y("Count:Q", title="Number of Cases"),
                    tooltip=["Iterations", "Count"]
                ).properties(title="Iteration Distribution")
                st.altair_chart(chart, use_container_width=True)

        if not calibration.empty:
            chart = alt.Chart(calibration).mark_line(point=True).encode(
                x=alt.X("bin:N", title="Confidence Bin"),
                y=alt.Y("accuracy:Q", title="Accuracy"),
                tooltip=["bin", "accuracy"]
            ).properties(title="Confidence Calibration Curve")

            st.altair_chart(chart, use_container_width=True)

    # ================= FAILURE =================
    elif sub_tab == "Failure Analysis":

        col1, col2 = st.columns(2)

        with col1:
            if not fail_dist.empty:
                chart = alt.Chart(fail_dist).mark_bar().encode(
                    x=alt.X("Failure:N", title="Failure Type"),
                    y=alt.Y("Count:Q", title="Number of Failures"),
                    tooltip=["Failure", "Count"]
                ).properties(title="Failure Distribution")

                st.altair_chart(chart, use_container_width=True)

        with col2:
            if not results.empty:
                matrix = results[~results["correct"]].groupby(
                    ["q_type", "failure_type"]
                ).size().reset_index(name="count")

                st.dataframe(matrix, use_container_width=True)

    # ================= CASE EXPLORER =================
    elif sub_tab == "Case Explorer":
    
        st.title("🔍 Case Explorer")
    
        col1, col2, col3 = st.columns(3)
    
        filter_option = col1.selectbox(
            "Filter", ["ALL", "CORRECT", "WRONG"], key="ce_filter"
        )
    
        search = col2.text_input(
            "Search case/type", key="ce_search"
        )
    
        sort_by = col3.selectbox(
            "Sort by", ["case", "time_s"], key="ce_sort"
        )
    
        df = results.copy()
    
        # ---------------- FILTER ----------------
        if filter_option == "CORRECT":
            df = df[df["correct"]]
        elif filter_option == "WRONG":
            df = df[~df["correct"]]
    
        # ---------------- SEARCH ----------------
        if search:
            df = df[
                df["case"].astype(str).str.contains(search, case=False, na=False) |
                df["q_type"].astype(str).str.contains(search, case=False, na=False)
            ]
    
        # ---------------- SORT ----------------
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=False)
    
        # ---------------- SUMMARY METRICS ----------------
        if not df.empty:
            colA, colB = st.columns(2)
            colA.metric("Cases", len(df))
            colB.metric("Accuracy", f"{df['correct'].mean()*100:.1f}%")
    
        # ---------------- TABLE ----------------
        st.dataframe(df, use_container_width=True)
    
        # ---------------- HARDEST CASES ----------------
        st.subheader("🔥 Hardest Cases (Slow + Wrong)")
    
        if not results.empty:
            hard = results[~results["correct"]].sort_values(
                by="time_s", ascending=False
            ).head(5)
    
            st.dataframe(hard, use_container_width=True)
    
        # ---------------- DOWNLOAD ----------------
        if not df.empty:
            st.download_button(
                "⬇ Download CSV",
                df.to_csv(index=False),
                "medqa_results.csv",
                "text/csv"
            )

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧭 Navigation")

tab = st.sidebar.radio(
    "Go to",
    ["Home", "Evaluation", "Live Agent Pipeline", "Dynamic Evaluation"]
)


# =========================================================
# 🏠 HOME PANEL
# =========================================================
if tab == "Home":

    st.title("🧠 Multi-Agent Clinical Decision Support System")

    st.markdown("""
    Welcome to the **Multi-Agent Medical QA Evaluation Dashboard**.

    This system demonstrates:
    
    ### ⚡ Core Capabilities
    - Multi-agent clinical reasoning pipeline
    - Retrieval-Augmented Generation (RAG) using medical knowledge base
    - Adaptive iteration with confidence-based stopping
    - End-to-end evaluation on medical QA datasets

    ---
    
    ### Available Panels

    #### 📊 Evaluation ( Static Evaluation)
    - Displays precomputed results from dataset
    - Includes accuracy, failure analysis, calibration

    #### ⚡ Live Agent
    - Run the full 5-agent pipeline on a single custom input
    - View reasoning and final prediction

    #### 🚀 Dynamic Evaluation
    - Run full pipeline on dataset (user-defined size)
    - Generates evaluation metrics in real-time
    - Download results as JSON / CSV

    ---
    
    ### 🧠 Pipeline Overview
    1. Clinical Text Clarifier  
    2. RAG-Based Knowledge Retrieval  
    3. Evidence Scanner  
    4. Data Fusion Agent  
    5. Adaptive Optimizer  

    ---
    
    ### 📌 Notes
    - Each case may take **30–60 seconds**
    - Dynamic evaluation may take several minutes
    - Results are generated live using LLM APIs

    ---
    
    👉 Use the sidebar to navigate through the system.
    """)

    st.divider()

    st.success("System Ready for Interaction 🚀")

# =========================================================
# 📊 STATIC
# =========================================================
elif tab == "Evaluation":

    st.title("📊 Static Evaluation")
    render_dashboard(static_data, dynamic=False)

# =========================================================
# ⚡ LIVE AGENT
# =========================================================
elif tab == "Live Agent Pipeline":

    st.title("Live Agent Pipeline")

    # ---------------- INPUTS ----------------
    question = st.text_area("Question")
    options = st.text_area("Options (one per line)")

    run_clicked = st.button("Run")

    # ---------------- SESSION STATE INIT ----------------
    if "result_data" not in st.session_state:
        st.session_state.result_data = None

    if "logs_data" not in st.session_state:
        st.session_state.logs_data = ""

    if "last_question" not in st.session_state:
        st.session_state.last_question = ""

    if "last_options" not in st.session_state:
        st.session_state.last_options = ""

    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "Answer"

    # ---------------- RUN PIPELINE ----------------
    if run_clicked:

        with st.spinner("Running multi-agent pipeline..."):
            options_list = [opt.strip() for opt in options.split("\n") if opt.strip()]
            response = requests.post(
                f"{BACKEND_URL}/run-case",
                json={
                    "question": question,
                    "options": options_list
                }
            )

        if response.status_code != 200:
            st.error(f"Backend Error: {response.text}")
            st.stop()

        data = response.json()

        # Store results
        st.session_state.result_data = data.get("result", {})
        st.session_state.logs_data = data.get("logs", "")
        st.session_state.last_question = question
        st.session_state.last_options = options

        # Default to Answer after run
        st.session_state.view_mode = "Answer"

    # ---------------- DISPLAY ----------------
    if st.session_state.result_data:

        result = st.session_state.result_data
        logs = st.session_state.logs_data
        question = st.session_state.last_question
        options = st.session_state.last_options

        # ---------------- VIEW SWITCH ----------------
        col_btn1, col_btn2 = st.columns(2)

        if col_btn1.button("Answer"):
            st.session_state.view_mode = "Answer"

        if col_btn2.button("Logs"):
            st.session_state.view_mode = "Logs"

        st.divider()

        # ---------------- LOG PANEL ----------------
        if st.session_state.view_mode == "Logs":

            import re

            st.subheader("Agent Execution Logs")

            agent_logs = split_logs_by_agent(logs)

            # 🔹 Helper: normalize confidence in logs
            def normalize_log_confidence(text):

                def repl(match):
                    try:
                        val = float(match.group(1))
                        if val > 1:
                            val = val / 10
                        return f"CONFIDENCE: {round(val, 2)}"
                    except:
                        return match.group(0)

                return re.sub(r"CONFIDENCE:\s*(\d*\.?\d+)", repl, text)

            for agent, content in agent_logs.items():

                # 🔥 Apply normalization line-by-line
                normalized_content = [
                    normalize_log_confidence(line) for line in content
                ]

                with st.expander(agent, expanded=(agent == "General")):
                    st.code("\n".join(normalized_content))


        # ---------------- ANSWER PANEL ----------------
        elif st.session_state.view_mode == "Answer":

            st.subheader("Final Answer")

            answer = result.get("answer", "N/A")
            confidence = result.get("confidence", 0)
            iterations = result.get("iterations", 0)
            raw_output = result.get("raw_output", "")

            col1, col2, col3 = st.columns(3)

            col1.metric("Answer", answer)
            col2.metric("Confidence", f"{confidence:.2f}")
            col3.metric("Iterations", iterations)

            if confidence >= 0.8:
                st.success("High confidence")
            elif confidence >= 0.5:
                st.warning("Moderate confidence")
            else:
                st.error("Low confidence")

            st.divider()

            # ---------------- EXPLANATION ----------------
            clean_text = raw_output.replace("\\n", "\n")

            st.subheader("Explanation")

            import re

            def clean_and_format(text):

                # 🔹 Fix inline sections → force newline before them
                sections = [
                    "Key Clinical Clues:",
                    "Clinical Interpretation:",
                    "Differential Diagnosis:",
                    "Why Correct Answer:",
                    "Option Elimination:",
                    "Core Concept:",
                    "Edge Case Note:"
                ]

                for sec in sections:
                    text = re.sub(rf"\s*-?\s*{sec}", f"\n\n{sec}", text)

                # 🔹 Fix A: B: C: D:
                text = re.sub(r"\s([A-D]:)", r"\n\1", text)

                # 🔹 Fix bullets
                text = re.sub(r"\s-\s", "\n- ", text)

                # 🔹 Bold section headers
                for sec in sections:
                    text = text.replace(
                        sec,
                        f"<div style='font-size:16px; font-weight:700; margin-top:16px'>{sec}</div>"
                    )

                return text


            formatted = clean_and_format(clean_text)

            #  FINAL DISPLAY (same theme, improved hierarchy)
            st.markdown(f"""
            <div style="
                background-color:#111;
                padding:20px;
                border-radius:10px;
                color:#0f0;
                font-family:monospace;
                line-height:1.7;
            ">
            {formatted}
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # ---------------- DOWNLOAD ----------------
            st.subheader("Download")

            st.download_button(
                "Download JSON",
                json.dumps(result, indent=2),
                file_name="result.json"
            )

            st.download_button(
                "Download Logs",
                logs,
                file_name="agent_logs.txt"
            )

            combined = f"""
                QUESTION:
                {question}

                OPTIONS:
                {options}

                RESULT:
                {json.dumps(result, indent=2)}

                LOGS:
                {logs}
                """

            st.download_button(
                "Download Full Report",
                combined,
                file_name="full_report.txt"
            )

        # ---------------- CLEAR BUTTON ----------------
        if st.button("Clear Output"):
            st.session_state.result_data = None
            st.session_state.logs_data = ""
            st.session_state.last_question = ""
            st.session_state.last_options = ""
            st.session_state.view_mode = "Answer"
            st.rerun()



# =========================================================
# 🚀 DYNAMIC
# =========================================================
elif tab == "Dynamic Evaluation":

    st.title("🚀 Dynamic Evaluation")

    # 🔹 SESSION STATE INIT
    if "dynamic_data" not in st.session_state:
        st.session_state.dynamic_data = None

    num_cases = st.number_input(
        "Enter cases",
        min_value=1,
        max_value=50,
        value=5
    )

    if st.button("Run Dataset"):

        st.info("Press Ctrl+C in terminal to stop execution if needed")

        with st.spinner("Running full dataset evaluation... This may take several minutes"):

            try:
                response = requests.post(
                    f"{BACKEND_URL}/run-dataset",
                    json={"num_cases": int(num_cases)}
                )

                if response.status_code != 200:
                    st.error(f"Backend Error: {response.text}")
                    st.stop()

                st.session_state.dynamic_data = response.json()

            except Exception as e:
                st.error(f"Request failed: {str(e)}")
                st.stop()

    # 🔥 ALWAYS RENDER FROM SESSION STATE
    if st.session_state.dynamic_data:

        data = st.session_state.dynamic_data

        st.success("Evaluation Complete ✅")

        render_dashboard(data, dynamic=True)

        # 🔹 DOWNLOADS (SAFE)
        st.download_button(
            "Download JSON",
            json.dumps(data, indent=2),
            file_name="dynamic.json"
        )

        st.download_button(
            "Download CSV",
            pd.DataFrame(data.get("results", [])).to_csv(index=False),
            file_name="dynamic.csv"
        )
