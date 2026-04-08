import streamlit as st
from core.runner import run_agents
import json

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.title("🤖 Autonomous AI Research Assistant")
st.caption("Powered by ReAct Agent Architecture")

# ---------------- SESSION STATE ----------------
if "running" not in st.session_state:
    st.session_state.running = False

# ---------------- INPUT ----------------
query = st.text_input("🔍 Enter your query:")

# ---------------- BUTTON ----------------
run_clicked = st.button(
    "🚀 Run Agent",
    disabled=st.session_state.running
)

# ---------------- EXECUTION ----------------
if run_clicked:

    if not query:
        st.warning("Please enter a query")

    else:
        st.session_state.running = True

        with st.spinner("🤖 Agent is thinking..."):
            result = run_agents(query)

        st.session_state.running = False

        st.success("✅ Completed!")
        
        # with st.expander("debug"):
        #     st.write(result)
        # ---------------- FINAL ANSWER ----------------
        if isinstance(result, dict) and result.get("final_answer"):
            st.subheader("🎯 Final Answer")
            st.write(result["final_answer"])
            st.divider()
        

        # ---------------- COMPARISON ----------------
        comparison = result.get("comparison")
        if comparison:
            st.subheader("📊 Comparison")

            if isinstance(comparison, dict):
                data = comparison.get("results")

                try:
                    parsed = json.loads(data)
                except:
                    parsed = data

                if isinstance(parsed, dict):
                    st.write(parsed.get("summary", ""))
                    st.table(parsed.get("comparison_table", []))
                else:
                    st.write(parsed)

            else:
                st.write(comparison)

            st.divider()


        # ---------------- CODE ----------------
        code = result.get("code")
        if code:
            st.subheader("💻 Code")

            if isinstance(code, dict):
                st.code(code.get("results", ""), language="python")
            else:
                st.code(code, language="python")

            st.divider()


        # ---------------- SUMMARY ----------------
        summary = result.get("summary")
        if summary:
            st.subheader("🧾 Explanation")

            if isinstance(summary, dict):
                st.write(summary.get("results", ""))
            else:
                st.write(summary)

            st.divider()


        # ---------------- REPORT ----------------
        report = result.get("report")
        if report:
            st.subheader("📝 Detailed Report")

            if isinstance(report, dict):
                st.write(report.get("results", ""))
            else:
                st.write(report)

            st.divider()


        # ---------------- SEARCH RESULTS ----------------
        search = result.get("search")
        if search:
            st.subheader("🔍 Sources")

            if isinstance(search, dict):
                data = search.get("results")

                if isinstance(data, list):
                    for item in data:
                        st.write(f"**{item.get('title', '')}**")
                        st.write(item.get("snippet", ""))
                        if item.get("link"):
                            st.markdown(f"[Read more]({item.get('link')})")
                        st.write("---")
                else:
                    st.write(data)