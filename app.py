import streamlit as st
from core.runner import run_agents

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.title("🤖 AI Research & Implementation Assistant")

# ---------------- INPUT ----------------
query = st.text_input("Enter your query:")

# ---------------- BUTTON ----------------
if st.button("Run Agent"):

    if not query:
        st.warning("Please enter a query")

    else:
        with st.spinner("Processing..."):
            result = run_agents(query)

        st.success("Done!")

        # ---------------- DEBUG (OPTIONAL) ----------------
        with st.expander("🔍 Debug Output"):
            st.json(result)

        # ---------------- SEARCH RESULTS ----------------
        if "search" in result:
            st.subheader("🔍 Search Results")

            if isinstance(result["search"], list):
                for item in result["search"]:
                    st.write(f"**{item.get('title', '')}**")
                    st.write(item.get("snippet", ""))
                    if item.get("link"):
                        st.write(f"[Read more]({item.get('link')})")
                    st.write("---")

        # ---------------- SUMMARY ----------------
        if "summary" in result:
            st.subheader("🧾 Summary")

            summary = result["summary"]
            if isinstance(summary, dict):
                st.write(summary.get("summary", ""))
            else:
                st.write(summary)

        # ---------------- CODE ----------------
        if "code" in result:
            st.subheader("💻 Code")

            code = result["code"]
            if isinstance(code, dict):
                st.code(code.get("code", ""), language="python")
            else:
                st.code(code, language="python")

        # ---------------- COMPARISON ----------------
        if "comparison" in result:
            st.subheader("📊 Comparison")

            comparison = result["comparison"]
            if isinstance(comparison, dict):
                st.write(comparison.get("comparison", ""))
            else:
                st.write(comparison)

        # ---------------- REPORT ----------------
        if "report" in result:
            st.subheader("📝 Final Report")

            report = result["report"]
            if isinstance(report, dict):
                st.write(report.get("report", ""))
            else:
                st.write(report)

        # ---------------- EMPTY CASE ----------------
        if not result:
            st.warning("No output generated. Try another query.")