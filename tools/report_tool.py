def report_tool(data):
    """
    data = {
        "summary": "...",
        "code": "...",
        "comparison": "..."
    }
    """

    prompt = (
        "Generate a final structured report using the following information:\n\n"
        f"Summary:\n{data.get('summary', '')}\n\n"
        f"Code:\n{data.get('code', '')}\n\n"
        f"Comparison:\n{data.get('comparison', '')}\n\n"
        "Requirements:\n"
        "- Explain clearly\n"
        "- Keep it structured\n"
        "- Make it easy to understand\n"
        "- Highlight key insights\n"
    )

    try:
        result = report_llm(prompt)

        return {
            "status": "ok",
            "report": result
        }

    except Exception as e:
        print("Primary Report LLM Failed:", e)

        try:
            result2 = report_llm2(prompt)

            return {
                "status": "ok",
                "report": result2
            }

        except Exception as e:
            print("Secondary Report LLM Failed:", e)

            return {
                "status": "error",
                "report": ""
            }
