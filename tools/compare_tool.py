def compare_tool(query):
    prompt = (
        f"Compare the following concepts or models:\n{query}\n\n"
        "Requirements:\n"
        "- Provide comparison in structured JSON format\n"
        "- Include pros and cons for each item\n"
        "- Include a final summary\n\n"
        "Format:\n"
        "{\n"
        "  'item1': {'pros': [...], 'cons': [...]},\n"
        "  'item2': {'pros': [...], 'cons': [...]},\n"
        "  'summary': '...'\n"
        "}"
    )

    try:
        result = compare_llm(prompt)

        return {
            "status": "ok",
            "comparison": result
        }

    except Exception as e:
        print("Primary Comparison LLM Failed:", e)

        try:
            result2 = compare_llm2(prompt)

            return {
                "status": "ok",
                "comparison": result2
            }

        except Exception as e:
            print("Secondary LLM Failed:", e)

            return {
                "status": "error",
                "comparison": ""
            }
