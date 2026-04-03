def code_tool(query):
    prompt = (
        f"Generate clean, well-structured Python code for the following task:\n{query}\n\n"
        "Requirements:\n"
        "- Code should be readable\n"
        "- Include comments\n"
        "- Follow best practices\n"
    )

    try:
        result = code_llm(prompt)

        return {
            "status": "ok",
            "code": result
        }

    except Exception as e:
        print("Primary LLM Failed:", e)

        try:
            result_second = code_llm2(prompt)

            return {
                "status": "ok",
                "code": result_second
            }

        except Exception as e:
            print("Secondary LLM Failed:", e)

            return {
                "status": "error",
                "code": ""
            }
