def summarize_tool(data):
    try:
        result = summarize_llm(
            f"Summarize the following content clearly and concisely:\n{data}"
        )

        return {
            "status": "ok",
            "summary": result
        }

    except Exception as e:
        print("Primary LLM Failed:", e)

        try:
            result_second = summarize_llm2(
                f"Summarize the following content clearly and concisely:\n{data}"
            )

            return {
                "status": "ok",
                "summary": result_second
            }

        except Exception as e:
            print("Secondary LLM Failed:", e)

            return {
                "status": "error",
                "summary": ""
            }
