from utils.llm import generate_text,generate_text_fallback
from crewai.tools import BaseTool
from utils.llm import generate_text

class SummarizeTool(BaseTool):
    name: str = "Summarization Tool"
    description: str = "Summarizes text into a concise explanation"

    def _run(self, data: str):
        try:
            result = generate_text(
                f"Summarize the following content clearly and concisely:\n{data}"
            )

            return {
                "status": "ok",
                "summary": result
            }

        except Exception as e:
            print("Primary LLM Failed:", e)

            try:
                result_second = generate_text_fallback(
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

    