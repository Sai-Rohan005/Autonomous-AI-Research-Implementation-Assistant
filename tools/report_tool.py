from utils.llm import generate_text,generate_text_fallback
from crewai.tools import BaseTool
from utils.llm import generate_text

class ReportTool(BaseTool):
    name: str = "Report Tool"
    description: str = "Generate the final report using all available information. \n Ensure the output is clear, structured, and easy to understand."

    def _run(self, data: dict):
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
            result = generate_text(prompt)

            return {
                "status": "ok",
                "report": result
            }

        except Exception as e:
            print("Primary Report LLM Failed:", e)

            try:
                result2 = generate_text_fallback(prompt)

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
