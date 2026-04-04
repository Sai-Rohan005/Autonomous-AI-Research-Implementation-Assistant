from utils.llm import generate_text,generate_text_fallback
from crewai.tools import BaseTool
from utils.llm import generate_text

class CodeTool(BaseTool):
    name: str = "Code Genrator Tool"
    description: str = "Generate high-quality code for the given query."

    def _run(self, data: str):
        prompt = (
                f"Generate clean, well-structured Python code for the following task:\n{data}\n\n"
                "Requirements:\n"
                "- Code should be readable\n"
                "- Include comments\n"
                "- Follow best practices\n"
            )

        try:
            result = generate_text(prompt)

            return {
                "status": "ok",
                "code": result             
            }

        except Exception as e:
                print("Primary LLM Failed:", e)

                try:
                    result_second = generate_text_fallback(prompt)

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

