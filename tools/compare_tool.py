from utils.llm import generate_text,generate_text_fallback
from crewai.tools import BaseTool
from utils.llm import generate_text

class CompareTool(BaseTool):
    name: str = "Comparison Tool"
    description: str = "Compare the given concepts or models. \n Return structured output including pros, cons, and a summary."

    def _run(self, data: str):
        prompt = (
            f"Compare the following concepts or models:\n{data}\n\n"
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
            result = generate_text(prompt)

            return {
                "status": "ok",
                "comparison": result
            }

        except Exception as e:
            print("Primary Comparison LLM Failed:", e)

            try:
                result2 = generate_text_fallback(prompt)

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

        
