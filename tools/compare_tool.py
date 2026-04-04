from utils.llm import generate_text,generate_text_fallback
from crewai.tools import BaseTool
from utils.llm import generate_text

class CompareTool(BaseTool):
    name: str = "Comparison Tool"
    description: str = "Compare the given concepts or models. \n Return structured output including pros, cons, and a summary."

    def _run(self, data: str):
        prompt = (
    "You are a world-class technical analyst known for clear, balanced, and insightful comparisons.\n\n"
    
    "Compare the following concepts or models:\n"
    f"{data}\n\n"
    
    "Requirements:\n"
    "- Provide a deep, fair, objective, and nuanced comparison\n"
    "- Analyze across relevant dimensions (performance, features, usability, scalability, cost, learning curve, ecosystem, best use cases, limitations, etc.)\n"
    "- Highlight key trade-offs clearly\n"
    "- Include meaningful and specific pros and cons for each item\n"
    "- End with a practical, decision-making summary\n\n"
    
    "Output Format:\n"
    "Return **only** a valid JSON object with the following structure (no extra text, no explanations outside the JSON):\n\n"
    "{\n"
    "  \"comparison_table\": [\n"
    "    {\n"
    "      \"Aspect\": \"Performance\",\n"
    "      \"Item1_Name\": \"...\",\n"
    "      \"Item2_Name\": \"...\"\n"
    "    },\n"
    "    {\n"
    "      \"Aspect\": \"Ease of Use\",\n"
    "      \"Item1_Name\": \"...\",\n"
    "      \"Item2_Name\": \"...\"\n"
    "    }\n"
    "    // add more rows as needed\n"
    "  ],\n"
    "  \"pros_cons\": {\n"
    "    \"Item1_Name\": {\n"
    "      \"pros\": [\"specific point\", \"specific point\", ...],\n"
    "      \"cons\": [\"specific point\", \"specific point\", ...]\n"
    "    },\n"
    "    \"Item2_Name\": {\n"
    "      \"pros\": [\"specific point\", \"specific point\", ...],\n"
    "      \"cons\": [\"specific point\", \"specific point\", ...]\n"
    "    }\n"
    "  },\n"
    "  \"summary\": \"A clear, balanced, and actionable final summary that helps the user decide when to choose which one.\"\n"
    "}\n\n"
    
    "Important Rules:\n"
    "- Use the **actual names** of the concepts/models as keys (do not use 'Item1_Name')\n"
    "- Make the comparison_table rich with 8–12 meaningful aspects/rows\n"
    "- Make pros and cons specific and useful (avoid vague statements)\n"
    "- Ensure the JSON is perfectly valid and parseable\n"
    "- The table should help users quickly compare at a glance"
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

        
