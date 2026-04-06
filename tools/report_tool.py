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
    "You are a principal technical writer and architect renowned for producing exceptionally clear, structured, and high-impact reports.\n\n"
    
    "Create a professional final report by synthesizing the following information:\n\n"
    f"Summary:\n{data.get('summary', '')}\n\n"
    f"Code:\n{data.get('code', '')}\n\n"
    f"Comparison:\n{data.get('comparison', '')}\n\n"
    
    "Strict Requirements:\n"
    "- Produce a polished, executive-level report\n"
    "- Ensure excellent structure, flow, and readability\n"
    "- Highlight key insights, trade-offs, strengths, and limitations clearly\n"
    "- Use professional yet accessible language\n"
    "- Make complex ideas easy to understand\n"
    "- Focus on what matters most to the reader (decision-making and implementation)\n\n"
    
    "Output using this **exact Markdown structure**:\n\n"
    "# [Clear and Descriptive Report Title]\n\n"
    "## Executive Summary\n\n"
    "## Key Insights\n\n"
    "## Comparison Analysis\n\n"
    "## Proposed Solution & Code\n\n"
    "## Detailed Breakdown\n\n"
    "## Recommendations\n\n"
    "## Conclusion\n\n"
    
    "Rules:\n"
    "- Use Markdown formatting generously for readability (headings, bullets, bold, tables if useful)\n"
    "- Place all code in well-formatted ```python blocks\n"
    "- Add brief explanations before/after code sections\n"
    "- Keep each section concise but informative\n"
    "- End with strong, actionable recommendations\n"
    "Think carefully and produce the best possible report."
)

        try:
            result = generate_text(prompt)

            return {
                "status": "ok",
                "results": result
            }

        except Exception as e:
            print("Primary Report LLM Failed:", e)

            try:
                result2 = generate_text_fallback(prompt)

                return {
                    "status": "ok",
                    "results": result2
                }

            except Exception as e:
                print("Secondary Report LLM Failed:", e)

                return {
                    "status": "error",
                    "results": ""
                }
