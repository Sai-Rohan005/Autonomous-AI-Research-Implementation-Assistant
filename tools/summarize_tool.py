from utils.llm import generate_text,generate_text_fallback
from crewai.tools import BaseTool
from utils.llm import generate_text

class SummarizeTool(BaseTool):
    name: str = "Summarization Tool"
    description: str = "Summarizes text into a concise explanation"

    def _run(self, data: str):
        prompt = (
    "You are a world-class summarizer and technical communicator, expert at distilling complex information into clear, concise, and powerful summaries.\n\n"
    
    "Task:\n"
    "Summarize the following content:\n\n"
    f"{data}\n\n"
    
    "Strict Requirements:\n"
    "- Be **highly concise** while preserving all critical information and insights\n"
    "- Make the summary **scannable and readable** (use short sentences, clear structure, and bullet points when it improves clarity)\n"
    "- Focus on the **core message**, key arguments, main findings, and actionable takeaways\n"
    "- Use precise, professional language\n"
    "- Eliminate redundancy, filler words, and weak phrasing\n"
    "- Maintain the original meaning and tone accurately\n"
    "- Do not add any new information, opinions, or interpretations\n\n"
    
    "Response Rules:\n"
    "- Output **only** the summary\n"
    "- Do not use phrases like 'Summary:', 'Here is a summary of...', or any meta-commentary\n"
    "- Start directly with the content\n"
    "- Aim for the perfect balance between brevity and completeness"
)
        try:
            result = generate_text(prompt)

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

    