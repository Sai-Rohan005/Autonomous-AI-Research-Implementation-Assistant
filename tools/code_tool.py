from utils.llm import generate_text,generate_text_fallback
from crewai.tools import BaseTool
from utils.llm import generate_text

class CodeTool(BaseTool):
    name: str = "Code Genrator Tool"
    description: str = "Generate high-quality code for the given query."

    def _run(self, data: str):
        prompt = (
    "You are an expert in software engineer with 15+ years of experience, "
    "known for writing exceptionally clean, idiomatic, production-ready code.\n\n"
    
    "Generate high-quality, clean, and well-structured code for the following task:\n\n"
    f"{data}\n\n"
    
    "Requirements:\n"
    "- Write **production-grade, professional-quality** code\n"
    "- Follow **PEP 8** style guide and best practices strictly\n"
    "- Use **modern features (type hints, dataclasses, context managers, etc. where appropriate)\n"
    "- Make the code **highly readable**, self-documenting, and maintainable\n"
    "- Include **clear, concise, and useful comments** only where they add value\n"
    "- Add comprehensive **docstrings** (Google or NumPy style) for all functions and classes\n"
    "- Handle edge cases and errors gracefully with proper exception handling\n"
    "- Include proper input validation where relevant\n"
    "- Choose the most appropriate data structures and algorithms for efficiency and clarity\n"
    "- If the task involves performance-critical sections, optimize thoughtfully\n"
    "- Prefer simplicity and readability over cleverness\n"
    
    "Output Format:\n"
    "1. First, provide a brief explanation of your approach (2-4 sentences)\n"
    "2. Then output the complete, ready-to-run code inside a single markdown code block\n"
    "3. At the end, optionally suggest any improvements or alternative approaches if relevant\n\n"
    
    "Do not include any explanations outside the requested structure. "
    "Focus on writing the best possible code you can produce."
)

        try:
            result = generate_text(prompt)

            return {
                "status": "ok",
                "results": result             
            }

        except Exception as e:
                print("Primary LLM Failed:", e)

                try:
                    result_second = generate_text_fallback(prompt)

                    return {
                        "status": "ok",
                        "results": result_second
                    }

                except Exception as e:
                    print("Secondary LLM Failed:", e)

                    return {
                        "status": "error",
                        "results": ""
                    }

