from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from utils.llm import generate_text

class SearchTool(BaseTool):
    name: str = "Search Tool"
    description: str = "Searches the web and returns relevant information"

    def _run(self, query: str):
        refined_query = f"{query} explanation implementation example"

        results = []

        # ---------------- DDG SAFE BLOCK ----------------
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(refined_query, max_results=5):
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "link": r.get("href", "")
                    })

        except Exception as ddg_error:
            print("DDG Internal Error:", ddg_error)
            try:
                fallback = generate_text(f"Explain {query}")
                print("After DDG Fallback\n",fallback)
                return [{
                    "title": "Fallback",
                    "snippet": fallback,
                    "link": ""
                }]
            except Exception as e:
                print("Fallback Error:", e)
                return [{
                    "title": "Error",
                    "snippet": "Unable to fetch results",
                    "link": ""
                }]

        return results