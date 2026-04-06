from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from utils.llm import generate_text
import time

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

            if not results:
                raise Exception("No results")

            return {
                "status":"ok", 
                "search":results
                }

        except Exception as e:
            print("Search Error:", e)

            # 🔥 WAIT + RETRY
            time.sleep(2)

            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(refined_query, max_results=3))
                    return {
                        "status":"ok", 
                        "results":results
                        }
            except:
                print("DDG Internal Error:")
                try:
                    fallback = generate_text(f"Explain {query}")
                    print("After DDG Fallback\n",fallback)
                    return {
                        "status":"ok",
                        "results":fallback
                    }
                except Exception as e:
                    print("Fallback Error:", e)
                    return {
                        "status":"error",
                        "results":""
                    }

        return {
                "status":"ok", 
                "results":results
                }