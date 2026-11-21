from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from utils.llm import generate_text
from utils.logger import log_error, log_info
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
                "results":results
                }

        except Exception as e:
            log_error("tool_failed")

            # 🔥 WAIT + RETRY
            log_info("search_retry")
            time.sleep(2)

            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(refined_query, max_results=3))
                    return {
                        "status":"ok", 
                        "results":results
                        }
            except:
                log_error("tool_failed")
                try:
                    fallback = generate_text(f"Explain {query}")
                    log_info("search_fallback")
                    return {
                        "status":"ok",
                        "results":fallback
                    }
                except Exception as e:
                    log_error("tool_failed")
                    return {
                        "status":"error",
                        "results":""
                    }

        return {
                "status":"ok", 
                "results":results
                }
