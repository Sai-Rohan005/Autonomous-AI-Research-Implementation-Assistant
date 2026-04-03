
import requests


def search_tool(query):
    refined_query = f"{query} explanation implementation example"

    try:
        response = requests.get(
            f"https://api.example.com/search?q={refined_query}",
            timeout=5
        )

        if response.status_code != 200:
            raise Exception("Primary API failed")

        data = response.json()

    except Exception as e:
        print("Primary API Error:", e)

        try:
            response = requests.get(
                f"https://api.example2.com/search?q={refined_query}",
                timeout=5
            )

            if response.status_code != 200:
                raise Exception("Fallback API failed")

            data = response.json()

        except Exception as e:
            print("Fallback API Error:", e)
            return {
                "status": "error",
                "data": [],
                "source": None
            }

    results = []
    for item in data.get("results", []):
        snippet = item.get("snippet", "")
        title = item.get("title", "")

        if snippet:
            results.append({
                "title": title,
                "snippet": snippet
            })

    return {
        "status": "ok",
        "query": refined_query,
        "data": results[:5]
    }