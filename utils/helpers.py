def clean_text(text):
    return text.strip().replace("\n", " ")

def format_response(data):
    return {
        "summary": data.get("summary", ""),
        "code": data.get("code", ""),
        "comparison": data.get("comparison", ""),
        "report": data.get("report", "")
    }

def validate_query(query):
    return len(query) > 3