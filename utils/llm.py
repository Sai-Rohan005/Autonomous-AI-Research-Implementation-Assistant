import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------- PRIMARY FUNCTION ----------------
def generate_text(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            # model="gemini-2.5-flash",
            # model="gemini-2.0-flash",
            contents=prompt
        )

        if response.text:
            return response.text
        else:
            raise Exception("Empty response from primary")

    except Exception as e:
        print("Primary Model Error:", e)

        return generate_text_fallback(prompt)


# ---------------- FALLBACK FUNCTION ----------------
def generate_text_fallback(prompt):
    try:
        print("Using fallback model...")

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )

        if response.text:
            return response.text
        else:
            raise Exception("Empty response from fallback")

    except Exception as e:
        print("Fallback Model Error:", e)

        return ""