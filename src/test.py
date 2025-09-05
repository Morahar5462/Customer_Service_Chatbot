from langchain_google_genai import GoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

print(llm("Hello, Gemini!"))
