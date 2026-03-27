# import requests

# api_key = ""

# url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# # url = "https://generativelanguage.googleapis.com/v1beta/models"

# payload = {
#     "contents": [
#         {
#             "parts": [{"text": "Hello"}]
#         }
#     ]
# }

# response = requests.post(url, params={"key": api_key}, json=payload)

# print(response.status_code)
# print(response.text)

# import requests


# # Updated the URL to use the active 'gemini-2.5-flash' model
# url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# payload = {
#     "contents": [
#         {
#             "parts": [{"text": "Explain gravity simply"}]
#         }
#     ]
# }

# response = requests.post(url, params={"key": api_key}, json=payload)

# print(response.status_code)
# print(response.text)




from openai import OpenAI

# 1. Leave your PAT here
github_token = "PAT" 

# 2. Use the standard Azure endpoint for Python
client = OpenAI(
    base_url="https://models.inference.ai.azure.com", 
    api_key=github_token,
)

# 3. Switch back to the free-tier model
response = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=[
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)