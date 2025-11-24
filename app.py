from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='gpt-oss:20b', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
# or access fields directly from the response object
print(response.message.content)