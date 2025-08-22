import ollama 

# response: ChatResponse = chat(model='gemma3', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])
# # or access fields directly from the response object
# print(response.message.content)

img_path = 'neom-cZkdauWij50-unsplash.jpg'

resp = ollama.generate(model='llava:34b', prompt='What is the image', images=[img_path])
print(resp)
print(resp.response)
