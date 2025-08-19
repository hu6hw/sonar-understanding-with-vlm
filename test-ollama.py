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

img_path = 'md_fls_dataset/data/watertank-cropped/shampoo-bottle/shampoo-bottle-13.png'

resp = ollama.generate(model='gemma3', prompt='Which of the following classes is the image most likely to represent? Only reply with one of the class names. can, bottle, drink-carton, chain, propeller, tire, hook, valve, shampoo-bottle, standing-bottle, background', images=[img_path])
print(resp)
print(resp.response)