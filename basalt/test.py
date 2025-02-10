from basalt import Basalt

basalt = Basalt(api_key="sk-7bd58c904a7f6b148b0cf5c915ae72b46fb2a164d7542ee1cefd4e3784ecb5ee")

err, prompts = basalt.prompt.list()

for prompt in prompts:
    print(prompt.slug)

print(err)