import openai
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY_OPENAI= os.getenv('API_KEY_OPENAI')



client = OpenAI(api_key=API_KEY_OPENAI)
get_matches_declaration = [{
    "type": "function",
    "name": "get_ingredients",
    "description": "Return if the value is an element return its formula and tell trough boolean value if it is an active ingredient or not. ",
    "parameters": {
        "type": "object",
        "properties": {
            "formula": {
                "type": "string",
                "description": "The chemical formula of an element",
            },
            "active": {
                "type": "boolean",
                "description": "True if elements is active.",
            },
        },
        "required": ["formula", "active"],
    },
}]

input_list = [
    {"role": "user", "content": "water"}
]

response = client.responses.create(
    model="gpt-5-nano",
    tools=get_matches_declaration,
    input=input_list,
)

example = response.output[1].arguments
print(example)
