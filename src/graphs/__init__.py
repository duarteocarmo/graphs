import instructor
from litellm import completion
from openai import OpenAI

CLIENT = OpenAI()
MODEL = "gpt-4o-mini"
INSTRUCTOR_CLIENT = instructor.from_litellm(completion)
