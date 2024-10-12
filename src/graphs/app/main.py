import inspect
import io
from inspect import Parameter

import gradio as gr
import matplotlib.pyplot as plt
from litellm import completion
from PIL import Image
from pydantic import create_model
from wordcloud import WordCloud

from graphs import MODEL


def schema(f) -> dict:
    kw = {
        n: (o.annotation, ... if o.default == Parameter.empty else o.default)
        for n, o in inspect.signature(f).parameters.items()
    }
    s = create_model(f"Input for `{f.__name__}`", **kw).schema()

    return {
        "type": "function",
        "function": {
            "name": f.__name__,
            "description": f.__doc__,
            "parameters": s,
        },
    }


def create_wordcloud(messages):
    text = " ".join([msg["content"] for msg in messages])
    wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    plt.close()

    img = Image.open(img_buffer)
    return img


def echo_bot(message, history):
    history.append(
        {
            "role": "user",
            "content": message,
        }
    )

    bot_message = ""
    for chunk in completion(
        model=MODEL,
        messages=history,
        stream=True,
    ):
        if content := chunk.choices[0].delta.content:
            bot_message += content

        new_history = history + [
            {"role": "assistant", "content": bot_message},
        ]
        yield new_history, None

    history = history + [
        {"role": "assistant", "content": bot_message},
    ]
    wordcloud = create_wordcloud(history)
    yield history, wordcloud


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            wordcloud_output = gr.Image(label="Word Cloud of Chat Messages")

        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                label="Streaming Echo Chatbot", type="messages", height=500
            )
            msg = gr.Textbox(label="Message")

            clear = gr.Button("Clear")

    msg.submit(echo_bot, [msg, chatbot], [chatbot, wordcloud_output])
    msg.submit(lambda _: gr.update(value=""), [], [msg], queue=False)
    clear.click(
        lambda: ([], None), None, [chatbot, wordcloud_output], queue=False
    )

demo.launch(share=True)
