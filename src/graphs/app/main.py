import io
import time

import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud


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
    bot_message = ""
    for character in message:
        bot_message += character
        time.sleep(0.02)
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": bot_message},
        ]
        yield new_history, None

    final_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": bot_message},
    ]
    wordcloud = create_wordcloud(final_history)
    yield final_history, wordcloud


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            wordcloud_output = gr.Image(label="Word Cloud of Chat Messages")

        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                label="Streaming Echo Chatbot", type="messages"
            )
            msg = gr.Textbox(label="Message")

            clear = gr.Button("Clear")

    msg.submit(echo_bot, [msg, chatbot], [chatbot, wordcloud_output])
    msg.submit(lambda _: gr.update(value=""), [], [msg], queue=False)
    clear.click(
        lambda: ([], None), None, [chatbot, wordcloud_output], queue=False
    )

demo.launch(share=True)
