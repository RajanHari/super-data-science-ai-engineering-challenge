import gradio as gr


def respond(message, history):
    response = f"You said: {message}\
        \n And I say I love learning AI Engineering!"
    return response

gr.ChatInterface(fn=respond).launch()
