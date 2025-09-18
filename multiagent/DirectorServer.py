import random

from Director import graph

# config = {
#         "configurable":{
#             "thread_id" : random.randint(1,10000)
#             }
#     }
# #
# query = "给我一个冷笑话"
#
# res = graph.invoke({"messages":["乘法口诀是什么"]},
#                           config,
#                           stream_mode="values")
# print(res["messages"][-1].content)

import  gradio as gr
def process_input(text):
    config = {
        "configurable": {
            "thread_id": random.randint(1,1000)
        }
    }

    result = graph.invoke({"messages":[text]},config)
    return result["messages"][-1].content



with gr.Blocks() as demo:
    gr.Markdown("#langGraph Multi-agent")
    with gr.Row():
        with gr.Column():
            gr.Markdown(" ## 可以路线规划，对联，讲笑话，试一试吧")
            inputs_text = gr.Textbox(label="问题", placeholder="请输入你的问题", value="讲一个冷笑话")
            btn_start = gr.Button("start", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(label="output")
    btn_start.click(process_input, inputs=[inputs_text],outputs=[output_text])

demo.launch()


