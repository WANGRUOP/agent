import random

from Director import graph

config = {
        "configurable":{
            "thread_id" : random.randint(1,10000)
            }
    }

query = "给我一个冷笑话"

res = graph.invoke({"messages":["乘法口诀是什么"]},
                          config,
                          stream_mode="values")
print(res["messages"][-1].content)