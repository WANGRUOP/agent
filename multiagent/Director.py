import asyncio
from http.client import responses
from typing import TypedDict, Annotated
from operator import add
from xmlrpc.server import MultiPathXMLRPCServer
from langchain_redis import RedisConfig, RedisVectorStore

from langchain.chains.summarize.refine_prompts import prompt_template
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.checkpoint.memory import InMemorySaver
import os
from langchain_community.chat_models import ChatTongyi
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

nodes = ["supervisor", "travel", "couplet", "joke", "other"]

llm = ChatTongyi (
    model = "qwen-plus",
    api_key = os.getenv("DASHSCOPE_API_KEY")
)

class State(TypedDict):
    messages: Annotated[list[AnyMessage],add]
    type: str


def other_node(state:State):
    print(">>>other_node")
    writer = get_stream_writer()
    writer({"node",">>>>other_node"})
    
    return{"messages":[HumanMessage(content="我暂时无法回答这个问题")],"type":"other"}

def supervisor_node(state:State):
    print(">>>supervisor_node")
    writer = get_stream_writer()
    writer({"node",">>>>supervisor_node"})
    #根据问题进行分类，将分类结果保存至type
    prompt = """你是一个专业的客服助手，负责对用户户的问题进行准确分类，然后将问题分给其他agent进行处理。
                如果用户问题与旅游路线规划相关，请返回travel，
                如果用户希望讲一个笑话，请返回joke，
                如果用户想对一个对了，请返回couplet，
                如果是其他问题，请返回other，
                除了以上选项，不返回其他任何内容
"""

    prompts = [
        {"role" :"system", "content" : prompt},
        {"role" :"user", "content" : state["messages"][0]}
    ]

    #已经有type属性，直接返回
    if "type" in state:
        writer({"supervisor_step":f"已获得{state['type']}智能体处理结果"})
        return {"type": END}
    else:
        response = llm.invoke(prompts)
        typeRes = response.content
        writer({"supervisor_step":f"问题分类结果{typeRes}"})

        if typeRes in nodes:
            return {"type": typeRes}
        else:
            raise ValueError("type is not in (travel, couplet, joke, other)")



def travel_node(state:State):
    print(">>>travel_node")
    writer = get_stream_writer()
    writer({"node":">>>>travel_node"})

    system_prompt = "你是一个专业的旅行规划助手，可以使用可用工具（如地图服务）来帮助用户规划路线。你的回答应基于工具的调用结果，并尽可能简洁明了（不超过100字"

    prompts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["messages"][0]}
    ]

    # 1c655eead7f86c22a5377a02674770fb
    # 高德地图mcp配置信息
    client = MultiServerMCPClient(
        {
            "amap-maps":{
                "command" : "npx",
                "args" :[
                    "-y",
                    "@amap/amap-maps-mcp-server"
                ],
                "env":{
                    "AMAP_MAPS_API_KEY": "1c655eead7f86c22a5377a02674770fb"
                },
                "transport": "stdio"
            }
        }
    )

    tools = asyncio.run(client.get_tools())
    agent = create_react_agent(
        model = llm,
        tools = tools,
    )

    response = agent.invoke({"messages": prompts})

    return {"messages": [HumanMessage(content=response["messages"][-1].content)], "type": "travel"}

def joke_node(state:State):
    print(">>>joke_node")
    writer = get_stream_writer()
    writer({"node":">>>>joke_node"})

    system_prompt = "你是一个笑话大师，根据用户问题，写一个50字左右的笑话"

    prompts = [
        {"role" :"system", "content" : system_prompt},
        {"role" :"user", "content" : state["messages"][0]}
    ]

    response = llm.invoke(prompts)

    return{"messages":[HumanMessage(content = response.content)],"type":"joke"}

def couplet_node(state:State):
    print(">>>couplet_node")
    writer = get_stream_writer()
    writer({"node":">>>>couplet_node"})

    prompt_template = ChatPromptTemplate.from_template([
        ("system","""
        你是一个专业的对联大师，可以根据用户给出的上联，回答一个下联。
        回答时，可以参考下面对联。
        参考对联：
               {samples}
        用中文回答"""),
        ("user", "{text}")
    ])
    query = state["message"][0]


    embedding_model = DashScopeEmbeddings(model="text-embedding-v1")

    redis_url = "redis://localhost:6379"

    config = RedisConfig(
        index_name="couplet",
        redis_url=redis_url
    )
    vector_store = RedisVectorStore(embedding_model, config=config)

    samples=[]

    scored_results = vector_store.similarity_search_with_score(query, k=10)
    for doc, score in scored_results:
        samples.append(doc.page_content)

    prompt = prompt_template.invoke({"sample":samples,"text":query})
    response = llm.invoke(prompt)

    return{"messages":[HumanMessage(content=response.content)],"type":"couplet"}


def routing_func(state:State):
    if state["type"] == "travel":
        return "travel_node"
    elif state["type"] == "joke":
        return "joke_node"
    elif state["type"] == "couplet":
        return "couplet_node"
    elif state["type"] == END:
        return END
    else:
        return "other_node"
    



#构建图
builder = StateGraph(State)

#添加节点
builder.add_node("supervisor_node",supervisor_node)
builder.add_node("travel_node",travel_node)
builder.add_node("joke_node",joke_node)
builder.add_node("couplet_node",couplet_node)
builder.add_node("other_node",other_node)

#添加edge
builder.add_edge(START,"supervisor_node")
builder.add_conditional_edges("supervisor_node",routing_func,["travel_node","joke_node","couplet_node","other_node",END])
builder.add_edge("travel_node","supervisor_node")
builder.add_edge("joke_node","supervisor_node")
builder.add_edge("couplet_node","supervisor_node")
builder.add_edge("other_node","supervisor_node")

#构架graph
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


#测试代码
if __name__ == "__main__":
    config = {
        "configurable":{
            "thread_id" :"1"
            }
    }

    for chunk in graph.stream({"messages":["给我对一个对联，上联是上山打老虎，请出下联"]},
                          config,
                          stream_mode="values"):
        print(chunk)


# res = graph.invoke({"messages":["乘法口诀是什么"]},
#                           config,
#                           stream_mode="values")
# print(res["messages"][-1].content)