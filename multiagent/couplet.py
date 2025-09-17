import os

from langchain_community.embeddings import DashScopeEmbeddings
import redis


if not os.environ.get ("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = load_key("DASHSCOPE_API_KEY")



embedding_model = DashScopeEmbeddings(model="text-embedding-v1")

redis_url = "redis://localhost:6379"

redis_client = redis.from_url(redis_url)

# print(redis_client.ping())

from langchain_redis import RedisConfig, RedisVectorStore
config = RedisConfig(
    index_name="couplet",
    redis_url = redis_url
)
vector_store = RedisVectorStore(embedding_model,config= config)

lines = []
with open(r"D:\projects\langchain\multiagent\resource\couplettest.csv", "r", encoding="utf-8") as file:
    for line in file:
        print(line)
        lines.append(line)
vector_store.add_texts(lines)