from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI


def get_chat_response(prompt, memory, openai_api_key):
    model = ChatOpenAI(api_key=openai_api_key,
                       base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                       model='qwen-plus-2025-04-28')
    chain = ConversationChain(llm=model, memory=memory)

    response = chain.invoke({"input": prompt})
    return response["response"]
