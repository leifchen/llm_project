import tempfile

from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


def qa_agent(openai_api_key, memory, uploaded_file, question):
    model = ChatOpenAI(api_key=openai_api_key,
                       base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                       model='qwen-plus-2025-04-28')

    # 读取上传文件的内容
    file_content = uploaded_file.read()

    # 使用 tempfile 创建一个临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        # 将上传文件内容写入临时文件
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name  # 获取临时文件路径

    # 将上传文件的内容写入临时文件
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)

    # 使用PyPDFLoader加载临时文件
    loader = PyPDFLoader(temp_file_path)

    # 加载文档
    docs = loader.load()

    # 初始化文本分割器，用于将文档分割成较小的块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每个块的大小
        chunk_overlap=50,  # 块之间的重叠大小
        separators=["\n", "。", "！", "？", "，", "、", ""]  # 文本分割符
    )

    # 分割文档成小块
    texts = text_splitter.split_documents(docs)

    # 初始化嵌入模型
    embeddings_model = DashScopeEmbeddings(
        model="text-embedding-v3"
    )

    # 使用嵌入模型和分割的文本创建FAISS向量数据库
    db = FAISS.from_documents(texts, embeddings_model)

    # 创建检索器，用于从数据库中检索相关信息
    retriever = db.as_retriever()

    # 使用LLM模型、检索器和记忆组件创建对话检索链
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,  # 语言模型
        retriever=retriever,  # 信息检索器
        memory=memory  # 对话记忆组件
    )

    response = qa.invoke({"chat_history": memory, "question": question})
    return response
