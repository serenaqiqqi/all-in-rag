import os

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

markdown_path = '../../data/C1/markdown/easy-rl-chapter1.md'

loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

llm = ChatOpenAI(
    model='deepseek-chat',
    temperature=0.7,
    max_tokens = 4096,
    api_key = os.getenv('DEEPSEEK_API_KEY'),
    base_url ='https://api.deepseek.com'
)

question = "文中举了哪些例子？"

retrieved_docs = vectorstore.similarity_search(question,k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

answer = llm.invoke(prompt.format(question=question,context = docs_content))
print(answer)

