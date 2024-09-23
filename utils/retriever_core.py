from typing import Optional, Literal

from langchain.retrievers.multi_query import LineListOutputParser
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore
from loguru import logger
from pydantic import Field

from utils.embedding_core import BgeReranker
from utils.model_core import load_llm, load_embedding, load_reranker

GENERATE_QUESTION = """你是一名AI助手，现在你被给予了一个问题。
你的任务是通过对用户问题生成多个视角，帮助用户克服基于距离的相似性搜索的一些局限性。
你需要生成3个与之相近但不相同的问题以便从向量数据库中检索相关文档。
对于生成的问题，使用换行符隔开。
原问题: {question}
"""


def unique_doc(docs: list[Document]) -> list[Document]:
    result = []
    for doc in docs:
        if doc not in result:
            result.append(doc)

    return result


class RerankRetriever(BaseRetriever):
    vectorstore: VectorStore
    reranker: BgeReranker

    multi_query: bool = False
    llm_chain: Optional[Runnable] = None

    search_type: Literal['mmr', 'similarity'] = 'similarity'
    search_kwargs: dict = Field(default_factory=dict)

    top_k: int = 8

    def generate_queries(
            self,
            question: str,
            run_manager: CallbackManagerForRetrieverRun
    ) -> list[str]:
        response = self.llm_chain.invoke(
            {"question": question}, config={"callbacks": run_manager.get_child()}
        )

        return response

    async def agenerate_queries(
            self, question: str, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[str]:
        response = await self.llm_chain.ainvoke(
            {"question": question}, callbacks=run_manager.get_child()
        )

        return response

    def retrieve_documents(
            self,
            queries: list[str],
            run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        documents = []
        for query in queries:
            if self.search_type == 'mmr':
                docs: list[Document] = self.vectorstore.max_marginal_relevance_search(query, **self.search_kwargs)
            else:
                docs: list[Document] = self.vectorstore.similarity_search(query, **self.search_kwargs)
            documents.extend(docs)

        return documents

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.multi_query:
            queries = self.generate_queries(query, run_manager)
            queries.append(query)
            docs = self.retrieve_documents(queries, run_manager)
            docs = unique_doc(docs)
        else:
            if self.search_type == 'mmr':
                docs: list[Document] = self.vectorstore.max_marginal_relevance_search(query, **self.search_kwargs)
            else:
                docs: list[Document] = self.vectorstore.similarity_search(query, **self.search_kwargs)

        logger.info(f'retrieve {len(docs)} documents, reranking...')

        try:
            rerank_docs = self.reranker.compress_documents(docs, query)[:self.top_k]

            return rerank_docs
        except Exception as e:
            logger.error(f'catch exception {e} while check rerank')


def load_retriever() -> BaseRetriever:
    embedding = load_embedding(
        'BAAI/bge-m3',
        encode_kwargs={
            "normalize_embeddings": True
        },
        local_load=True,
        model_path='../data/model/BAAI/bge-m3'
    )
    vec_store = Chroma(
        collection_name="bang_dream",
        embedding_function=embedding,
        persist_directory="../data/chroma_db",
    )
    reranker = load_reranker(
        'BAAI/bge-reranker-v2-m3',
        encode_kwargs={
            "normalize": True
        },
        local_load=True,
        model_path='../data/model/BAAI/bge-reranker-v2-m3'
    )

    retriever_llm = load_llm()
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template=GENERATE_QUESTION,
    )
    parser = LineListOutputParser()
    llm_chain = query_prompt | retriever_llm | parser

    retriever = RerankRetriever(
        vectorstore=vec_store,
        reranker=reranker,
        multi_query=True,
        llm_chain=llm_chain,
        search_kwargs={"k": 10}
    )
    return retriever


def main() -> None:
    retriever = load_retriever()
    result = retriever.invoke("摩卡和兰是什么关系？")
    print(result)


if __name__ == '__main__':
    main()
