from typing import Type, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field

from utils.data_loader import StoryType
from utils.retriever_core import RerankRetriever


class VecstoreSearchInput(BaseModel):
    query: str = Field(description="用于从向量数据库中召回长文本的搜索文本")


class VecstoreSearchTool(BaseTool):
    name: str = 'search_from_vecstore'
    description: str = '根据查询文本从向量数据库中搜索相关的知识。AI在回答问题的过程中，如遇到不明确的知识点或术语，可以调用此工具从数据库中进行查询以获取相关信息。'
    args_schema: Type[BaseModel] = VecstoreSearchInput
    return_direct: bool = False
    handle_tool_error: bool = True

    target_collection: str = 'bang_dream'
    retriever: RerankRetriever

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """从向量数据库中进行查询操作"""
        logger.info(f'Calling VecstoreSearchTool with query {query}')

        def format_docs(docs: list[Document]) -> str:
            formatted = []
            for doc in docs:
                story_type = doc.metadata['story_type']
                format_str = {
                    StoryType.MAIN: '主线剧情\n',
                    StoryType.BAND: '乐队剧情\n',
                    StoryType.STORY: f"第{doc.metadata['story_no']}期活动剧情\n"
                }.get(story_type, '')


                format_str += (
                    f"名称: {doc.metadata['title']}\n"
                    f"剧情章节: {doc.metadata['subtitle']}\n"
                    f"剧情内容: {doc.page_content.replace('~', r'\~')}\n"
                )
                formatted.append(format_str)

            return '\n\n==========================\n\n'.join(formatted)

        def retrieve_from_vecstore(_query: str) -> str:
            chain = self.retriever | RunnableLambda(format_docs)
            output = chain.invoke(_query)

            return output

        return retrieve_from_vecstore(query)
