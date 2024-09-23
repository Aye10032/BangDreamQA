from operator import itemgetter

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from utils.model_core import load_llm
from utils.retriever_core import load_retriever
from utils.tool_core import VecstoreSearchTool


def answer(
        _chat_history: BaseChatMessageHistory,
        db_path: str | bytes,
        question: str
):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content='你是一名AI助手，当用户向你提问时，你会得到一些游戏《Bang Dream!》的剧情片段。你需要根据这些剧情回答用户的问题。'
                        '在回答时，最好指出具体依据于哪一（几）期活动的哪个章节。'
                        '例如：根据第4期活动《里美的赠礼之歌》中第3话: 能让人恢复精神的东西，可知多惠说录音棚里面时不时会传出毛骨悚然的呜呜声'),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{stories}\n\n根据以上相关剧情，回答我的问题：{input}')
        ]
    )

    llm = load_llm()
    retriever = load_retriever(db_path)
    search_tool = VecstoreSearchTool(retriever=retriever)
    messages = _chat_history.messages.copy()

    retrieval_chain = (
            {
                "chat_history": itemgetter("chat_history"),
                "input": itemgetter("question"),
                "stories": itemgetter("question") | search_tool
            }
            | prompt
            | llm
    )

    result = retrieval_chain.stream({
        'chat_history': messages,
        'question': question
    })

    return result
