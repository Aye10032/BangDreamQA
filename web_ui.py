import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from utils.chat_core import answer

st.set_page_config(
    page_title='邦邦我',
    layout='centered',
    menu_items={
        'Report a bug': 'https://github.com/Aye10032/BangDreamQA/issues',
        'About': 'https://github.com/Aye10032/BangDreamQA'
    }
)

st.subheader('BangDream剧情问答')

history = StreamlitChatMessageHistory(key="chat_messages")

chat_container = st.container(height=700, border=False)
with chat_container:
    for message in history.messages:
        # icon = 'logo.png' if message['role'] != 'user' else None
        with st.chat_message(message.type):
            st.markdown(message.content)

if prompt := st.chat_input('请输入问题'):
    chat_container.chat_message('human').write(prompt)

    response = answer(history, 'data', prompt)
    history.add_user_message(prompt)
    result = chat_container.chat_message('assistant').write_stream(response)

    history.add_ai_message(result)
