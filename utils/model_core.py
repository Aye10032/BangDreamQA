import os
from typing import Optional, Any

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from utils.embedding_core import BgeM3Embeddings, BgeReranker


@st.cache_resource(show_spinner=f'Loading embedding model...')
def load_embedding(
        model_name: str,
        *,
        fp16: bool = True,
        device: Optional[str] = None,
        encode_kwargs: dict[str, Any] = None,
        local_load: bool = False,
        model_path: Optional[str | bytes] = None
) -> BgeM3Embeddings:
    """
    Loads an embedding model with the specified configuration.

    :param model_name: The name of the model to load.
    :param fp16: Whether to use 16-bit floating point precision. Defaults to True.
    :param device: The device to load the model on (e.g., 'cpu', 'cuda'). Defaults to None.
    :param encode_kwargs: Additional keyword arguments for the encoding process. Defaults to None.
    :param local_load: Whether to load the model from a local path. Defaults to False.
    :param model_path: The local path to the model if local_load is True. Defaults to None.

    :return: The loaded embedding model.
    """
    embedding = BgeM3Embeddings(
        model_name=model_name,
        use_fp16=fp16,
        device=device,
        encode_kwargs=encode_kwargs,
        local_load=local_load,
        local_path=model_path
    )

    return embedding


@st.cache_resource(show_spinner=f'Loading rerank model...')
def load_reranker(
        model_name: str,
        *,
        fp16: bool = True,
        device: Optional[str] = None,
        encode_kwargs: dict[str, Any] = None,
        local_load: bool = False,
        model_path: Optional[str | bytes] = None
) -> BgeReranker:
    """
    Loads a reranker model with the specified configuration.

    :param model_name: The name of the model to load.
    :param fp16: Whether to use 16-bit floating point precision. Defaults to True.
    :param device: The device to load the model on (e.g., 'cpu', 'cuda'). Defaults to None.
    :param encode_kwargs: Additional keyword arguments for the encoding process. Defaults to None.
    :param local_load: Whether to load the model from a local path. Defaults to False.
    :param model_path: The local path to the model if local_load is True. Defaults to None.

    :return: The loaded reranker model.
    """
    reranker = BgeReranker(
        model_name=model_name,
        use_fp16=fp16,
        device=device,
        encode_kwargs=encode_kwargs,
        local_load=local_load,
        local_path=model_path
    )

    return reranker


@st.cache_resource(show_spinner='Loading GLM4-flash...')
def load_llm(temperature: float = 0.3) -> ChatOpenAI:
    load_dotenv()
    api_key = os.getenv('GML_KEY')

    llm = ChatOpenAI(
        model='glm-4-flash',
        openai_api_base='https://open.bigmodel.cn/api/paas/v4/',
        openai_api_key=api_key,
        temperature=temperature,
    )

    return llm
