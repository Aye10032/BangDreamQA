import os
from uuid import uuid4

from langchain_chroma import Chroma
from loguru import logger
from tqdm import tqdm

from utils.data_loader import load_txt
from utils.embedding_core import BgeM3Embeddings


def init_vector_db() -> Chroma:
    os.makedirs('data/model', exist_ok=True)

    embedding = BgeM3Embeddings(
        model_name='BAAI/bge-m3',
        use_fp16=True,
        encode_kwargs={
            'normalize_embeddings': True
        },
        local_load=True,
        local_path='./data/model/BAAI/bge-m3'
    )

    vector_store = Chroma(
        collection_name="bang_dream",
        embedding_function=embedding,
        persist_directory="./data/chroma_db",
    )

    return vector_store


def load_files(base_path: str | bytes, batch_size: int) -> None:
    """
    Loads files from the specified base path and adds them to the vector store in batches.

    :param base_path: The base path to the directory containing the files.
    :param batch_size: The number of documents to process in each batch.

    :return: None
    """
    vector_store = init_vector_db()
    logger.info('embedding model loaded')

    docs = []
    for root, _, files in os.walk(base_path):
        if not files:
            continue

        for _file in tqdm(files, total=len(files)):
            doc = load_txt(os.path.join(root, _file))
            if doc:
                docs.append(doc)
                if len(docs) >= batch_size:
                    uuids = [str(uuid4()) for _ in range(len(docs))]
                    vector_store.add_documents(docs, ids=uuids)
                    docs = []

    if docs:
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(docs, ids=uuids)

    logger.info('done')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='init vector database')
    parser.add_argument(
        '--data',
        '-D',
        required=True,
        type=str,
        help='Path to the data directory'
    )
    parser.add_argument(
        '--batch_size',
        '-B',
        required=False,
        type=int,
        default=10,
        help='Document batch size for add to vector database'
    )

    args = parser.parse_args()

    load_files(args.data, args.batch_size)
