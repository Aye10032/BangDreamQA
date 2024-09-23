from enum import IntEnum
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document


class StoryType(IntEnum):
    MAIN = 0
    BAND = 1
    STORY = 2


def load_txt(file_path: str | bytes) -> Optional[Document]:
    if Path(file_path).stat().st_size == 0:
        return None

    file_names = Path(file_path).stem.split('_')
    story_no = int(file_names[0])

    folder_name = Path(file_path).parent.name
    match folder_name:
        case 'mainstory':
            story_type = StoryType.MAIN
        case 'band':
            story_type = StoryType.BAND
        case _:
            story_type = StoryType.STORY

    with open(file_path, 'r', encoding='utf-8') as f:
        title = f.readline().strip()
        subtitle = f.readline().strip()
        f.readline()
        file_str = f.read()

    doc = Document(
        page_content=file_str,
        metadata={"story_no": story_no, "title": title, "subtitle": subtitle, "story_type": story_type.value}
    )
    return doc


def main() -> None:
    print(load_txt('../data/bang/mainstory/2.txt'))


if __name__ == '__main__':
    main()
