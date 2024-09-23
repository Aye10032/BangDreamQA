from pathlib import Path
from typing import Optional

from langchain_core.documents import Document


def load_txt(file_path: str | bytes) -> Optional[Document]:
    if Path(file_path).stat().st_size == 0:
        return None

    file_names = Path(file_path).stem.split('_')
    story_no = int(file_names[0])
    story_section = int(file_names[1])

    with open(file_path, 'r', encoding='utf-8') as f:
        title = f.readline().strip()
        subtitle = f.readline().strip()
        f.readline()
        file_str = f.read()

    doc = Document(
        page_content=file_str,
        metadata={"story_no": story_no, "story_section": story_section, "title": title, "subtitle": subtitle}
    )
    return doc


def main() -> None:
    print(load_txt('../Bang/2_8_wrong.txt'))


if __name__ == '__main__':
    main()
