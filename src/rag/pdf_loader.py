
from pathlib import Path
from pypdf import PdfReader
from pypdf.errors import EmptyFileError, PdfReadError


def load_pdf(pdf_path):

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Validate empty files
    if pdf_path.stat().st_size == 0:
        print(f"WARNING: Skipping empty PDF: {pdf_path.name}")
        return []

    try:
        reader = PdfReader(str(pdf_path))

        documents = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            if text.strip():
                documents.append({
                    "text": text.strip(),
                    "source": pdf_path.name,
                    "page": page_number
                })

        if not documents:
            print(
                f"WARNING: No extractable text in {pdf_path.name}. "
                "It may be scanned or image-only."
            )
            return []

        return documents

    except (EmptyFileError, PdfReadError) as error:
        print(f"WARNING: Cannot read {pdf_path.name}: {error}")
        return []
