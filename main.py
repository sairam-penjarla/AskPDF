
from src.reader import PDFReader

if __name__ == "__main__":
    PDF_PATH = "statement.pdf"
    user_query = "What is the statement about?"

    reader = PDFReader(pdf_path=PDF_PATH)
    result = reader.get_result(user_query=user_query)
    print(result)
