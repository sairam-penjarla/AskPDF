import PyPDF2

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        corpus = []
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            line = page.extract_text()
            corpus.append(line)
        return " ".join(corpus)
