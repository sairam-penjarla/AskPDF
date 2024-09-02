import pandas as pd
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm

from utils import read_pdf

class PDFReader:
    def __init__(self, pdf_path):
        self.corpus = [read_pdf(pdf_path)]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""],
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="thenlper/gte-small",
            multi_process=True,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.create_pipeline()

    def create_pipeline(self):
        df = pd.DataFrame({"Text": self.corpus})

        RAW_KNOWLEDGE_BASE = [LangchainDocument(page_content=str(row["Text"]), metadata={"source": "None"})
                              for index, row in df.iterrows()]
        docs_processed = []
        for doc in RAW_KNOWLEDGE_BASE:
            docs_processed += self.text_splitter.split_documents([doc])

        tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
        prompt_in_chat_format = [
            {"role": "system",
             "content": """Using the information contained in the context,
                           give a comprehensive answer to the question.
                           Respond only to the question asked, response should be concise and relevant to the question.
                           If the answer cannot be deduced from the context, do not give an answer."""},
            {"role": "user",
             "content": """Context:
                           {context}
                           ---
                           Now here is the question you need to answer.

                           Question: {question}"""}
        ]

        self.RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )
        self.KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
            docs_processed, self.embedding_model, distance_strategy=DistanceStrategy.COSINE
        )
        model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

        self.READER_LLM = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        )

    def get_result(self, user_query):
        retrieved_docs = self.KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
        retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
        context = "\nExtracted documents:\n"
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

        final_prompt = self.RAG_PROMPT_TEMPLATE.format(question=user_query, context=context)
        answer = self.READER_LLM(final_prompt)[0]["generated_text"]
        return answer
