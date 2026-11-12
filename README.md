🤖 MiniGPT Research Assistant

🔗 Live Demo: https://minigpt-research-assistant.streamlit.app/ 💻


📌 About the Project

MiniGPT Research Assistant is an AI-powered application that helps users
understand and explore PDF documents. Upload a PDF, ask questions in
natural language, and get answers based on relevant information
retrieved from the document.

The project combines Retrieval-Augmented Generation (RAG) with language
models. It also includes custom Transformer and attention
implementations using PyTorch to explore how language models work
internally.

✨ Key Features

-   📄 Upload and process PDF documents.
-   💬 Ask questions about uploaded documents.
-   🔎 Retrieve relevant passages using semantic search.
-   🧠 Generate answers using a language model and retrieved context.
-   📚 Display source document and page references.
-   📝 Summarize document content.
-   ⚙️ Implement attention and multi-head attention using PyTorch.
-   📊 Evaluate retrieval performance with a benchmark script.

🏗️ How It Works

1.  Upload a PDF document.
2.  Extract text and page information.
3.  Split the text into smaller chunks.
4.  Convert chunks into embeddings using Sentence Transformers.
5.  Store embeddings in a FAISS vector index.
6.  Retrieve relevant chunks for the user’s question.
7.  Send the question and retrieved context to the language model.
8.  Display the generated answer with source references.

🧰 Tech Stack

-   Python
-   PyTorch
-   Hugging Face Transformers
-   Sentence Transformers
-   FAISS
-   PyMuPDF
-   Streamlit
-   NumPy

📊 Evaluation

The project includes an evaluation script at src/rag/evaluate.py.

Reported results from the current 6-test development benchmark:

  Metric                         Result
  ---------------------------- --------
  Test cases                          6
  In-domain questions                 4
  Out-of-domain tests                 2
  Retrieval Hit Rate @ 3           100%
  Mean Reciprocal Rank (MRR)     1.0000

These results are from a small development test set and are not a
guarantee of production performance. The current evaluation output
reports a 0% hallucination-guard rate, so this feature needs further
testing and improvement.

Run the evaluation:

    python src/rag/evaluate.py

🚀 Run Locally

1. Clone the repository

    git clone https://github.com/Pradeepprajapat7/MiniGPT-Research-Assistant.git
    cd MiniGPT-Research-Assistant

2. Create a virtual environment

Windows PowerShell:

    python -m venv .venv
    .venv\Scripts\Activate.ps1

macOS / Linux:

    python -m venv .venv
    source .venv/bin/activate

3. Install dependencies

    pip install -r requirements.txt

If Streamlit is not included in requirements.txt, install it:

    pip install streamlit

4. Run the application

    streamlit run app.py

Open the local URL displayed in your terminal (usually
http://localhost:8501).

📁 Project Structure

    MiniGPT-Research-Assistant/
    ├── checkpoints/       # Model checkpoints
    ├── data/              # Input documents and evaluation data
    ├── src/
    │   ├── rag/           # PDF processing and RAG pipeline
    │   ├── attention.py   # Scaled dot-product attention
    │   ├── multi_head_attention.py
    │   ├── bpe_tokenizer.py
    │   └── pretrained_model.py
    ├── storage/
    │   └── rag/           # FAISS index and metadata
    ├── tests/             # Test files
    ├── app.py             # Streamlit application
    ├── requirements.txt
    └── README.md

🎯 Key Learnings

-   Understanding Transformer architecture and attention mechanisms.
-   Implementing attention components with PyTorch.
-   Integrating pretrained language models.
-   Building an end-to-end RAG pipeline.
-   Creating embeddings and searching with FAISS.
-   Displaying document sources for more traceable answers.
-   Evaluating retrieval quality using standard ranking metrics.

🔮 Future Improvements

-   Expand the evaluation dataset.
-   Improve out-of-domain question detection.
-   Strengthen hallucination checks.
-   Add conversation memory and multi-document comparison.
-   Expand automated testing.

👨‍💻 Author

Pradeep Prajapat

-   GitHub: https://github.com/Pradeepprajapat7
-   LinkedIn: https://linkedin.com/in/pradeep-prajapat-7p

------------------------------------------------------------------------

⭐ If you find this project interesting, feel free to explore the source
code and try the live demo.
