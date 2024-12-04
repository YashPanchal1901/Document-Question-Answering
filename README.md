
# Document Question Answering

### Overview

This application provides an interface for users to ask questions about their uploaded documents. The system processes the uploaded files, splits the content into manageable chunks, stores them in a FAISS vector store, and uses a conversational AI chain to generate responses.

### Features
- Supports **PDF**, **CSV**, and **TXT** file uploads.
- Processes documents using **LangChain** libraries.
- Leverages a **retrieval-augmented generation (RAG)** model for question answering.
- Includes memory to maintain conversational context.
- Interactive chat interface with custom HTML templates for user and bot messages.

---

### Prerequisites

1. **Python Version**: Python 3.9 or above.
2. **Packages**: Install the required libraries listed in the requirements section.
3. **HuggingFace API Key**: Obtain an API key from [HuggingFace](https://huggingface.co/settings/tokens).
4. **Logos**: Ensure that you have the image files for the logos:
   - `attachment_73051350.jpeg`
   - `Q&A logo.jfif`

---

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YashPanchal1901/Document-Question-Answering
   cd <DocumentQA>
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Linux/Mac
   venv\Scripts\activate          # On Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Add your HuggingFace API Key to a `secrets.toml` file inside the `.streamlit` folder:
   ```
   [secrets]
   HUGGINGFACEHUB_API_KEY = "your_huggingface_api_key"
   ```

---

### How to Run the Application

1. **Start the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

2. **Upload Documents**:
   - Use the sidebar to upload PDF, CSV, or TXT files.
   - Click on **Process** to process the documents.

3. **Ask Questions**:
   - Type your questions into the input box and interact with the chatbot.

---

### Project Structure

```plaintext
.
├── app.py                  # Main Streamlit application script
├── requirements.txt        # List of required Python packages
├── Q&A logo.jfif           # Bot logo
├── attachment_73051350.jpeg # Company or project logo
└── .streamlit/
    └── secrets.toml        # HuggingFace API key configuration
```

---

### Key Libraries and Tools

1. **Streamlit**: For building the web interface.
2. **LangChain**: Framework for conversational AI and document processing.
3. **FAISS**: Vector store for efficient document retrieval.
4. **HuggingFace Transformers**: For embeddings and language models.
5. **PIL**: Image processing for adding logos.

---

### How It Works

1. **Document Loading**:
   - The application loads supported documents using LangChain document loaders.
   - Files are temporarily stored for processing.

2. **Text Processing**:
   - Texts are split into smaller chunks using the `RecursiveCharacterTextSplitter`.

3. **Embeddings Creation**:
   - Converts document text into embeddings using the `HuggingFaceEmbeddings` model.

4. **Vector Store**:
   - Stores embeddings in a FAISS index for efficient similarity search.

5. **Conversational Chain**:
   - Creates a conversational chain using `ConversationalRetrievalChain` with context memory.

6. **Chat Interface**:
   - Displays chat messages using HTML templates for user and bot messages.

---

### Customization

1. **Change Embedding Model**:
   - Update the `model_name` in the `HuggingFaceEmbeddings` initialization.

2. **Adjust Chunk Size**:
   - Modify `chunk_size` and `chunk_overlap` in the `get_text_chunks` function for different text splitting needs.

3. **Add More File Types**:
   - Extend the `LOADER_MAPPING` dictionary with additional file loaders.

---

### Troubleshooting

1. **API Key Errors**:
   - Ensure the `HUGGINGFACEHUB_API_KEY` is correctly set in the `secrets.toml` file.

2. **Unsupported File Format**:
   - Ensure your file type is listed in `LOADER_MAPPING`.

3. **Missing Logos**:
   - Add `attachment_73051350.jpeg` and `Q&A logo.jfif` to the project directory.

---

### Future Enhancements

- **Multiple Language Support**: Enable support for questions in different languages.
- **Advanced Memory Mechanisms**: Incorporate more complex memory types like `ConversationSummaryMemory`.
- **Improved UI**: Add more intuitive features for document management and chat interactions.

---
