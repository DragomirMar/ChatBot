# Knowledge Graph enriched Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot that uses a knowledge graph to enrich its answers with structured entity and relationship information.

The chatbot allows users to upload documents or provide URLs and combines vector-based retrieval with knowledge graph reasoning to generate more accurate and informative responses. 


## Technologies

It was built with:
- Python
- [Ollama](https://ollama.com) (Model: llama3.1:8B)
- [Streamlit](https://streamlit.io/)
- [MongoDB](https://www.mongodb.com/)
- [Chroma](https://www.trychroma.com/)


## Prerequisites

Before getting started, ensure you have the following installed on your machine:

- Python 3.11 or higher
- Ollama (for running LLMs locally)
- MongoDB (as database to get nodes and relationships from)

## Setup and installation

Follow the steps below to set up the project locally:

### 1. Install Ollama
Download and install Ollama, which is required to run models locally.

- For **Windows** and **Mac** use the link: https://ollama.com/download

- For **Ubuntu** use the command:
```curl -fsSL https://ollama.com/install.sh | sh ```

#### Pull LLaMA 3.1
Once Ollama is installed, pull the LLaMA 3.1 model (or another model you prefer):
```bash
ollama pull llama3.1
```

### 2. Install MongoDB
#### For Windows & macOS:
Download and install MongoDB:
- **Windows**: https://www.mongodb.com/docs/v7.0/tutorial/install-mongodb-on-windows/
- **macOS**: https://www.mongodb.com/docs/v7.0/tutorial/install-mongodb-on-os-x/

After installation, open a terminal and create MongoDB Data Directory (first time only):
```bash
mkdir -p ~/data/db
```

Start MongoDB:
```bash
mongod --dbpath ~/data/db
```

Keep this terminal window open while using the application.

---

### For Ubuntu:

**Step 1: Install prerequisites**
```bash
sudo apt-get install gnupg curl
```

**Step 2: Import MongoDB public GPG key**
```bash
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg \
   --dearmor
```

**Step 3: Create MongoDB repository list file**
```bash
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
```

**Step 4: Reload package database**
```bash
sudo apt-get update
```

**Step 5: Install MongoDB Community Server**
``` bash
sudo apt install -y mongodb-org
```

**Step 6: Start MongoDB service**
``` bash 
sudo systemctl start mongod
```

**Step 7: Verify MongoDB is running**
``` bash 
sudo systemctl status mongod
```

### 3. Clone this repo
```bash
git clone https://github.com/DragomirMar/ChatBot.git
cd ChatBot
```

### 4. Create Virtual Environment
```bash
### Navigate to the project folder
cd Knowledge_Graph

### Create a virtual environment
python -m venv venv

### Activate it (macOS/Linux)
source venv/bin/activate
```
**Note:** On many Linux systems, `python` may still point to Python 2 so use  `python3` as a command.

### 5. Install Required Libraries
```bash
pip install -r requirements.txt 
```

**Note:** If pip is still using the global environment, run:
```bash
unalias pip
```

### 6. Configure Environment Variables

The application uses environment variables to manage database configuration and separate test/production environments.

## Configuration

At the moment, database connections are defined directly in the source code. To change them, open knowledge_graph_retriever.py and change database connection accordingly.

# Run the application
### 1. Start Ollama
Start Ollama either through the interface or through the terminal:
```bash
ollama serve
```

### 2. Ensure MongoDB is Running
**Windows, macOS**

Make sure MongoDB is still running in another terminal:
```bash
mongod --dbpath ~/data/db
```

### Ubuntu

Check if MongoDB is running:
``` bash
sudo systemctl status mongod
```

Start if not running:
``` bash
sudo systemctl start mongod
```

### 3. Activate Virtual Environment
**macOS/Linux**
If not already activated:
```bash
source venv/bin/activate
```

### 4. Run the app
Navigate to the source folder (cd src) and run:
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8502`


# Testing

The application includes a comprehensive test suite to ensure reliability.

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test level
pytest tests/unit -v
pytest tests/integration -v
pytest tests/system -v

# Run specific test file
pytest tests/unit/test_kg_retriever.py -v

# Run specific test
pytest tests/unit/test_kg_retriever.py::TestKnowledgeGraphRetrieverUnit::test_fuzzy_match_exact -v
```



### Integration tests
For integration tests you can determine which MongoDB database to be used - mock or a real one.
```bash
# Uses mongomock (in-memory, no MongoDB needed)
pytest tests/integration -v

# Requires MongoDB running on localhost:27017
USE_REAL_MONGODB=true pytest tests/integration -v

# Or with custom URI
USE_REAL_MONGODB=true MONGODB_URI=mongodb://localhost:27017/ pytest tests/integration -v
```
