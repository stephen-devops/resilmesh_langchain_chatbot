# Run a ReAct application ChatBot

## Requirements ##

- Python 3.11+
- Node.js v18.20.8+
- Docker & Docker Compose (latest)

## 1. Backend Setup ##
### Python dependencies ###


```bash
cd resilMesh5_2_ChatBot
pip install -r requirements.txt
```

## 2. Frontend Setup

```bash
cd frontend
npm install .
```

## 3. Starting Docker ##

```bash
sudo docker-compose up -d
```

To access OpenSearch cluster, open https://localhost:9200/
then enter the credentials 
admin: "admin"
password: "2022NovemberOpenSearch"

## 4. Running the app

### Backend ###

```bash
cd resilMesh5_2_ChatBot
uvicorn backend.convAgentFastAPIServer:app --reload --host 0.0.0.0 --port 8000
```

### Frontend ###

```bash
cd frontend
npm start
```