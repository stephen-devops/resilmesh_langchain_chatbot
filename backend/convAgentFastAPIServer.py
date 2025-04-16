import os
import re
import json
import requests
from datetime import datetime
from typing import Optional, List, Union, Dict

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from opensearchpy import OpenSearch
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_community.vectorstores import OpenSearchVectorSearch

from ctiKnowledgeSources.ctiKnowledgeRetrieval import CTIIntegration
from openSearchEmbedding.opensearch_embedding import OpenSearchEmbeddings

# --- Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "../config/config_env")
load_dotenv(dotenv_path)

opensearch_url = os.getenv("OPENSEARCH_URL")
model_id = os.getenv("MODEL_ID")
ioc_index_name = os.getenv("CTI_INDEX_NAME")
username = os.getenv("OPENSEARCH_USERNAME")
password = os.getenv("OPENSEARCH_PASSWORD")
CA_CERTS_PATH = os.getenv("CA_CERTS_PATH")
AUTH = (username, password)

# import text-embedding model from OpenSearch to extract embeddings
predict_url = f"{opensearch_url}/_plugins/_ml/models/{model_id}/_predict"

# === MEMORY ===
chat_history = []


def format_chat_history(my_chat_history):
    return "\n".join([f"{role}: {text}" for role, text in my_chat_history])


def ingest_ioc_data_report(ingest_query, ingest_report):
    opensearch_client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_auth=AUTH,
        http_compress=True,
        use_ssl=True,
        verify_certs=False,
        ca_certs=CA_CERTS_PATH
    )

    doc = {
        "query": ingest_query,
        "report": ingest_report,
        "timestamp": datetime.utcnow()
    }

    response = opensearch_client.index(
        index="my_cti_ioc_report_repo",
        body=doc
    )

    doc_id = response["_id"]
    print(f"Indexed Document ID: {doc_id}")

    with open("my_ioc_doc_ids.txt", "a") as f:
        f.write(doc_id + "\n")


# === EMBEDDING + RETRIEVER ===
def get_opensearch_embedding(text):
    headers = {"Content-Type": "application/json"}
    payload = {
        "text_docs": [text],
        "return_number": True,
        "target_response": ["sentence_embedding"]
    }

    response = requests.post(
        predict_url,
        headers=headers,
        data=json.dumps(payload),
        auth=AUTH,
        verify=False
    )

    if response.status_code == 200:
        return response.json()["inference_results"][0]["output"][0]["data"]
    else:
        raise Exception(f"OpenSearch Inference Error: {response.text}")


embedding_function = OpenSearchEmbeddings(embedding_function=get_opensearch_embedding)

retriever = OpenSearchVectorSearch(
    opensearch_url=opensearch_url,
    index_name=ioc_index_name,
    embedding_function=embedding_function,
    text_field="report",
    use_ssl=True,
    verify_certs=False,
    ca_certs=CA_CERTS_PATH,
    http_auth=AUTH
)


# === CUSTOM LLM WRAPPER ===
class DeepSeekLLM(Runnable):
    react_api_url = "http://192.168.200.205:5964/react/"
    generate_api_url = "http://192.168.200.205:5964/generate_report/"

    def invoke(self, input: Union[str, Dict, PromptValue], config: Optional[dict] = None) -> str:
        if isinstance(input, dict):
            prompt = input.get("query", "")
        elif isinstance(input, str):
            prompt = input
        elif isinstance(input, PromptValue):
            prompt = input.to_string()
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")

        return self._call(prompt)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        cti_results = self._get_react_decision(prompt)
        report = self._get_llm_report(prompt, cti_results)
        return report if report else "No response from LLM."

    def _get_react_decision(self, react_query: str) -> dict:
        payload = {"query": react_query}
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.react_api_url, json=payload, headers=headers)
            if response.status_code == 200:
                react_decision = response.json()
            else:
                print(f"ReAct decision error: {response.status_code}, {response.text}")
                return {}

            action = react_decision.get("action", "")
            entities = react_decision.get("entities", {})
            return self._query_cti_sources(action, entities)

        except Exception as e:
            print(f"ReAct error: {e}")
            return {}

    def _query_cti_sources(self, action: str, entities: dict) -> dict:
        results = {}

        for entity_type, values in entities.items():
            if not values:
                continue

            values = [values] if isinstance(values, str) else values
            for value in values:
                if action == "search_opensearch":
                    results[value] = {"OPENSEARCH": CTIIntegration.query_opensearch(value)}
                elif action == "search_misp":
                    results[value] = {"MISP": CTIIntegration.query_misp(value)}
                elif action == "search_otx" and entity_type in ["ip", "domain", "hash", "url"]:
                    results[value] = {"OTX": CTIIntegration.query_otx(value, entity_type)}
                else:
                    results[value] = {
                        "OPENSEARCH": CTIIntegration.query_opensearch(value),
                        "MISP": CTIIntegration.query_misp(value),
                    }
                    if entity_type in ["ip", "domain", "hash", "url"]:
                        results[value]["OTX"] = CTIIntegration.query_otx(value, entity_type)

        return results

    def _get_llm_report(self, report_query: str, cti_results: dict) -> Optional[str]:
        headers = {"Content-Type": "application/json"}
        payload = {"query": report_query, "cti_results": cti_results or {}}
        try:
            response = requests.post(self.generate_api_url, json=payload, headers=headers)
            if response.status_code == 200 and response.headers.get("Content-Type") == "application/json":
                return response.json()
            return None
        except Exception as e:
            print(f"LLM report error: {e}")
            return None


# === LangChain Chain Setup ===
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a cybersecurity assistant."),
    ("user", "Context:\n{context}\n\nQuestion:\n{question}")
])

llm = DeepSeekLLM()
llm_runnable = RunnableLambda(lambda input: llm.invoke(input))

retrieval_chain = (
    {
        "context": lambda x: retriever.as_retriever().invoke(x["question"]),
        "chat_history": lambda x: format_chat_history(chat_history),
        "question": lambda x: x["question"]
    }
    | prompt_template
    | llm_runnable
)

# === FastAPI Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat_endpoint(data: QueryRequest):
    question = data.question
    raw_response = retrieval_chain.invoke({"question": question})
    response = clean_llm_output(raw_response)

    chat_history.append(("user", question))
    chat_history.append(("assistant", response))

    ingest_ioc_data_report(question, response)
    return {"response": response}


def clean_llm_output(output: str) -> str:
    output = output.replace("<|endoftext|>", "").strip()
    match = re.search(r"(##|###|\*\*|Answer:|CVE-\d{4}-\d+)", output, re.IGNORECASE)
    return output[match.start():].strip() if match else output.strip()
