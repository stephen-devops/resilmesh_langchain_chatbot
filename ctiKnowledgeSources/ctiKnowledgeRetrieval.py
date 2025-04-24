import os
import sys
import json
import requests
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from openSearchCTIRetrieval.openSearchCTIAgent import CTIAgent


dotenv_path = os.path.join(os.path.dirname(__file__), "../config/config_env")
load_dotenv(dotenv_path)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../openSearchCTIRetrieval")))

# access Global CTI data sources
# OTX API
otx_api_key = os.getenv("OTX_API_KEY")

# access MISP URL & API_KEY
misp_url = os.getenv("MISP_URL")
misp_api_key = os.getenv("MISP_API_KEY")

# access local CTI data source
opensearch_url = os.getenv("OPENSEARCH_URL")
model_id = os.getenv("MODEL_ID")
index_name = os.getenv("INDEX_NAME")
CA_CERTS_PATH = os.getenv("CA_CERTS_PATH")
opensearch_username = os.getenv("OPENSEARCH_USERNAME")
opensearch_password = os.getenv("OPENSEARCH_PASSWORD")
AUTH = (opensearch_username, opensearch_password)

opensearch_client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=AUTH,
    http_compress=True,
    use_ssl=True,
    verify_certs=False,
    ca_certs=CA_CERTS_PATH
)


def get_opensearch_embedding(text):

    embedding_predict_url = f"{opensearch_url}/_plugins/_ml/models/{model_id}/_predict"

    headers = {"Content-Type": "application/json"}
    payload = {
        "text_docs": [text],
        "return_number": True,
        "target_response": ["sentence_embedding"]
    }

    response = requests.post(
        embedding_predict_url,
        headers=headers,
        data=json.dumps(payload),
        auth=AUTH,
        verify=False
    )

    # print(f"Get OpenSearch Embedding: {response.text}")

    if response.status_code == 200:
        try:

            result = response.json()["inference_results"][0]["output"][0]["data"]
            return result
        except KeyError:
            raise Exception(f"Unexpected OpenSearch response format: {response.json()}")
    else:
        raise Exception(f"OpenSearch Inference Error: {response.text}")


# ----------------------
# Unified Local & Global CTI Integration
# ----------------------
class CTIIntegration:

    @staticmethod
    def query_otx(indicator: str, ioc_type: str) -> dict:
        endpoint_map = {"ip": "IPv4", "domain": "domain", "hash": "file", "url": "url"}
        try:
            response = requests.get(
                f"https://otx.alienvault.com/api/v1/indicators/{endpoint_map[ioc_type]}/{indicator}/general",
                headers={"X-OTX-API-KEY": otx_api_key},
                timeout=15
            )
            # print(f"OTX Query Responding ... {response.json()}")
            # print(f"OTX Query Responding ... {type(response.json())}")
            return response.json()
        except Exception as e:
            print(f"OTX Query Error: {str(e)}")
            return {}

    @staticmethod
    def query_misp(keyword: str) -> dict:

        try:
            response = requests.post(
                f"{misp_url}/events/restSearch",
                headers={"Authorization": misp_api_key},
                json={
                    "returnFormat": "json",
                    "search_string": keyword,
                    "limit": 5
                },
                verify=False,
                timeout=15
            )
            # print(f"MISP Query Responding ... {response.json()}")
            return response.json().get('response', [])
        except Exception as e:
            print(f"MISP Query Error: {str(e)}")
            return {}

    @staticmethod
    def query_opensearch(keyword: str) -> dict:
        try:
            agent = CTIAgent(keyword)
            response = agent.ask(keyword)
            return response

        except Exception as e:
            print(f"OpenSearch Query Error: {str(e)}")
            return {}
