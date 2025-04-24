import os
import json
import time
import urllib3
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from openSearchEmbedding.opensearch_embedding import OpenSearchEmbeddings

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ========== CONFIG + ENV LOADING ==========
def load_config():
    dotenv_path = os.path.join(os.path.dirname(__file__), "../config/config_env")
    load_dotenv(dotenv_path)
    return {
        "opensearch_url": os.getenv("OPENSEARCH_URL"),
        "model_id": os.getenv("MODEL_ID"),
        "index_name": os.getenv("INDEX_NAME"),
        "ca_certs": os.getenv("CA_CERTS_PATH"),
        "username": os.getenv("OPENSEARCH_USERNAME"),
        "password": os.getenv("OPENSEARCH_PASSWORD"),
    }


# ========== EMBEDDING WRAPPER ==========
class OpenSearchEmbeddingWrapper:
    def __init__(self, config):
        self.predict_url = f"{config['opensearch_url']}/_plugins/_ml/models/{config['model_id']}/_predict"
        self.auth = (config["username"], config["password"])
        self.embedding_model = self._wrap_embedding()

    def _embedding_function(self, text):
        headers = {"Content-Type": "application/json"}
        payload = {
            "text_docs": [text],
            "return_number": True,
            "target_response": ["sentence_embedding"]
        }

        response = requests.post(
            self.predict_url,
            headers=headers,
            data=json.dumps(payload),
            auth=self.auth,
            verify=False
        )

        if response.status_code == 200:
            return response.json()["inference_results"][0]["output"][0]["data"]
        else:
            raise Exception(f"Embedding error: {response.text}")

    def _wrap_embedding(self):
        return OpenSearchEmbeddings(embedding_function=self._embedding_function)


# ========== PROMPT TEMPLATES ==========
def get_prompt_templates(query_type):
    if query_type == "tactic":
        system_prompt = SystemMessagePromptTemplate.from_template(
            "You are a cybersecurity analyst. Use the following retrieved documents from OpenSearch to extract structured MITRE ATT&CK data.\n\n"
            "Your task is to return a complete JSON object representing a MITRE ATT&CK tactic and **all techniques** associated with it.\n\n"
            "Instructions:\n"
            "- First, identify the tactic document where the `external_id` field equals the `question` value. From this document, extract the `id`, `type`, `external_id`, and `url`.\n"
            "- Use the `id` field of this tactic document to find **all documents** where `tactic_id` matches this `id`. Each such document contains a `tech_id` field pointing to a technique.\n"
            "- For **each** `tech_id`, extract its corresponding document and retrieve the `external_id`, `name`, and the **full unmodified `description`** field.\n"
            "- Do **not** skip or filter out any `tech_id` matches. Return all techniques without omission.\n"
            "- Return the full, original `description` text from the tactic document and from each technique document. Do not summarize or modify these descriptions in any way.\n\n"
            "Return your answer strictly in JSON format as follows:\n\n"
            "{{\n"
            '  "id": "TA0001",\n'
            '  "name": "Initial Access",\n'
            '  "type": "tactic",\n'
            '  "description": "<FULL tactic description>",\n'
            '  "url": "https://attack.mitre.org/tactics/TA0001",\n'
            '  "techniques": {{\n'
            '    "T1133": {{\n'
            '      "id": "T1133",\n'
            '      "name": "External Remote Services",\n'
            '      "description": "<FULL technique description>"\n'
            "    }},\n"
            "    ...\n"
            "  }}\n"
            "}}\n\n"
            "Respond only with the raw JSON object (no quotes or markdown formatting). Do not include any natural language explanation.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}"
        )
    elif query_type == "technique":
        system_prompt = SystemMessagePromptTemplate.from_template(
            "You are a cybersecurity analyst. Use the following retrieved documents from OpenSearch to extract structured MITRE ATT&CK technique data.\n\n"
            "Your goal is to generate a JSON object where:\n"
            "- The technique document with `external_id` = {question} is used to extract its `id`, `type`, `name`, `description`, and `url`.\n"
            "- You MUST extract and return the full original `description` field exactly as it appears in the context. Do not summarize or shorten any description.\n"
            "- Use the mapping document to find the `tactic_id` associated with the technique.\n"
            "- Then, use the corresponding tactic document (with `id` matching `tactic_id`) to extract the tactic's `name`.\n"
            "- Next, find all `relationship` documents where the `target_id` matches the technique's `id`.\n"
            "- For each relationship, use the `source_id` to find the related mitigation document.\n"
            "- From the mitigation document, extract:\n"
            "    - `external_id` (as mitigation ID)\n"
            "    - `name`\n"
            "    - `description` field from the *relationship* document, as this describes how the mitigation applies to this specific technique.\n\n"
            "Return your answer strictly in JSON format like this:\n\n"
            "{{\n"
            '  "id": "T1189",\n'
            '  "name": "Drive-by Compromise",\n'
            '  "type": "technique",\n'
            '  "description": "Adversaries may gain access to a system through a user visiting a website over the normal course of browsing...",\n'
            '  "url": "https://attack.mitre.org/techniques/T1189",\n'
            '  "tactic": "Initial Access",\n'
            '  "mitigations": {{\n'
            '    "M1048": {{\n'
            '      "id": "M1048",\n'
            '      "name": "Application Isolation and Sandboxing",\n'
            '      "description": "Browser sandboxes can be used to mitigate some of the impact of exploitation, but sandbox escapes may still exist."\n'
            "    }},\n"
            "    ...\n"
            "  }}\n"
            "}}\n\n"
            "Respond only with a raw JSON object (not in quotes or markdown). Do not include any natural language explanations.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}"
        )
    else:
        raise ValueError("Invalid query type")

    human_prompt = HumanMessagePromptTemplate.from_template("{question}")
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    return chat_prompt


# ========== MAIN AGENT CLASS ==========
class CTIAgent:
    def __init__(self, question, llm_model="gpt-4"):
        self.config = load_config()
        self.question = question.strip()
        self.query_type = self._detect_query_type()
        self.llm = ChatOpenAI(model_name=llm_model)
        self.embedding = OpenSearchEmbeddingWrapper(self.config).embedding_model
        self.vectorstore = self._init_vectorstore()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.prompt_template = get_prompt_templates(self.query_type)
        self.agent = self._build_chain()

    def _detect_query_type(self):
        if self.question.upper().startswith("TA"):
            return "tactic"
        elif self.question.upper().startswith("T"):
            return "technique"
        else:
            raise ValueError("Unsupported MITRE ID format. Use TAxxxx or Txxxx.")

    def _init_vectorstore(self):
        return OpenSearchVectorSearch(
            opensearch_url=self.config["opensearch_url"],
            index_name=self.config["index_name"],
            embedding_function=self.embedding,
            use_ssl=True,
            verify_certs=False,
            ca_certs=self.config["ca_certs"],
            http_auth=(self.config["username"], self.config["password"])
        )

    def _build_chain(self):
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory,
            combine_docs_chain_kwargs={
                "prompt": self.prompt_template,
                "document_variable_name": "context"
            }
        )

    def ask(self, input_query, max_retries=7, delay=3):
        for attempt in range(max_retries):
            try:
                query_response = self.agent.invoke({"question": input_query})
                answer_text = query_response["answer"]
                return json.loads(answer_text)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    return {"error": "Failed to parse JSON after multiple attempts", "raw_output": answer_text}
            except Exception as e:
                return {"error": str(e)}


# ========== ENTRY POINT ==========
if __name__ == "__main__":
    input_question = "TA0003"  # Change this for testing
    agent = CTIAgent(input_question)
    response = agent.ask(input_question)

    print("\n\nLLM Response:")
    print(json.dumps(response, indent=4) if isinstance(response, dict) else response)
