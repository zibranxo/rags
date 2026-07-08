import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import random
import time

from src.generation.llm_client import LLMClient
from src.utils.logger import setup_logger
from src.ingestion.indexer import _get_client, COLLECTION_NAME

logger = setup_logger("rags.crag.bootstrap")

PROMPT = """You are an expert AI assistant tasked with generating queries to train a Retrieval-Augmented Generation (RAG) classification model.
Given the following context passage, generate EXACTLY three questions:

1. A "Correct" question: This question should be fully and perfectly answered by the information in the passage.
2. An "Ambiguous" question: This question should be related to the passage, and the passage should provide partial information, but it requires MORE context to fully answer. For example, asking about a specific detail not mentioned, or a broader concept that the passage only touches upon.
3. An "Incorrect" question: This question should sound like it COULD be related to the passage's topic (e.g. same field of study), but the passage provides absolutely NO information to answer it.

Output the result strictly in this format, with one question per line, no extra text:
[Correct] <question>
[Ambiguous] <question>
[Incorrect] <question>

Context Passage:
{text}
"""

def generate_bootstrap_data(num_chunks: int = 50, out_file: str = "data/crag_bootstrap.jsonl"):
    client = _get_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        logger.error(f"Failed to get collection: {e}")
        return

    # Get all chunk IDs
    result = collection.get(include=["metadatas", "documents"])
    ids = result["ids"]
    metadatas = result["metadatas"]
    documents = result["documents"]

    if not ids:
        logger.error("No chunks in collection!")
        return

    # Filter out parent chunks
    child_indices = [i for i, m in enumerate(metadatas) if not m.get("is_parent", False)]
    
    if len(child_indices) < num_chunks:
        logger.warning(f"Only {len(child_indices)} child chunks available, using all of them.")
        sampled_indices = child_indices
    else:
        sampled_indices = random.sample(child_indices, num_chunks)

    llm = LLMClient()
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Bootstrapping {len(sampled_indices)} chunks -> ~{len(sampled_indices)*3} training pairs...")

    with open(out_path, "w", encoding="utf-8") as f:
        for idx in sampled_indices:
            chunk_id = ids[idx]
            text = documents[idx]
            
            try:
                resp = llm.generate(
                    system_prompt="You are a data generation assistant.",
                    user_prompt=PROMPT.format(text=text),
                    temperature=0.7,
                    max_tokens=300
                )
                
                lines = resp.answer.strip().split("\n")
                
                # Parse output
                for line in lines:
                    line = line.strip()
                    if line.startswith("[Correct]"):
                        q = line.replace("[Correct]", "").strip()
                        label = "Correct"
                    elif line.startswith("[Ambiguous]"):
                        q = line.replace("[Ambiguous]", "").strip()
                        label = "Ambiguous"
                    elif line.startswith("[Incorrect]"):
                        q = line.replace("[Incorrect]", "").strip()
                        label = "Incorrect"
                    else:
                        continue
                        
                    if q:
                        record = {
                            "query": q,
                            "chunk_id": chunk_id,
                            "chunk_text": text,
                            "label": label
                        }
                        f.write(json.dumps(record) + "\n")
                        
                time.sleep(0.5) # Slight delay to respect rate limits if using API
                
            except Exception as e:
                logger.warning(f"Failed to generate queries for chunk {chunk_id}: {e}")
                
    logger.info(f"Saved bootstrapped dataset to {out_path}")

if __name__ == "__main__":
    generate_bootstrap_data()
