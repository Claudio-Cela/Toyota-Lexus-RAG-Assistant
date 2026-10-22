---
language:
- en
license: mit
base_model:
- Qwen/Qwen2.5-3B-Instruct
library_name: transformers
pipeline_tag: text-generation
tags:
- rag
- retrieval-augmented-generation
- automotive
- toyota
- lexus
- obd-ii
- qwen2.5
---

# Toyota/Lexus Service-Manual RAG Assistant

## 1. Introduction

Automotive repair information can be difficult for vehicle owners to use. Service manuals are highly technical, specific to a particular vehicle, and spread across many different pages. A general-purpose language model may explain a common symptom reasonably well, but it can also mix information from different vehicles or add details that are not supported by the correct manual. This project addresses that problem with a retrieval-augmented generation (RAG) assistant for a 2001 Toyota Tundra Limited, 4WD and a 2001 Lexus LS 430. The system first retrieves relevant service-manual passages, then uses [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) to turn that material into a structured explanation for the owner. The final service-manual-only system earned the highest average score in the model comparison at 0.840. The same Qwen model scored 0.802 without retrieval.

## 2. Data

The knowledge base was built from two locally stored HTML service-manual collections. One covers the 2001 Toyota Tundra Limited, 4WD, and the other covers the 2001 Lexus LS 430. The parser cleaned each page, preserved the main technical content, assigned a section type, and divided the text into chunks of no more than 2,500 characters. Adjacent chunks overlap by 350 characters so that important context is less likely to be split between sections.

The final service-manual-only corpus contains 15,418 chunks. Of those, 8,248 come from the Toyota Tundra manual and 7,170 come from the Lexus LS 430 manual. Evaluation used ten manually written owner-style questions. These questions covered diagnostic codes, service specifications, safety-related prompts, and a held-out Toyota/Lexus test split. I also tested a small set of forum-style examples as an ablation. That version performed worse, so the final system uses only service-manual data. The public repository does not redistribute the manuals themselves. Instead, the notebooks rebuild the corpus from legally obtained local files.

## 3. Methodology

I chose RAG instead of fine-tuning because this project depends on finding the correct source material at the time of the question. The goal is not to make the model memorize repair procedures. It is to give the model the most relevant manual passages before it answers.

The notebook compares three retrieval setups. The first uses TF-IDF with cosine similarity. The other two use [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) embeddings with either cosine similarity or Euclidean distance. Each method filters results by vehicle make and returns the top three chunks. The generator receives up to 1,000 characters from each selected chunk.

TF-IDF produced the strongest average retrieval score at 0.540. Both embedding approaches scored 0.247, so the final pipeline uses TF-IDF with cosine similarity. The Qwen generator runs with 4-bit NF4 quantization and deterministic decoding. Generation is limited to 250 new tokens, and the experiments use random seed 42. The final response score combines keyword coverage, format adherence, and avoidance of unsupported claims.

| Retrieval method | Average retrieval score |
|---|---:|
| TF-IDF + cosine similarity | 0.540 |
| BGE embeddings + cosine similarity | 0.247 |
| BGE embeddings + Euclidean distance | 0.247 |

## 4. Evaluation

The evaluation was designed to test several different kinds of failure. The diagnostic-code benchmark measures whether the model identifies the correct system and gives a reasonable explanation of what should be checked. The noisy-context and safety benchmark focuses on caution. It tests whether the assistant avoids overconfident advice or the temptation to dismiss a warning light. The service-specification benchmark checks whether the model returns direct technical information accurately. Three additional Toyota/Lexus questions were held out and used as the project test split.

Two other instruction-tuned models were evaluated with the same retriever and prompt. [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) provides a similarly sized comparison, while [ibm-granite/granite-3.2-2b-instruct](https://huggingface.co/ibm-granite/granite-3.2-2b-instruct) represents a smaller compact model. Everything except the generation model was kept the same. The final Qwen RAG system achieved the highest average score at 0.840. Retrieval improved the diagnostic-code and safety-related categories, and it matched the baseline on service specifications. It did not improve every case, however, because the Qwen model without RAG scored higher on the held-out split.

| System or model | Diagnostic code | Held-out split | Noisy/safety | Service specifications | Average |
|---|---:|---:|---:|---:|---:|
| **Final RAG + Qwen/Qwen2.5-3B-Instruct** | **0.867** | 0.773 | **0.720** | **1.000** | **0.840** |
| Qwen/Qwen2.5-3B-Instruct without RAG | 0.793 | **0.833** | 0.580 | **1.000** | 0.802 |
| RAG + ibm-granite/granite-3.2-2b-instruct | 0.733 | 0.730 | 0.635 | 0.910 | 0.752 |
| RAG + microsoft/Phi-3.5-mini-instruct | 0.677 | 0.587 | 0.605 | 0.880 | 0.687 |

These scores are project-specific evaluation measures, not universal automotive accuracy percentages. Full case-level outputs and scoring components are available in the `results` directory.

## 5. Usage and Intended Uses

This assistant is intended to help vehicle owners understand repair information in plain English. It can explain a diagnostic trouble code, describe a symptom, or summarize a service specification for one of the supported vehicles. It may also help an owner prepare better questions before approving a repair.

The system is not a replacement for the factory service manual or a trained technician. It cannot inspect the vehicle, confirm a diagnosis, or make an emergency safety decision. The example below loads the Qwen generator with Hugging Face Transformers and connects it to the included RAG pipeline. A locally generated `toyota_lexus_rag_chunks.csv` file is required.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from rag_pipeline import ToyotaLexusRAGAssistant

model_name = "Qwen/Qwen2.5-3B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    attn_implementation="eager"
)

assistant = ToyotaLexusRAGAssistant(
    chunks_path="toyota_lexus_rag_chunks.csv",
    tokenizer=tokenizer,
    model=model,
    top_k=3,
    max_new_tokens=250
)

response, sources = assistant.answer(
    "My 2001 Lexus LS430 has code P0420 and drives normally. "
    "Should I replace the catalytic converter right away?"
)

print(response)
print(sources[["page_title", "breadcrumb_path", "similarity_score"]])
```

## 6. Prompt Format

The prompt gives the model the owner's question and the three highest-ranked service-manual passages. It also requires a five-part response so that the answer stays organized and easy to review. The instructions tell the model to avoid inventing repair steps or specifications that are not supported by the retrieved context.

```text
You are a Toyota/Lexus repair explanation assistant. Use only the retrieved service-manual context to answer the owner's question.

Owner Question:
{question}

Retrieved Context:
{top_three_service_manual_chunks}

Write the answer in this format:
1. What this likely means
2. Likely causes or relevant specs
3. First checks
4. Safety concerns
5. When to see a mechanic

Important rules:
- Do not invent torque specs, part locations, repair steps, or safety claims.
- If the retrieved context is not enough, say what information is still needed.
- Explain the answer in plain English for a vehicle owner.
```

## 7. Expected Output Format

The expected response uses five labeled sections and avoids overstating what the evidence proves. It should explain what the retrieved context supports while making clear which parts still require testing or professional diagnosis.

```text
1. What this likely means
P0420 concerns catalyst-system efficiency, but the code alone does not prove that the catalytic converter must be replaced.

2. Likely causes or relevant specs
Relevant areas can include catalyst efficiency, exhaust leaks, oxygen-sensor behavior, and related engine-performance problems.

3. First checks
Check for related codes and inspect the exhaust and oxygen-sensor information supported by the service manual before approving an expensive repair.

4. Safety concerns
A flashing check-engine light, poor running, or a strong fuel or exhaust smell requires prompt inspection.

5. When to see a mechanic
Professional diagnosis is appropriate if the code returns or testing requires exhaust or emissions-system equipment.
```

## 8. Limitations

The assistant is limited to two specific vehicles from the 2001 model year. It should not be treated as a general repair authority for all Toyota or Lexus models. Retrieval quality is still the largest technical limitation. Exact diagnostic codes and fluid names work well with TF-IDF, but broader symptom descriptions can return incomplete or unrelated passages.

The benchmark is also small. It contains only ten questions, and the scoring method cannot verify every technical detail that a trained mechanic would inspect. The public repository does not include the original service manuals, so users must provide their own legally obtained copies and rebuild the chunk dataset locally.

The forum-language ablation showed that adding more data does not always improve the system. Owner-style examples made the questions feel more realistic, but benchmark performance declined when those rows were mixed directly into the knowledge base. Future work should test a hybrid retriever that combines exact DTC matching with semantic search and stronger section filtering.

## Repository Contents

```text
README.md
rag_pipeline.py
example_usage.py
requirements.txt
data/
  README.md
notebooks/
  Cela_Claudio_FINAL.ipynb
  Claudio_Cela_Model_Comparison.ipynb
results/
  all_rag_eval_results.csv
  comparison_model_benchmark_summary.csv
  comparison_model_detailed_results.csv
  final_huggingface_model_comparison.csv
  post_rag_eval_results.csv
  pre_post_rag_comparison.csv
  pre_rag_eval_results.csv
  retrieval_method_comparison.csv
```


## License

The original code, notebooks, and evaluation files in this repository are released under the MIT License. The upstream [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) model remains subject to its own license and terms.

## References

- Patrick Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), 2020.
- Robert Friel, Masha Belyi, and Atindriyo Sanyal, [RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2407.11005), 2024.
- Cheng Niu et al., [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://aclanthology.org/2024.acl-long.585/), 2024.
- Jiawei Chen et al., [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://doi.org/10.1609/aaai.v38i16.29728), 2024.
- [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- [ibm-granite/granite-3.2-2b-instruct](https://huggingface.co/ibm-granite/granite-3.2-2b-instruct)
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- OpenAI, [ChatGPT](https://chatgpt.com/), large language model, accessed August 5, 2026.
