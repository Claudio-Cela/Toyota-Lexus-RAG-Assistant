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
    "My 2001 Toyota Tundra has an EVAP code P0440. What should I check first?"
)

print(response)
print(sources[["page_title", "breadcrumb_path", "similarity_score"]])