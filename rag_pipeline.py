from pathlib import Path

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ToyotaLexusRAGAssistant:
    def __init__(
        self,
        chunks_path,
        tokenizer,
        model,
        top_k=3,
        max_new_tokens=250,
        max_features=50000
    ):
        self.chunks_path = Path(chunks_path)
        self.tokenizer = tokenizer
        self.model = model
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Chunk file not found: {self.chunks_path}. "
                "Build toyota_lexus_rag_chunks.csv from legally obtained local manuals first."
            )

        self.chunks_df = pd.read_csv(self.chunks_path)

        required_columns = {
            "text_chunk",
            "vehicle_make",
            "vehicle_model",
            "model_year",
            "trim",
            "source_type",
            "page_title",
            "section_type",
            "breadcrumb_path"
        }
        missing = required_columns.difference(self.chunks_df.columns)

        if missing:
            raise ValueError(
                f"The chunk file is missing required columns: {sorted(missing)}"
            )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.chunks_df["text_chunk"].fillna("")
        )

    def _vehicle_filter(self, question):
        question_lower = question.lower()

        if "lexus" in question_lower or "ls430" in question_lower or "ls 430" in question_lower:
            return {"vehicle_make": "Lexus"}

        if "toyota" in question_lower or "tundra" in question_lower:
            return {"vehicle_make": "Toyota"}

        return {}

    def retrieve(self, question):
        search_df = self.chunks_df.copy()

        for column, value in self._vehicle_filter(question).items():
            filtered = search_df[
                search_df[column].astype(str).str.lower() == value.lower()
            ]
            if len(filtered) > 0:
                search_df = filtered

        search_matrix = self.vectorizer.transform(
            search_df["text_chunk"].fillna("")
        )
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, search_matrix).flatten()
        top_indices = scores.argsort()[::-1][:self.top_k]

        retrieved = search_df.iloc[top_indices].copy()
        retrieved["similarity_score"] = scores[top_indices]
        return retrieved

    @staticmethod
    def _clean_text(text):
        return " ".join(str(text).split())

    def build_context(self, retrieved_df, max_chunk_chars=1000):
        context_parts = []

        for source_number, (_, row) in enumerate(
            retrieved_df.iterrows(),
            start=1
        ):
            text = self._clean_text(row["text_chunk"])

            if len(text) > max_chunk_chars:
                text = text[:max_chunk_chars].rsplit(" ", 1)[0] + "..."

            warning = row.get("warning_text", "")
            warning_line = (
                f"Warning: {warning}\n"
                if isinstance(warning, str) and warning.strip()
                else ""
            )

            context_parts.append(
                f"""
Source {source_number}
Vehicle: {row.get("model_year", "")} {row.get("vehicle_make", "")} {row.get("vehicle_model", "")} {row.get("trim", "")}
Source Type: {row.get("source_type", "")}
Page Title: {row.get("page_title", "")}
Section Type: {row.get("section_type", "")}
Breadcrumb: {row.get("breadcrumb_path", "")}
{warning_line}Text:
{text}
""".strip()
            )

        return "\n\n---\n\n".join(context_parts)

    def build_prompt(self, question, retrieved_df):
        context = self.build_context(retrieved_df)

        return f"""
You are a Toyota/Lexus repair explanation assistant. Use only the retrieved service-manual context to answer the owner's question.

Owner Question:
{question}

Retrieved Context:
{context}

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
""".strip()

    def _format_chat_prompt(self, prompt):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful Toyota/Lexus repair explanation assistant. "
                    "Do not invent unsupported repair details. "
                    "If source context is missing, say what information is still needed."
                )
            },
            {"role": "user", "content": prompt}
        ]

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            return messages[0]["content"] + "\n\n" + prompt

    def generate(self, prompt):
        formatted_prompt = self._format_chat_prompt(prompt)

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )
        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = output_ids[0][
            inputs["input_ids"].shape[-1]:
        ]

        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

    def answer(self, question):
        retrieved = self.retrieve(question)
        prompt = self.build_prompt(question, retrieved)
        response = self.generate(prompt)
        return response, retrieved