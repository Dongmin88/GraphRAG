import networkx as nx
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from typing import List, Dict, Tuple
import numpy as np

class GraphRAG:
    def __init__(self, model_name: str = "Bllossom/llama-3.2-Korean-Bllossom-3B", device: str = "cuda"):
        self.device = device
        # Initialize LLaMA model and tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name).to(device)
        
        # Initialize knowledge graph
        self.knowledge_graph = nx.Graph()
        
    def add_knowledge(self, triples: List[Tuple[str, str, str]]):
        """Add knowledge triples to the graph"""
        for head, relation, tail in triples:
            self.knowledge_graph.add_edge(head, tail, relation=relation)
    
    def retrieve_subgraph(self, query: str, max_hops: int = 2) -> List[Tuple[str, str, str]]:
        """Retrieve relevant subgraph based on query"""
        # Simple entity matching (could be improved with better NER)
        entities = self._extract_entities(query)
        
        relevant_triples = []
        for entity in entities:
            if entity in self.knowledge_graph:
                # Get k-hop neighborhood
                neighborhood = nx.ego_graph(self.knowledge_graph, entity, radius=max_hops)
                
                # Extract triples from neighborhood
                for edge in neighborhood.edges(data=True):
                    head, tail = edge[0], edge[1]
                    relation = edge[2]['relation']
                    relevant_triples.append((head, relation, tail))
                    
        return relevant_triples

    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction (could be improved with better NER)"""
        # For demo purposes, just split by spaces and check if words are in the graph
        potential_entities = text.lower().split()
        return [entity for entity in potential_entities if entity in self.knowledge_graph]

    def _format_prompt(self, query: str, retrieved_triples: List[Tuple[str, str, str]]) -> str:
        """Format the prompt with query and retrieved knowledge"""
        context = "\n".join([f"{h} {r} {t}" for h, r, t in retrieved_triples])
        prompt = f"""Given the following knowledge:
{context}

Please answer this question:
{query}

Answer:"""
        return prompt

    def generate_answer(self, query: str) -> str:
        """Generate answer using retrieved knowledge"""
        # Retrieve relevant subgraph
        retrieved_triples = self.retrieve_subgraph(query)
        
        # Format prompt
        prompt = self._format_prompt(query, retrieved_triples)
        
        # Generate answer using LLaMA
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=100,
                num_beams=5,
                no_repeat_ngram_size=2,
                temperature=0.7
            )
            
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated answer part
        answer = answer.split("Answer:")[-1].strip()
        
        return answer

def test_graphrag():
    # Initialize GraphRAG
    graphrag = GraphRAG(model_name="meta-llama/Llama-2-7b-hf")
    
    # Add some test knowledge
    test_triples = [
        ("paris", "is_capital_of", "france"),
        ("france", "located_in", "europe"),
        ("paris", "has_landmark", "eiffel_tower"),
        ("eiffel_tower", "built_in", "1889"),
    ]
    graphrag.add_knowledge(test_triples)
    
    # Test question
    question = "What is the capital of France and what famous landmark is there?"
    
    # Generate answer
    answer = graphrag.generate_answer(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    test_graphrag()