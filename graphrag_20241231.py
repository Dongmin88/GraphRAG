import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import numpy as np
import re

class GraphRAG:
    def __init__(self, model_name: str = "Bllossom/llama-3.2-Korean-Bllossom-3B", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Using device: {self.device}")
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        
        self.knowledge_graph = nx.Graph()
        
    def add_knowledge(self, triples: List[Tuple[str, str, str]]):
        """Add knowledge triples to the graph"""
        for head, relation, tail in triples:
            # Store original and lowercase versions
            self.knowledge_graph.add_edge(head.lower(), tail.lower(), 
                                        relation=relation.lower(),
                                        original_head=head,
                                        original_tail=tail,
                                        original_relation=relation)
        print(f"Added {len(triples)} triples to knowledge graph")
        
    def _clean_text(self, text: str) -> str:
        """Clean text for entity matching"""
        # Convert to lowercase
        text = text.lower()
        # Replace special characters with space
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_entities(self, text: str) -> List[str]:
        """Improved entity extraction"""
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        # Get all nodes from the graph
        graph_entities = set(self.knowledge_graph.nodes())
        
        # Find matches in the text
        found_entities = []
        words = cleaned_text.split()
        
        # Try matching single words
        found_entities.extend([word for word in words if word in graph_entities])
        
        # Try matching pairs of words
        for i in range(len(words)-1):
            pair = words[i] + "_" + words[i+1]
            if pair in graph_entities:
                found_entities.append(pair)
        
        return list(set(found_entities))  # Remove duplicates

    def retrieve_subgraph(self, query: str, max_hops: int = 2) -> List[Tuple[str, str, str]]:
        """Retrieve relevant subgraph based on query"""
        entities = self._extract_entities(query)
        print(f"Found entities: {entities}")
        
        relevant_triples = []
        for entity in entities:
            if entity in self.knowledge_graph:
                # Get k-hop neighborhood
                neighborhood = nx.ego_graph(self.knowledge_graph, entity, radius=max_hops)
                
                # Extract triples from neighborhood
                for edge in neighborhood.edges(data=True):
                    head, tail = edge[0], edge[1]
                    data = edge[2]
                    # Use original cases for better readability
                    triple = (
                        data.get('original_head', head),
                        data.get('original_relation', data['relation']),
                        data.get('original_tail', tail)
                    )
                    if triple not in relevant_triples:
                        relevant_triples.append(triple)
        
        print(f"Retrieved {len(relevant_triples)} relevant triples")
        return relevant_triples

    def _format_prompt(self, query: str, retrieved_triples: List[Tuple[str, str, str]]) -> str:
        """Format the prompt with query and retrieved knowledge"""
        context = "\n".join([f"{h} {r} {t}" for h, r, t in retrieved_triples])
        
        prompt = f"""Using this knowledge:
{context}

Please answer this question:
{query}

Answer:"""
        return prompt

    def generate_answer(self, query: str) -> str:
        """Generate answer using retrieved knowledge"""
        retrieved_triples = self.retrieve_subgraph(query)
        
        if not retrieved_triples:
            return "I don't have enough information to answer this question."
        
        prompt = self._format_prompt(query, retrieved_triples)
        print("\nGenerated prompt:", prompt)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        print("Generating answer...")
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=100,
                    num_beams=3,
                    no_repeat_ngram_size=2,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = answer.split("Answer:")[-1].strip()
                
            except Exception as e:
                print(f"Error during generation: {e}")
                answer = "Sorry, I encountered an error while generating the answer."
        
        return answer

def test_graphrag():
    print("Initializing GraphRAG...")
    graphrag = GraphRAG()
    
    # Add test knowledge
    test_triples = [
        ("Paris", "is_capital_of", "France"),
        ("France", "located_in", "Europe"),
        ("Paris", "has_landmark", "Eiffel_Tower"),
        ("Eiffel_Tower", "built_in", "1889"),
        ("France", "has_population", "67_million"),
        ("Paris", "has_population", "2.2_million"),
    ]
    graphrag.add_knowledge(test_triples)
    
    # Test questions
    test_questions = [
        "What is the capital of France?",
        "Where is the Eiffel Tower located?",
        "When was the Eiffel Tower built?",
        "What is the population of France?",
        "Tell me about Paris and its landmarks."
    ]
    
    for question in test_questions:
        print("\n" + "="*50)
        print(f"Question: {question}")
        answer = graphrag.generate_answer(question)
        print(f"Answer: {answer}")
        print("="*50)

if __name__ == "__main__":
    test_graphrag()