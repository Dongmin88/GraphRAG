import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Dict, Any, Tuple
from huggingface_hub import login
import pdfplumber
import os
from tqdm import tqdm

class PDFLlama3GraphRAG:
    def __init__(self):
        """Initialize the GraphRAG system with Llama 3 model"""
        try:
            # Login to Hugging Face
            login(token="you-token.")
            
            # Initialize model and tokenizer
            print("Loading Llama 3 model...")
            self.model_name = "meta-llama/Llama-3.2-1b"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Initialize graph structure
            self.graph = nx.Graph()
            self.node_embeddings = {}
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("System initialized successfully")
            
        except Exception as e:
            print(f"Error initializing system: {str(e)}")
            raise

    def read_pdf(self, pdf_path: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Read and process a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Maximum characters per text chunk
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        documents = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                current_chunk = ""
                page_numbers = []
                
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        words = text.split()
                        
                        for word in words:
                            if len(current_chunk) + len(word) + 1 > chunk_size:
                                # Store complete chunk
                                documents.append({
                                    'text': current_chunk.strip(),
                                    'source': pdf_path,
                                    'pages': page_numbers.copy()
                                })
                                current_chunk = word
                                page_numbers = [page_num + 1]
                            else:
                                current_chunk += " " + word
                                if page_num + 1 not in page_numbers:
                                    page_numbers.append(page_num + 1)
                
                # Add final chunk
                if current_chunk:
                    documents.append({
                        'text': current_chunk.strip(),
                        'source': pdf_path,
                        'pages': page_numbers
                    })
                    
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
            
        return documents

    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using Llama 3 model
        
        Args:
            text: Input text to encode
            
        Returns:
            Numpy array of text embedding
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
                return embedding
                
        except Exception as e:
            print(f"Error encoding text: {str(e)}")
            raise

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text using Llama 3 model
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of extracted entities
        """
        try:
            prompt = (
                "Extract key entities (including technical terms, concepts, proper nouns) "
                "from the following text. Output only the entities, one per line.\n\n"
                f"Text: {text}\n\n"
                "Entities:"
            )
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.1,
                    do_sample=False
                )
                
            entities = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            entities = entities.split("Entities:")[-1].strip()
            return [e.strip() for e in entities.split("\n") if e.strip()]
            
        except Exception as e:
            print(f"Error extracting entities: {str(e)}")
            raise

    def construct_graph(self, documents: List[Dict[str, Any]]):
        """
        Construct knowledge graph from documents
        
        Args:
            documents: List of document dictionaries containing text and metadata
        """
        try:
            for doc in tqdm(documents, desc="Processing documents"):
                entities = self._extract_entities(doc['text'])
                
                # Add nodes with document metadata
                for entity in entities:
                    if entity not in self.graph.nodes:
                        self.graph.add_node(
                            entity,
                            text=entity,
                            documents=[{
                                'source': doc['source'],
                                'pages': doc['pages']
                            }]
                        )
                        self.node_embeddings[entity] = self._encode_text(entity)
                    else:
                        # Update document references
                        self.graph.nodes[entity]['documents'].append({
                            'source': doc['source'],
                            'pages': doc['pages']
                        })
                
                # Add edges between co-occurring entities
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        if self.graph.has_edge(entities[i], entities[j]):
                            self.graph[entities[i]][entities[j]]['weight'] += 1
                        else:
                            self.graph.add_edge(entities[i], entities[j], weight=1)
                            
        except Exception as e:
            print(f"Error constructing graph: {str(e)}")
            raise

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve relevant nodes based on query
        
        Args:
            query: Input query string
            k: Number of nodes to retrieve
            
        Returns:
            List of (node, similarity_score) tuples
        """
        try:
            query_embedding = self._encode_text(query)
            
            similarities = []
            for node, embedding in self.node_embeddings.items():
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((node, float(similarity)))
            
            return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
            
        except Exception as e:
            print(f"Error retrieving nodes: {str(e)}")
            raise

    def get_subgraph(self, nodes: List[str], n_hops: int = 1) -> nx.Graph:
        """
        Get subgraph containing input nodes and their n-hop neighbors
        
        Args:
            nodes: List of starting nodes
            n_hops: Number of hops to expand from starting nodes
            
        Returns:
            NetworkX subgraph
        """
        try:
            nodes_to_keep = set()
            
            for node in nodes:
                nodes_to_keep.add(node)
                for other_node in self.graph.nodes():
                    try:
                        path_length = nx.shortest_path_length(self.graph, node, other_node)
                        if path_length <= n_hops:
                            nodes_to_keep.add(other_node)
                    except nx.NetworkXNoPath:
                        continue
            
            return self.graph.subgraph(nodes_to_keep)
            
        except Exception as e:
            print(f"Error getting subgraph: {str(e)}")
            raise

    def generate_response(self, query: str, subgraph: nx.Graph) -> str:
        """
        Generate response based on query and retrieved subgraph
        
        Args:
            query: Input query string
            subgraph: Retrieved subgraph
            
        Returns:
            Generated response string
        """
        try:
            # Collect graph information and sources
            graph_info = []
            sources = set()
            
            for node in subgraph.nodes():
                # Add node relationships
                neighbors = list(subgraph.neighbors(node))
                if neighbors:
                    graph_info.append(f"{node} is connected to: {', '.join(neighbors)}")
                
                # Collect sources
                for doc in self.graph.nodes[node]['documents']:
                    sources.add(f"{os.path.basename(doc['source'])} (pages {', '.join(map(str, doc['pages']))})")
            
            graph_context = "\n".join(graph_info)
            sources_text = "\nSources: " + "; ".join(sorted(sources))
            
            # Generate response
            prompt = (
                "Based on the following information, answer the query comprehensively "
                "and accurately. Include relevant details from the sources.\n\n"
                f"Information:\n{graph_context}\n\n"
                f"Query: {query}\n\n"
                "Answer:"
            )
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Answer:")[-1].strip()
            
            # Add sources to response
            return response + "\n" + sources_text
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            raise