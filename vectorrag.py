import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Dict, Any, Tuple
from huggingface_hub import login
import pdfplumber
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class PDFLlama3VectorRAG:
    def __init__(self):
        """Initialize the VectorRAG system with Llama 3 model"""
        try:
            # Login to Hugging Face
            login(token="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")
            
            # Initialize model and tokenizer
            print("Loading Llama 3 model...")
            self.model_name = "meta-llama/Llama-3.2-1b"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Initialize vector store
            self.document_store = []
            self.document_embeddings = []
            
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

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents in the vector store
        
        Args:
            documents: List of document dictionaries containing text and metadata
        """
        try:
            for doc in tqdm(documents, desc="Indexing documents"):
                # Generate embedding for the document
                embedding = self._encode_text(doc['text'])
                
                # Store document and its embedding
                self.document_store.append(doc)
                self.document_embeddings.append(embedding)
                
        except Exception as e:
            print(f"Error indexing documents: {str(e)}")
            raise

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve relevant documents based on query
        
        Args:
            query: Input query string
            k: Number of documents to retrieve
            
        Returns:
            List of (document, similarity_score) tuples
        """
        try:
            query_embedding = self._encode_text(query)
            
            # Convert list of embeddings to numpy array for efficient computation
            embeddings_array = np.array(self.document_embeddings)
            
            # Calculate cosine similarities
            similarities = cosine_similarity([query_embedding], embeddings_array)[0]
            
            # Get top k documents
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            results = [
                (self.document_store[i], float(similarities[i]))
                for i in top_k_indices
            ]
            
            return results
            
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            raise

    def generate_response(self, query: str, retrieved_docs: List[Tuple[Dict[str, Any], float]]) -> str:
        """
        Generate response based on query and retrieved documents
        
        Args:
            query: Input query string
            retrieved_docs: List of (document, similarity_score) tuples
            
        Returns:
            Generated response string
        """
        try:
            # Collect context and sources
            context_parts = []
            sources = set()
            
            for doc, score in retrieved_docs:
                # Add relevant text passages
                context_parts.append(f"Passage (relevance score {score:.3f}):\n{doc['text']}")
                
                # Collect sources
                sources.add(f"{os.path.basename(doc['source'])} (pages {', '.join(map(str, doc['pages']))})")
            
            context = "\n\n".join(context_parts)
            sources_text = "\nSources: " + "; ".join(sorted(sources))
            
            # Generate response
            prompt = (
                "Based on the following passages, answer the query comprehensively "
                "and accurately. Include relevant details from the sources.\n\n"
                f"Context:\n{context}\n\n"
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