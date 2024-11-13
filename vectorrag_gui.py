import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Dict, Any, Tuple
from huggingface_hub import login
import pdfplumber
import os
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import graphrag
import vectorrag

class PDFGraphRAGApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PDF GraphRAG System")
        self.root.geometry("800x600")
        
        # Initialize GraphRAG
        self.rag = None
        self.documents = []
        
        self.create_gui()
        
    def create_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File selection
        ttk.Label(main_frame, text="Select PDF Files:").grid(row=0, column=0, sticky=tk.W)
        self.file_button = ttk.Button(main_frame, text="Choose Files", command=self.select_files)
        self.file_button.grid(row=0, column=1, pady=5)
        
        # Selected files display
        self.files_text = tk.Text(main_frame, height=5, width=70)
        self.files_text.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Process button
        self.process_button = ttk.Button(main_frame, text="Process PDFs", command=self.process_pdfs)
        self.process_button.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Query input
        ttk.Label(main_frame, text="Enter Query:").grid(row=3, column=0, sticky=tk.W)
        self.query_entry = ttk.Entry(main_frame, width=70)
        self.query_entry.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Query button
        self.query_button = ttk.Button(main_frame, text="Submit Query", command=self.submit_query)
        self.query_button.grid(row=5, column=0, columnspan=2, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, length=300, mode='determinate', variable=self.progress_var)
        self.progress_bar.grid(row=6, column=0, columnspan=2, pady=5)
        
        # Response display
        ttk.Label(main_frame, text="Response:").grid(row=7, column=0, sticky=tk.W)
        self.response_text = tk.Text(main_frame, height=10, width=70)
        self.response_text.grid(row=8, column=0, columnspan=2, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.grid(row=9, column=0, columnspan=2, pady=5)
        
        # Disable buttons initially
        self.query_button.state(['disabled'])
        
    def select_files(self):
        filetypes = (('PDF files', '*.pdf'), ('All files', '*.*'))
        filenames = filedialog.askopenfilenames(
            title='Select PDF files',
            filetypes=filetypes
        )
        
        if filenames:
            self.files_text.delete('1.0', tk.END)
            self.files_text.insert('1.0', '\n'.join(filenames))
            self.process_button.state(['!disabled'])
            
    def initialize_rag(self):
        if self.rag is None:
            self.status_var.set("Initializing Llama model...")
            self.rag = vectorrag.PDFLlama3VectorRAG()
            
    def process_pdfs(self):
        # Disable buttons during processing
        self.process_button.state(['disabled'])
        self.file_button.state(['disabled'])
        
        # Get selected files
        files = self.files_text.get('1.0', tk.END).strip().split('\n')
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self.process_pdfs_thread, args=(files,))
        thread.start()
        
    def process_pdfs_thread(self, files):
        try:
            self.initialize_rag()
            
            # Process each PDF
            self.documents = []
            total_files = len(files)
            
            for i, file in enumerate(files):
                self.status_var.set(f"Processing PDF {i+1}/{total_files}: {os.path.basename(file)}")
                self.progress_var.set((i / total_files) * 50)
                self.root.update_idletasks()
                
                docs = self.rag.read_pdf(file)
                self.documents.extend(docs)
            
            # Construct graph
            self.status_var.set("Constructing knowledge graph...")
            self.progress_var.set(50)
            self.root.update_idletasks()
            
            self.rag.index_documents(self.documents)
            
            self.status_var.set("Ready for queries")
            self.progress_var.set(100)
            
            # Enable query button
            self.query_button.state(['!disabled'])
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
        finally:
            # Re-enable buttons
            self.process_button.state(['!disabled'])
            self.file_button.state(['!disabled'])
            
    def submit_query(self):
        query = self.query_entry.get().strip()
        if not query:
            self.status_var.set("Please enter a query")
            return
            
        # Disable buttons during processing
        self.query_button.state(['disabled'])
        
        # Start query processing in a separate thread
        thread = threading.Thread(target=self.process_query_thread, args=(query,))
        thread.start()
        
    def process_query_thread(self, query):
        try:
            self.status_var.set("Processing query...")
            self.progress_var.set(0)
            
            # Retrieve relevant nodes
            relevant_nodes = self.rag.retrieve(query)
            self.progress_var.set(33)
            
            # Get subgraph
            nodes = [node for node, _ in relevant_nodes]
            subgraph = self.rag.retrieve(query, k=5)
            self.progress_var.set(66)
            
            # Generate response
            response = self.rag.generate_response(query, subgraph)
            self.progress_var.set(100)
            
            # Update response text
            self.response_text.delete('1.0', tk.END)
            self.response_text.insert('1.0', response)
            
            self.status_var.set("Query completed")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
        finally:
            # Re-enable query button
            self.query_button.state(['!disabled'])
    
    def run(self):
        self.root.mainloop()

def main():
    app = PDFGraphRAGApp()
    app.run()

if __name__ == "__main__":
    main()