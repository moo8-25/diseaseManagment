"""
Plant Disease RAG System - Railway Deployment
Fixed JSON serialization issue
"""

import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime

app = FastAPI(title="Plant Disease RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# RAG SYSTEM
# ============================================

class PlantDiseaseRAG:
    def __init__(self):
        self.index = None
        self.documents = None
        self.embedding_model = None
        self.load_or_build()
    
    def load_or_build(self):
        """Load existing models or build from CSV"""
        base_dir = os.path.dirname(__file__)
        index_path = os.path.join(base_dir, "faiss_index.bin")
        docs_path = os.path.join(base_dir, "documents.pkl")
        csv_path = os.path.join(base_dir, "disease_management.csv")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            print("Loading existing models...")
            self.index = faiss.read_index(index_path)
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"✅ Loaded {len(self.documents)} diseases")
        elif os.path.exists(csv_path):
            print("Building from CSV...")
            self.build_from_csv(csv_path)
        else:
            print(f"ERROR: No CSV found at {csv_path}")
    
    def build_from_csv(self, csv_path):
        """Build FAISS index from CSV"""
        print(f"Reading CSV from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} rows")
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Prepare documents
        self.documents = []
        for _, row in self.df.iterrows():
            doc = {
                "disease_name": row['disease_name'],
                "crop": row['crop'],
                "chemical": row['chemical_control'],
                "organic": row['organic_control'],
                "prevention": row['prevention'],
                "cultural": row['cultural_practices'],
                "severity": row['severity'],
                "urgency": row['treatment_urgency'],
                "text": f"Disease: {row['disease_name']}. Chemical: {row['chemical_control']}. Organic: {row['organic_control']}"
            }
            self.documents.append(doc)
        
        # Build FAISS
        texts = [doc['text'] for doc in self.documents]
        embeddings = self.embedding_model.encode(texts)
        embeddings = np.array(embeddings).astype('float32')
        
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        
        # Save models
        faiss.write_index(self.index, os.path.join(os.path.dirname(csv_path), "faiss_index.bin"))
        with open(os.path.join(os.path.dirname(csv_path), "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        print(f"✅ Built with {len(self.documents)} diseases")
    
    def query(self, disease_name: str) -> dict:
        """Query the RAG system - returns JSON serializable types"""
        if self.index is None:
            return {"success": False, "message": "Models not loaded"}
        
        try:
            # Embed query
            query_vec = self.embedding_model.encode([disease_name])
            query_vec = np.array(query_vec).astype('float32')
            
            # Search
            distances, indices = self.index.search(query_vec, 1)
            
            if indices[0][0] == -1:
                return {"success": False, "message": "No matching disease found"}
            
            # Convert numpy types to Python native types
            similarity = float(1 / (1 + distances[0][0]))
            disease = self.documents[indices[0][0]]
            
            return {
                "success": True,
                "confidence": similarity,
                "disease_name": str(disease['disease_name']),
                "crop": str(disease['crop']),
                "severity": str(disease['severity']),
                "urgency": str(disease['urgency']),
                "chemical_control": str(disease['chemical']),
                "organic_control": str(disease['organic']),
                "prevention": str(disease['prevention']),
                "cultural_practices": str(disease['cultural'])
            }
        except Exception as e:
            return {"success": False, "message": f"Query error: {str(e)}"}
    
    def get_all_diseases(self) -> List[dict]:
        """Get all diseases"""
        if self.documents is None:
            return []
        return [
            {
                "disease_name": str(d['disease_name']),
                "crop": str(d['crop']),
                "severity": str(d['severity'])
            } 
            for d in self.documents
        ]

# Initialize RAG
print("Initializing RAG system...")
rag = PlantDiseaseRAG()
print(f"RAG initialized: index={rag.index is not None}, docs={len(rag.documents) if rag.documents else 0}")

# ============================================
# API ENDPOINTS
# ============================================

class DiseaseQuery(BaseModel):
    disease_name: str

@app.get("/")
async def root():
    return {
        "service": "Plant Disease RAG System",
        "status": "running",
        "models_loaded": rag.index is not None,
        "diseases_loaded": len(rag.documents) if rag.documents else 0
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": rag.index is not None,
        "diseases_loaded": len(rag.documents) if rag.documents else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/diseases")
async def get_diseases():
    diseases = rag.get_all_diseases()
    return {"total": len(diseases), "diseases": diseases}

@app.post("/query")
async def query_disease(query: DiseaseQuery):
    result = rag.query(query.disease_name)
    if not result.get('success', False):
        raise HTTPException(status_code=404, detail=result.get('message', 'Disease not found'))
    return result

@app.get("/search")
async def search(q: str):
    diseases = rag.get_all_diseases()
    matches = [d for d in diseases if q.lower() in d['disease_name'].lower()]
    return {"query": q, "results": matches[:10], "count": len(matches)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)