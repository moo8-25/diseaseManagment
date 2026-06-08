"""
RAG System Deployment on Railway
FastAPI + FAISS + Sentence Transformers
"""

import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from datetime import datetime

# ============================================
# RAG SYSTEM CLASS
# ============================================

class PlantDiseaseRAG:
    def __init__(self, model_dir: str = "rag_models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.faiss_path = os.path.join(model_dir, "faiss_index.bin")
        self.documents_path = os.path.join(model_dir, "documents.pkl")
        
        self.index = None
        self.documents = None
        self.embedding_model = None
        
        # Try to load existing models
        if not self.load_models():
            print("No models found. Building from CSV...")
            self.build("disease_management.csv")
    
    def build(self, csv_path: str):
        """Build FAISS index from CSV"""
        print("Building RAG models...")
        
        # Load data
        self.df = pd.read_csv(csv_path)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Prepare documents
        self.documents = []
        for _, row in self.df.iterrows():
            doc = {
                "disease_code": row['disease_code'],
                "disease_name": row['disease_name'],
                "crop": row['crop'],
                "chemical": row['chemical_control'],
                "organic": row['organic_control'],
                "prevention": row['prevention'],
                "cultural": row['cultural_practices'],
                "severity": row['severity'],
                "urgency": row['treatment_urgency'],
                "text": f"""
                DISEASE: {row['disease_name']}
                CROP: {row['crop']}
                SEVERITY: {row['severity']}
                CHEMICAL: {row['chemical_control']}
                ORGANIC: {row['organic_control']}
                PREVENTION: {row['prevention']}
                """
            }
            self.documents.append(doc)
        
        # Build FAISS
        texts = [doc['text'] for doc in self.documents]
        embeddings = self.embedding_model.encode(texts)
        embeddings = np.array(embeddings).astype('float32')
        
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        
        # Save models
        self.save_models()
        print(f"✅ Built: {len(self.documents)} diseases indexed")
    
    def save_models(self):
        """Save models to disk"""
        faiss.write_index(self.index, self.faiss_path)
        with open(self.documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
        print("✅ Models saved")
    
    def load_models(self) -> bool:
        """Load models from disk"""
        if not os.path.exists(self.faiss_path) or not os.path.exists(self.documents_path):
            return False
        
        try:
            self.index = faiss.read_index(self.faiss_path)
            with open(self.documents_path, 'rb') as f:
                self.documents = pickle.load(f)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"✅ Loaded {len(self.documents)} diseases")
            return True
        except Exception as e:
            print(f"Error loading: {e}")
            return False
    
    def retrieve(self, query: str, k: int = 1) -> dict:
        """Retrieve most relevant disease"""
        query_vec = self.embedding_model.encode([query])
        query_vec = np.array(query_vec).astype('float32')
        
        distances, indices = self.index.search(query_vec, k)
        
        if indices[0][0] == -1:
            return None
        
        similarity = 1 / (1 + distances[0][0])
        return {
            "disease": self.documents[indices[0][0]],
            "similarity": similarity
        }
    
    def query(self, disease_name: str) -> dict:
        """Query RAG system"""
        result = self.retrieve(disease_name)
        
        if not result or result['similarity'] < 0.4:
            return {
                "success": False,
                "message": f"Disease '{disease_name}' not found with high confidence",
                "suggestions": self.get_suggestions(disease_name)
            }
        
        disease = result['disease']
        return {
            "success": True,
            "confidence": result['similarity'],
            "disease_name": disease['disease_name'],
            "crop": disease['crop'],
            "severity": disease['severity'],
            "urgency": disease['urgency'],
            "chemical_control": disease['chemical'],
            "organic_control": disease['organic'],
            "prevention": disease['prevention'],
            "cultural_practices": disease['cultural']
        }
    
    def get_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """Get disease suggestions"""
        suggestions = []
        for doc in self.documents:
            if query.lower() in doc['disease_name'].lower() or query.lower() in doc['crop'].lower():
                suggestions.append(doc['disease_name'])
                if len(suggestions) >= limit:
                    break
        return suggestions
    
    def get_all_diseases(self) -> List[dict]:
        """Get all diseases"""
        return [
            {
                "disease_name": doc['disease_name'],
                "crop": doc['crop'],
                "severity": doc['severity']
            }
            for doc in self.documents
        ]


# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="🌱 Plant Disease RAG System API",
    description="Retrieval-Augmented Generation for Plant Disease Management",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag = PlantDiseaseRAG()

# Request/Response Models
class DiseaseQuery(BaseModel):
    disease_name: str

class DiseaseResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    confidence: Optional[float] = None
    disease_name: Optional[str] = None
    crop: Optional[str] = None
    severity: Optional[str] = None
    urgency: Optional[str] = None
    chemical_control: Optional[str] = None
    organic_control: Optional[str] = None
    prevention: Optional[str] = None
    cultural_practices: Optional[str] = None
    suggestions: Optional[List[str]] = None


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "🌱 Plant Disease RAG System",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This info",
            "GET /health": "Health check",
            "GET /diseases": "List all diseases",
            "POST /query": "Query disease management",
            "GET /disease/{name}": "Get disease by name"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": rag.index is not None,
        "total_diseases": len(rag.documents) if rag.documents else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/diseases")
async def get_all_diseases(limit: int = 100, crop: Optional[str] = None):
    """Get all diseases with optional filtering"""
    diseases = rag.get_all_diseases()
    
    if crop:
        diseases = [d for d in diseases if d['crop'].lower() == crop.lower()]
    
    return {
        "total": len(diseases),
        "diseases": diseases[:limit]
    }

@app.get("/disease/{disease_name}")
async def get_disease_by_name(disease_name: str):
    """Get management plan for a disease"""
    result = rag.query(disease_name)
    
    if not result['success']:
        raise HTTPException(status_code=404, detail=result['message'])
    
    return result

@app.post("/query")
async def query_disease(query: DiseaseQuery):
    """Query disease management (POST method)"""
    result = rag.query(query.disease_name)
    
    if not result['success']:
        raise HTTPException(status_code=404, detail=result['message'])
    
    return result

@app.get("/search")
async def search_diseases(q: str):
    """Search for diseases"""
    suggestions = rag.get_suggestions(q)
    return {
        "query": q,
        "results": suggestions,
        "count": len(suggestions)
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    diseases = rag.get_all_diseases()
    severity_counts = {}
    crop_counts = {}
    
    for d in diseases:
        severity = d['severity']
        crop = d['crop']
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        crop_counts[crop] = crop_counts.get(crop, 0) + 1
    
    return {
        "total_diseases": len(diseases),
        "severity_distribution": severity_counts,
        "crop_distribution": crop_counts,
        "models_loaded": rag.index is not None
    }


# ============================================
# FOR LOCAL DEVELOPMENT
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)