"""
REAL RAG SYSTEM - Disease Name → LLM Generated Treatment
Fixed 500 error
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
from groq import Groq

app = FastAPI(title="Plant Disease RAG API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# RAG SYSTEM WITH GROQ
# ============================================

class PlantDiseaseRAG:
    def __init__(self):
        self.index = None
        self.documents = None
        self.embedding_model = None
        self.groq_client = None
        
        # Initialize Groq if API key available
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key and groq_api_key != "":
            try:
                self.groq_client = Groq(api_key=groq_api_key)
                print("✅ Groq LLM enabled")
            except Exception as e:
                print(f"⚠️ Groq init error: {e}")
                self.groq_client = None
        else:
            print("⚠️ GROQ_API_KEY not set - using template mode")
        
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
                "disease_code": str(row['disease_code']),
                "disease_name": str(row['disease_name']),
                "crop": str(row['crop']),
                "severity": str(row['severity']),
                "symptoms": str(row['symptoms']),
                "chemical_control": str(row['chemical_control']),
                "organic_control": str(row['organic_control']),
                "prevention": str(row['prevention']),
                "cultural_practices": str(row['cultural_practices']),
                "urgency": str(row['treatment_urgency']),
                "action_required": str(row['action_required']),
                "text": f"{row['disease_name']} {row['crop']} {row['symptoms']} {row['chemical_control']}"
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
    
    def get_disease_by_name(self, disease_name: str):
        """Find disease by code or name"""
        # Exact match on disease_code
        for doc in self.documents:
            if doc['disease_code'].lower() == disease_name.lower():
                return doc, 1.0
        
        # Exact match on disease_name
        for doc in self.documents:
            if doc['disease_name'].lower() == disease_name.lower():
                return doc, 0.95
        
        # FAISS similarity search
        try:
            query_vec = self.embedding_model.encode([disease_name])
            query_vec = np.array(query_vec).astype('float32')
            distances, indices = self.index.search(query_vec, 1)
            
            if indices[0][0] != -1:
                similarity = float(1 / (1 + distances[0][0]))
                doc = self.documents[indices[0][0]]
                return doc, similarity
        except Exception as e:
            print(f"Search error: {e}")
        
        return None, 0
    
    def generate_treatment(self, disease_name: str, doc: dict, confidence: float) -> str:
        """Generate treatment plan using Groq"""
        
        # If Groq is not available, use template
        if not self.groq_client:
            return self._template_treatment(doc)
        
        try:
            prompt = f"""You are an expert agricultural advisor. Generate a COMPLETE TREATMENT PLAN for the disease below.

DISEASE QUERY: {disease_name}

RETRIEVED INFORMATION:
{context}

Generate a response with these exact sections:

1. DIAGNOSIS SUMMARY: (2-3 sentences identifying the disease and risk level)

2. IMMEDIATE ACTION (Next 24 hours):
   - What the farmer must do TODAY
   - Specific steps with timing

3. CHEMICAL TREATMENT:
   - Specific product names
   - Application rates and frequency
   - Safety precautions

4. ORGANIC ALTERNATIVES:
   - Natural treatment options
   - Homemade remedy recipes (with ratios)
   - Where to purchase products

5. PREVENTION STRATEGIES:
   - How to stop spread
   - Long-term prevention

6. URGENCY LEVEL: (Critical/High/Medium/Low - explain why)

Make it practical, specific, and actionable for a farmer. Use bullet points.

TREATMENT PLAN:
"""
            
            completion = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are an agricultural expert. Provide practical treatment plans."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Groq error: {e}")
            return self._template_treatment(doc)
    
    def _template_treatment(self, doc: dict) -> str:
        """Template fallback"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    TREATMENT PLAN                            ║
╚══════════════════════════════════════════════════════════════╝

DISEASE: {doc['disease_name']}
CROP: {doc['crop']}
SEVERITY: {doc['severity']}
URGENCY: {doc['urgency']}

SYMPTOMS:
{doc['symptoms']}

CHEMICAL TREATMENT:
{doc['chemical_control']}

ORGANIC TREATMENT:
{doc['organic_control']}

PREVENTION:
{doc['prevention']}

CULTURAL PRACTICES:
{doc['cultural_practices']}
"""
    
    def get_treatment(self, disease_name: str) -> dict:
        """Main method - get treatment plan"""
        
        doc, confidence = self.get_disease_by_name(disease_name)
        
        if not doc:
            return {
                "success": False,
                "error": f"Disease '{disease_name}' not found"
            }
        
        # Generate treatment plan
        treatment_plan = self.generate_treatment(disease_name, doc, confidence)
        
        return {
            "success": True,
            "disease_name": doc['disease_name'],
            "crop": doc['crop'],
            "severity": doc['severity'],
            "confidence": confidence,
            "treatment_plan": treatment_plan
        }
    
    def get_all_diseases(self) -> List[dict]:
        return [
            {
                "disease_code": doc['disease_code'],
                "disease_name": doc['disease_name'],
                "crop": doc['crop']
            }
            for doc in self.documents
        ]


# Initialize RAG
print("="*50)
print("Initializing RAG System...")
print("="*50)
rag = PlantDiseaseRAG()
print(f"✅ Loaded {len(rag.documents)} diseases")
print(f"✅ Groq: {'ENABLED' if rag.groq_client else 'DISABLED'}")
print("="*50)

# ============================================
# API ENDPOINTS
# ============================================

class DiseaseRequest(BaseModel):
    disease_name: str

@app.get("/")
async def root():
    return {
        "service": "Plant Disease RAG API",
        "llm_enabled": rag.groq_client is not None,
        "diseases_loaded": len(rag.documents)
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "diseases_loaded": len(rag.documents),
        "llm_enabled": rag.groq_client is not None
    }

@app.get("/diseases")
async def list_diseases():
    return {"total": len(rag.documents), "diseases": rag.get_all_diseases()}

@app.post("/treatment")
async def treatment(request: DiseaseRequest):
    try:
        result = rag.get_treatment(request.disease_name)
        
        if not result.get('success'):
            raise HTTPException(status_code=404, detail=result.get('error'))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)



