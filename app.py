"""
REAL RAG SYSTEM: Disease Name → LLM Generated Treatment
- FAISS retrieves relevant disease info
- Groq Llama 3 generates custom treatment plan
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

app = FastAPI(title="Plant Disease RAG API - Real Generation", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# REAL RAG SYSTEM WITH GENERATION
# ============================================

class PlantDiseaseRAG:
    def __init__(self):
        self.index = None
        self.documents = None
        self.embedding_model = None
        self.groq_client = None
        
        # Initialize Groq if API key available
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            self.groq_client = Groq(api_key=groq_api_key)
            print("✅ Groq LLM enabled")
        else:
            print("⚠️ GROQ_API_KEY not set - will use template fallback")
        
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
        
        # Prepare rich documents with ALL treatment info
        self.documents = []
        for _, row in self.df.iterrows():
            # Create a rich text document for retrieval
            rich_text = f"""
DISEASE: {row['disease_name']}
CROP: {row['crop']}
SEVERITY: {row['severity']}
SYMPTOMS: {row['symptoms']}
CHEMICAL CONTROL: {row['chemical_control']}
ORGANIC CONTROL: {row['organic_control']}
PREVENTION: {row['prevention']}
CULTURAL PRACTICES: {row['cultural_practices']}
URGENCY: {row['treatment_urgency']}
"""
            doc = {
                "disease_code": row['disease_code'],
                "disease_name": row['disease_name'],
                "crop": row['crop'],
                "severity": row['severity'],
                "symptoms": row['symptoms'],
                "chemical_control": row['chemical_control'],
                "organic_control": row['organic_control'],
                "prevention": row['prevention'],
                "cultural_practices": row['cultural_practices'],
                "urgency": row['treatment_urgency'],
                "action_required": row['action_required'],
                "text": rich_text
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
    
    def retrieve(self, disease_name: str, top_k: int = 2) -> List[dict]:
        """Retrieve most relevant disease documents"""
        
        # First try exact match on disease_code
        for doc in self.documents:
            if doc['disease_code'].lower() == disease_name.lower():
                return [doc]
        
        # Then FAISS similarity search
        query_vec = self.embedding_model.encode([disease_name])
        query_vec = np.array(query_vec).astype('float32')
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                similarity = 1 / (1 + distances[0][i])
                results.append({
                    "document": self.documents[idx],
                    "similarity": similarity,
                    "rank": i + 1
                })
        
        return results
    
    def generate_treatment(self, disease_name: str, retrieved_docs: List) -> str:
        """GENERATION: Use LLM to create custom treatment plan"""
        
        if not retrieved_docs:
            return f"No information found for disease: {disease_name}"
        
        # Build context from retrieved documents
        context = ""
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                d = doc['document']
                context += f"""
DISEASE: {d['disease_name']}
CROP: {d['crop']}
SEVERITY: {d['severity']}
SYMPTOMS: {d['symptoms']}
CHEMICAL: {d['chemical_control']}
ORGANIC: {d['organic_control']}
PREVENTION: {d['prevention']}
CULTURAL: {d['cultural_practices']}
URGENCY: {d['urgency']}
---
"""
            else:
                d = doc
                context += f"""
DISEASE: {d['disease_name']}
CROP: {d['crop']}
SEVERITY: {d['severity']}
CHEMICAL: {d['chemical_control']}
ORGANIC: {d['organic_control']}
PREVENTION: {d['prevention']}
---
"""
        
        # If Groq is available, generate real LLM response
        if self.groq_client:
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
            
            try:
                completion = self.groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "You are an expert agricultural advisor. Provide detailed, practical treatment plans for farmers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4,
                    max_tokens=1500
                )
                
                llm_output = completion.choices[0].message.content
                
                # Add retrieval info
                return f"""
╔══════════════════════════════════════════════════════════════════╗
║         🌾 LLM-GENERATED TREATMENT PLAN (Groq Llama 3) 🌾        ║
║         Disease: {retrieved_docs[0]['document']['disease_name'] if isinstance(retrieved_docs[0], dict) else retrieved_docs[0]['disease_name']}                    
╚══════════════════════════════════════════════════════════════════╝

{llm_output}

╔══════════════════════════════════════════════════════════════════╗
║  📊 Retrieved from: {len(retrieved_docs)} disease record(s)                     ║
║  🤖 Generated by: Groq Llama 3 70B                                        ║
╚══════════════════════════════════════════════════════════════════╝
"""
            except Exception as e:
                print(f"Groq error: {e}")
                return self._template_generation(disease_name, retrieved_docs)
        
        # Fallback to template generation
        return self._template_generation(disease_name, retrieved_docs)
    
    def _template_generation(self, disease_name: str, retrieved_docs: List) -> str:
        """Template-based generation (fallback when LLM unavailable)"""
        
        doc = retrieved_docs[0]['document'] if isinstance(retrieved_docs[0], dict) else retrieved_docs[0]
        
        severity_icon = "🔴" if doc['severity'] in ["Critical", "High"] else "🟡" if doc['severity'] == "Medium" else "🟢"
        
        return f"""
{severity_icon}══════════════════════════════════════════════════════════════════{severity_icon}
                    TREATMENT PLAN (Template Mode)
                    {doc['disease_name'].upper()} on {doc['crop'].upper()}
{severity_icon}══════════════════════════════════════════════════════════════════{severity_icon}

📋 DIAGNOSIS SUMMARY
─────────────────────────────────────────────────────────────────
Disease: {doc['disease_name']}
Crop: {doc['crop']}
Severity: {doc['severity']}
Urgency: {doc['urgency']}

Symptoms: {doc['symptoms']}

🚨 IMMEDIATE ACTION (Next 24 hours)
─────────────────────────────────────────────────────────────────
{doc['action_required']}

💊 CHEMICAL TREATMENT
─────────────────────────────────────────────────────────────────
{doc['chemical_control']}

🌱 ORGANIC ALTERNATIVES
─────────────────────────────────────────────────────────────────
{doc['organic_control']}

🛡️ PREVENTION STRATEGIES
─────────────────────────────────────────────────────────────────
{doc['prevention']}

🏡 CULTURAL PRACTICES
─────────────────────────────────────────────────────────────────
{doc['cultural_practices']}

⚠️ URGENCY LEVEL: {doc['urgency']}

{severity_icon}══════════════════════════════════════════════════════════════════{severity_icon}
💡 TIP: Add GROQ_API_KEY to Railway environment variables for AI-generated responses!
{severity_icon}══════════════════════════════════════════════════════════════════{severity_icon}
"""
    
    def get_treatment(self, disease_name: str) -> dict:
        """Main method: Retrieve + Generate"""
        
        # Step 1: RETRIEVE relevant documents
        retrieved = self.retrieve(disease_name, top_k=2)
        
        if not retrieved:
            return {
                "success": False,
                "message": f"Disease '{disease_name}' not found",
                "suggestions": self.get_suggestions(disease_name)
            }
        
        # Step 2: GENERATE treatment plan
        treatment_plan = self.generate_treatment(disease_name, retrieved)
        
        # Get primary document info
        primary = retrieved[0]['document'] if isinstance(retrieved[0], dict) else retrieved[0]
        
        return {
            "success": True,
            "disease_name": primary['disease_name'],
            "crop": primary['crop'],
            "severity": primary['severity'],
            "treatment_plan": treatment_plan,
            "retrieved_confidence": retrieved[0]['similarity'] if isinstance(retrieved[0], dict) else 0.95
        }
    
    def get_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """Get disease suggestions"""
        suggestions = []
        for doc in self.documents:
            if query.lower() in doc['disease_name'].lower() or query.lower() in doc['crop'].lower():
                suggestions.append(doc['disease_code'])
            if len(suggestions) >= limit:
                break
        return suggestions
    
    def get_all_diseases(self) -> List[dict]:
        """Get all disease codes and names"""
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
print("Initializing REAL RAG with Generation...")
print("="*50)
rag = PlantDiseaseRAG()
print(f"\n✅ Ready: {len(rag.documents) if rag.documents else 0} diseases loaded")
print(f"✅ LLM Generation: {'ENABLED (Groq)' if rag.groq_client else 'DISABLED (template mode)'}")
print("="*50)

# ============================================
# API ENDPOINTS
# ============================================

class DiseaseRequest(BaseModel):
    disease_name: str

@app.get("/")
async def root():
    return {
        "service": "Plant Disease RAG System - REAL Generation",
        "description": "FAISS retrieval + Groq Llama 3 generation",
        "llm_enabled": rag.groq_client is not None,
        "usage": {
            "endpoint": "POST /treatment",
            "body": {"disease_name": "Tomato___Late_blight"},
            "response": "LLM-generated treatment plan"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "diseases_loaded": len(rag.documents) if rag.documents else 0,
        "llm_enabled": rag.groq_client is not None
    }

@app.get("/diseases")
async def list_diseases():
    diseases = rag.get_all_diseases()
    return {"total": len(diseases), "diseases": diseases}

@app.post("/treatment")
async def get_treatment(request: DiseaseRequest):
    """Get LLM-generated treatment plan for a disease"""
    result = rag.get_treatment(request.disease_name)
    
    if not result.get('success', False):
        raise HTTPException(
            status_code=404,
            detail={
                "message": result.get('message'),
                "suggestions": result.get('suggestions', [])
            }
        )
    
    return result

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)