"""
REAL RAG System for Plant Disease Management
- Multi-document retrieval with FAISS
- LLM generation with Groq (Llama 3)
- API key from environment variable (Railway)
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================
# REAL RAG SYSTEM
# ============================================

class RealPlantDiseaseRAG:
    def __init__(self, csv_path: str = "disease_management.csv"):
        """Initialize REAL RAG system"""
        
        print("="*50)
        print("🚀 INITIALIZING REAL RAG SYSTEM")
        print("="*50)
        
        # Load data
        self.df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(self.df)} disease records")
        
        # Embedding model
        print("📦 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
        
        # Prepare rich documents
        self.documents = []
        for _, row in self.df.iterrows():
            rich_text = f"""
            DISEASE INFORMATION
            ===================
            Disease Name: {row['disease_name']}
            Crop: {row['crop']}
            Severity: {row['severity']}
            Status: {row['status']}
            Urgency: {row['treatment_urgency']}
            
            SYMPTOMS:
            {row['symptoms']}
            
            CHEMICAL CONTROL:
            {row['chemical_control']}
            
            ORGANIC CONTROL:
            {row['organic_control']}
            
            PREVENTION METHODS:
            {row['prevention']}
            
            CULTURAL PRACTICES:
            {row['cultural_practices']}
            
            ACTION REQUIRED:
            {row['action_required']}
            """
            
            self.documents.append({
                "id": row['disease_code'],
                "text": rich_text,
                "metadata": {
                    "disease_name": row['disease_name'],
                    "crop": row['crop'],
                    "severity": row['severity'],
                    "chemical": row['chemical_control'],
                    "organic": row['organic_control'],
                    "prevention": row['prevention'],
                    "cultural": row['cultural_practices'],
                    "symptoms": row['symptoms'],
                    "urgency": row['treatment_urgency']
                }
            })
        
        # Build FAISS index
        print("🔍 Building FAISS index...")
        texts = [doc['text'] for doc in self.documents]
        embeddings = self.embedding_model.encode(texts)
        embeddings = np.array(embeddings).astype('float32')
        
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"✅ FAISS index built with {len(self.documents)} documents")
        
        # Initialize Groq client (READS FROM ENVIRONMENT VARIABLE)
        groq_api_key = os.environ.get("GROQ_API_KEY")
        
        if groq_api_key:
            print("✅ Groq API key found - LLM generation ENABLED")
            self.groq_client = Groq(api_key=groq_api_key)
            self.llm_enabled = True
        else:
            print("⚠️ GROQ_API_KEY not set - using template mode")
            print("   Add GROQ_API_KEY to Railway environment variables")
            self.llm_enabled = False
            self.groq_client = None
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve top-k most relevant documents"""
        
        query_vec = self.embedding_model.encode([query])
        query_vec = np.array(query_vec).astype('float32')
        
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                similarity = 1 / (1 + distances[0][i])
                results.append({
                    "document": self.documents[idx],
                    "similarity": float(similarity),
                    "rank": i + 1
                })
        
        return results
    
    def generate(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Generate response using Groq Llama 3.1 (if enabled)"""
        
        if not self.llm_enabled or not self.groq_client:
            return self._fallback_generation(query, retrieved_docs)
        
        # Build context from retrieved documents
        context = ""
        for doc in retrieved_docs:
            context += f"\n--- SOURCE {doc['rank']} (Relevance: {doc['similarity']:.2%}) ---\n"
            context += doc['document']['text']
            context += "\n"
        
        prompt = f"""You are an expert agricultural advisor with 20 years of experience. Use the retrieved information to answer the user's query.

USER QUERY: {query}

RETRIEVED INFORMATION:
{context}

INSTRUCTIONS:
1. Synthesize information from ALL retrieved sources
2. If information conflicts, explain the different approaches
3. Provide a COMPLETE, ACTIONABLE response with:
   - Quick diagnosis summary
   - Immediate actions (first 24-48 hours)
   - Chemical treatment options with specific products
   - Organic/natural alternatives
   - Prevention strategies
   - When to call a professional
4. Be specific: include product names, mixing ratios, application timing
5. Use bullet points for clarity
6. If the query is about a healthy plant, confirm and give maintenance tips

RESPONSE:
"""
        
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are an expert agricultural advisor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            return f"⚠️ LLM Error: {str(e)}\n\n{self._fallback_generation(query, retrieved_docs)}"
    
    def _fallback_generation(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Fallback when LLM unavailable"""
        
        if not retrieved_docs:
            return f"No information found for: {query}"
        
        best = retrieved_docs[0]['document']
        meta = best['metadata']
        
        return f"""
╔══════════════════════════════════════════════════════════════╗
║              PLANT DISEASE MANAGEMENT REPORT                 ║
║                    (Template Mode)                           ║
╚══════════════════════════════════════════════════════════════╝

📋 DISEASE: {meta['disease_name']}
🌾 CROP: {meta['crop']}
⚠️  SEVERITY: {meta['severity']}
⚡ URGENCY: {meta['urgency']}
📊 CONFIDENCE: {retrieved_docs[0]['similarity']:.1%}

🔬 SYMPTOMS:
{meta['symptoms']}

💊 CHEMICAL TREATMENT:
{meta['chemical']}

🌱 ORGANIC TREATMENT:
{meta['organic']}

🛡️ PREVENTION:
{meta['prevention']}

🏡 CULTURAL PRACTICES:
{meta['cultural']}

{'='*60}
💡 TIP: Add GROQ_API_KEY to enable AI-powered responses!
"""
    
    def chat(self, user_input: str) -> Dict:
        """Complete RAG pipeline: Retrieve → Generate"""
        
        retrieved = self.retrieve(user_input, k=3)
        
        if not retrieved:
            return {
                "response": f"No relevant information found for: {user_input}",
                "retrieved_docs": [],
                "llm_enabled": self.llm_enabled
            }
        
        response = self.generate(user_input, retrieved)
        
        return {
            "response": response,
            "retrieved_docs": [
                {
                    "disease": r['document']['metadata']['disease_name'],
                    "crop": r['document']['metadata']['crop'],
                    "similarity": r['similarity']
                }
                for r in retrieved
            ],
            "llm_enabled": self.llm_enabled
        }


# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(title="Real Plant Disease RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG
print("\n" + "="*50)
print("🔧 INITIALIZING RAG SYSTEM")
print("="*50)
rag = RealPlantDiseaseRAG()
print("\n✅ API Ready!")
print(f"   LLM Enabled: {rag.llm_enabled}")
print("="*50 + "\n")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {
        "service": "Real Plant Disease RAG System v2",
        "llm_enabled": rag.llm_enabled,
        "endpoints": {
            "POST /chat": "Chat with AI",
            "GET /health": "Health check",
            "GET /diseases": "List all diseases",
            "GET /status": "System status"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "documents_loaded": len(rag.documents),
        "llm_enabled": rag.llm_enabled
    }

@app.get("/status")
async def status():
    return {
        "llm_enabled": rag.llm_enabled,
        "total_diseases": len(rag.documents),
        "message": "LLM generation is " + ("ENABLED" if rag.llm_enabled else "DISABLED - Add GROQ_API_KEY")
    }

@app.get("/diseases")
async def list_diseases():
    diseases = [doc['metadata']['disease_name'] for doc in rag.documents]
    return {"total": len(diseases), "diseases": diseases}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = rag.chat(request.message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)