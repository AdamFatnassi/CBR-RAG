# src/rag_generator.py
"""
Module de Génération (G) du pipeline RAG utilisant l'API Groq.
Ce module prend le prompt construit par augmentation.py (Partie 2) 
et génère la réponse finale via le LLM.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

from src.augmentation import build_prompt

load_dotenv()

class CBR_RAG_Generator:
    def __init__(self, use_api: bool = True):
        """
        :param use_api: Si True, utilise l'API Groq. Si False, simule.
        """
        self.use_api = use_api
        self.client = None
        self.model_name = "openai/gpt-oss-20b" 
        
        if self.use_api:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("⚠️ Pas de clé GROQ_API_KEY trouvée. Passage en mode simulation.")
                self.use_api = False
            else:
                self.client = OpenAI(
                    api_key=api_key, 
                    base_url="https://api.groq.com/openai/v1"
                )

    def generate_explanation(self, query_row_id: int) -> str:
        """
        Prend l'ID du client (query_row_id), récupère le prompt via augmentation.py,
        et génère l'explication avec Groq.
        """
        # 1. Appel à la partie Augmented (ton collègue)
        system_prompt, user_prompt, decision, truth = build_prompt(query_row_id)
        
        # --- MODE SIMULATION (Si pas de clé API) ---
        if not self.use_api:
            return (
                f"🤖 [MODE SIMULATION LLM]\n"
                f"Décision calculée : {decision}\n"
                f"--- Début du prompt système ---\n{system_prompt[:200]}...\n"
                f"--- Début du prompt utilisateur ---\n{user_prompt[:400]}..."
            )
        
        # --- MODE REEL (APPEL API GROQ) ---
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3, 
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Erreur lors de l'appel à l'API Groq : {str(e)}"