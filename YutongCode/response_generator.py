from openai import OpenAI
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
from AdaptiveDialogueAgent import OptimizedMemoryAgent
from dataclasses import dataclass

@dataclass
class ResponseContext:
    """Context information for response generation"""
    dialogue: str
    current_time: datetime
    search_results: Dict[str, Any]
    patient_profile: Dict[str, Any]

class EmpatheticResponseGenerator:
    """Generates concise, empathetic responses for dementia patients"""
    
    def __init__(self, api_key: str, model: str = "GPT-4o mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.MAX_RESPONSE_LENGTH = 100  # Maximum characters for main response
        
    def generate_response(self, context: ResponseContext) -> str:
        """Generate empathetic response with social connection focus"""
        prompt = f"""As a compassionate care assistant, provide a warm, person-centered response to someone with dementia, emphasizing their social connections and relationships.

        Context Information:
        Dialogue: "{context.dialogue}"
        Time: {context.current_time.strftime('%I:%M %p')}
        Current Activity: {json.dumps(context.search_results.get('current_activity', {}))}
        People Information: {json.dumps(context.search_results.get('people_info', []))}
        Memories: {json.dumps(context.search_results.get('memories', []))}

        Response Guidelines:

        1. Keep response under {self.MAX_RESPONSE_LENGTH} characters
        2. Use warm, simple, and clear language
        3. Provide gentle orientation to time/place when needed
        4. Reference familiar daily routines and activities
        6. Emphasize social connections by:
        - Mentioning family members and friends by name when relevant
        - Acknowledging shared activities and relationships
        - Referencing positive social memories
        - Validating feelings about relationships
        - Highlighting the presence of supportive people
        7. Maintain emotional attunement by:
        - Acknowledging their feelings about people and relationships
        - Offering comfort through familiar social connections
        - Recognizing the importance of their relationships
        - Supporting their sense of belonging
        - Providing reassurance through social context
        8. When discussing activities or memories:
        - Highlight the social aspects
        - Mention who was/is involved
        - Reference shared experiences
        - Acknowledge important relationships
        - Connect past social experiences to present
        9. The response is 2-5 sentence long

        Remember to:
        - Reference actual schedule/people
        - Provide reassurance with facts
        - Use positive memories as anchors
        - Validate emotions while redirecting
        - Prioritize emotional connection over factual correction
        - Use familiar names and relationships
        - Validate their social bonds and experiences
        - Maintain a supportive and understanding tone
        - Focus on positive social connections
 
        
        Return a concise, warm response that validates their experience and emphasizes their social connections."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a compassionate care assistant helping someone with dementia, focusing on maintaining their social connections."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm here with you. Let's talk about that."


            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Generate a brief, empathetic response."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=50 # Limit token length
                )
                
                # Ensure response fits length limit
                response_text = response.choices[0].message.content
                if len(response_text) > self.MAX_RESPONSE_LENGTH:
                    response_text = response_text[:self.MAX_RESPONSE_LENGTH] + "..."
                    
                return response_text
                
            except Exception as e:
                print(f"Error generating response: {e}")
                return "I understand your question. Let me help you with that."

def integrated_memory_search(patient_id: str, 
                           api_key: str,
                           dialogue: str,
                           current_time: datetime) -> Dict[str, Any]:
    """Integrate memory search and response generation"""
    # Initialize agents
    memory_agent = OptimizedMemoryAgent(
        patient_id=patient_id,
        api_key=api_key
    )
    response_generator = EmpatheticResponseGenerator(api_key=api_key)
    
    # Get search results
    search_results = memory_agent.process_query(dialogue, current_time)
    
    # Create response context
    context = ResponseContext(
        dialogue=dialogue,
        current_time=current_time,
        search_results={
            'current_activity': search_results['search_results']['current_activity'],
            'people_info': search_results['search_results']['people_info'],
            'memories': search_results['search_results']['memories']
        },
        patient_profile=search_results.get('patient_profile', {})
    )
    
    # Generate response
    response = response_generator.generate_response(context)
    
    # Compile final output
    return {
        "query": {
            "text": dialogue,
            "time": current_time.strftime("%I:%M %p")
        },
        "search_results": search_results['search_results'],
        "response": response,
        "search_performance": search_results['search_performance']
    }

def main():
    # Test settings
    PATIENT_ID = "P002"
    API_KEY = "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"
    TEST_TIME = datetime.strptime("11:00", "%H:%M")
    TEST_DIALOGUE = "Is Mary coming today? She usually brings cookies... I was looking forward to that."
    
    # Run integrated search and response generation
    results = integrated_memory_search(
        patient_id=PATIENT_ID,
        api_key=API_KEY,
        dialogue=TEST_DIALOGUE,
        current_time=TEST_TIME
    )
    
    # Print results
    print("\nSearch Results and Response:")
    print(json.dumps(results, indent=2))
    
    # Print just the response
    print("\nGenerated Response:")
    print(results['response'])

if __name__ == "__main__":
    main()
