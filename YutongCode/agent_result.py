from openai import OpenAI
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass
import os
from tqdm import tqdm
from pathlib import Path
from AdaptiveDialogueAgent import OptimizedMemoryAgent  # 添加这个导入

def clean_patient_id(patient_id: str) -> str:
    """Remove '_dialogues' suffix from patient ID if present"""
    return patient_id.replace('_dialogues', '')



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
        self.MAX_RESPONSE_LENGTH = 100
        
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
        5. Emphasize social connections by:
        - Mentioning family members and friends by name when relevant
        - Acknowledging shared activities and relationships
        - Referencing positive social memories
        - Validating feelings about relationships
        - Highlighting the presence of supportive people
        6. Maintain emotional attunement by:
        - Acknowledging their feelings about people and relationships
        - Offering comfort through familiar social connections
        - Recognizing the importance of their relationships
        - Supporting their sense of belonging
        - Providing reassurance through social context
        7. When discussing activities or memories:
        - Highlight the social aspects
        - Mention who was/is involved
        - Reference shared experiences
        - Acknowledge important relationships
        - Connect past social experiences to present
        8. The response is 2-5 sentence long

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




def integrated_memory_search(
    patient_id: str, 
    api_key: str,
    dialogue: str,
    current_time: datetime
) -> Dict[str, Any]:
    """Integrate memory search and response generation"""
    # Clean patient ID before using it
    clean_id = clean_patient_id(patient_id)
    
    try:
        # Initialize agents
        memory_agent = OptimizedMemoryAgent(
            patient_id=clean_id,  
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
    except Exception as e:
        print(f"Error in integrated_memory_search for patient {clean_id}: {e}")
        # Return a simplified response when there's an error
        return {
            "query": {
                "text": dialogue,
                "time": current_time.strftime("%I:%M %p")
            },
            "search_results": {},
            "response": "I'm here to help. Let's talk about that.",
            "search_performance": {}
        }


def load_and_process_dialogues(json_path: str) -> List[Dict[str, str]]:
    """Load dialogues from a patient's JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        dialogues = []
        for entry in data['dialogues']:
            dialogues.append({
                'time': entry['time'],
                'query': entry['query']['text']
            })
        return dialogues
    except Exception as e:
        print(f"Error loading dialogues from {json_path}: {e}")
        return []

def process_and_save_test_results(
    patient_data_dir: str,
    output_dir: str,
    api_key: str
) -> None:
    """Process patient files and save test results"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each JSON file
    for json_file in tqdm(list(Path(patient_data_dir).glob("*.json")), desc="Processing patient files"):
        patient_id = json_file.stem
        
        # Load dialogues
        dialogues = load_and_process_dialogues(str(json_file))
        
        if not dialogues:
            continue
            
        # Prepare test results
        test_results = {
            "patient_id": patient_id,
            "test_cases": []
        }
        
        # Process each dialogue
        for dialogue in tqdm(dialogues, desc=f"Processing dialogues for {patient_id}", leave=False):
            try:
                # Convert time string to datetime
                current_time = datetime.strptime(dialogue['time'], "%H:%M")
                
                # Generate response using integrated memory search
                results = integrated_memory_search(
                    patient_id=patient_id,
                    api_key=api_key,
                    dialogue=dialogue['query'],
                    current_time=current_time
                )
                
                # Add to test results
                test_results["test_cases"].append({
                    "time": dialogue['time'],
                    "question": dialogue['query'],
                    "response": results["response"],
                    "search_results": results["search_results"]  # 保留搜索结果
                })
                
            except Exception as e:
                print(f"Error processing dialogue for {patient_id} at {dialogue['time']}: {e}")
                test_results["test_cases"].append({
                    "time": dialogue['time'],
                    "question": dialogue['query'],
                    "response": f"Error: {str(e)}",
                    "search_results": {}
                })
        
        # Save test results
        output_path = os.path.join(output_dir, f"{patient_id}_test_results.json")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)
            print(f"\nTest results saved for {patient_id}")
        except Exception as e:
            print(f"Error saving test results for {patient_id}: {e}")

def main():
    # Configuration
    API_KEY = "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"
    PATIENT_DATA_DIR = "Patient_Data/Ground_Truth"
    OUTPUT_DIR = "test_results"
    
    # Process and save test results
    process_and_save_test_results(PATIENT_DATA_DIR, OUTPUT_DIR, API_KEY)
    
    print("\nProcessing complete!")
    print(f"Test results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()