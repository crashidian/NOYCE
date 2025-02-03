from openai import OpenAI
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass
import networkx as nx
import pathlib

@dataclass
class DialogueContext:
    """Context information from dialogue"""
    text: str
    current_time: datetime
    current_activity: Optional[Dict[str, Any]]

@dataclass
class SearchWeights:
    """Search weights for different types of information"""
    routine_weight: float = 0.9
    story_weight: float = 0.1

    def adjust(self, story_weight: float):
        """Adjust weights while maintaining sum of 1.0"""
        self.story_weight = min(1.0, max(0.0, story_weight))
        self.routine_weight = 1.0 - self.story_weight

@dataclass
class SearchResult:
    """Structure for search results"""
    activity_info: Optional[Dict[str, Any]]
    people_info: List[Dict[str, Any]]
    related_memories: List[Dict[str, Any]]

@dataclass
class SemanticContext:
    """Enhanced context with semantic expansions"""
    original_keywords: Dict[str, List[str]]
    expanded_keywords: Dict[str, List[str]]
    relationships: Dict[str, List[Dict[str, str]]]

class MemoryGraphManager:
    def __init__(self, patient_id: str, base_path: str = "Patient_Data"):
        self.base_path = pathlib.Path(base_path)
        self.patient_id = patient_id
        self.load_patient_data()
        self.memory_network = self.construct_memory_networks()

    # def load_patient_data(self):
    #     """Load patient data files"""
    #     story_path = self.base_path / "Life_Stories" / f"{self.patient_id}_story.json"
    #     routine_path = self.base_path / "Daily_Routines" / f"{self.patient_id}_routine.json"
        
    #     with open(story_path) as f:
    #         self.story_data = json.load(f)
    #     with open(routine_path) as f:
    #         self.routine_data = json.load(f)
            
    #     self.profile = self.story_data['profile']
    def load_patient_data(self):
        """Load patient data files including profile, story, and routine data."""
        profile_path = self.base_path / "Profiles" / f"{self.patient_id}_profile.json"
        story_path = self.base_path / "Life_Stories" / f"{self.patient_id}_story.json"
        routine_path = self.base_path / "Daily_Routines" / f"{self.patient_id}_routine.json"
        
        try:
            # Load profile
            with open(profile_path) as f:
                self.profile = json.load(f)
                
            # Load life story
            with open(story_path) as f:
                self.story_data = json.load(f)
                
            # Load daily routine
            with open(routine_path) as f:
                self.routine_data = json.load(f)
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find required data files for patient {self.patient_id}: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON data for patient {self.patient_id}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading data for patient {self.patient_id}: {str(e)}")   

    def construct_memory_networks(self) -> nx.MultiDiGraph:
        """Build memory network"""
        G = nx.MultiDiGraph()
        
        # Add routine activities
        for activity in self.routine_data['activities']:
            activity_id = f"activity_{activity['activity']}"
            G.add_node(activity_id, type='activity', **activity)
            
            # Connect participants
            for person in activity.get('participants', []):
                person_id = f"person_{person['name']}"
                G.add_node(person_id, type='person', **person)
                G.add_edge(activity_id, person_id, type='involves')
                
        # Add memories
        for interview in self.story_data['interviews']:
            for memory in interview['memories']:
                memory_id = f"memory_{memory.get('id', hash(memory['description']))}"
                G.add_node(memory_id, type='memory', **memory)
                
                # Connect people mentioned in memory
                if 'people' in memory:
                    for person in memory['people']:
                        person_id = f"person_{person}"
                        if person_id in G:
                            G.add_edge(memory_id, person_id, type='involves')
                
        return G
        
    def get_current_activity(self, current_time: datetime) -> Optional[Dict[str, Any]]:
        """Get activity for current time"""
        time_str = current_time.strftime("%H:%M")
        
        for activity in self.routine_data['activities']:
            if activity['time_start'] <= time_str <= activity['time_end']:
                return activity
        return None

    def search_memories(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search memories based on keywords"""
        memories = []
        
        for interview in self.story_data['interviews']:
            for memory in interview['memories']:
                memory_text = f"{memory['description']} {memory.get('details', '')}".lower()
                if any(kw.lower() in memory_text for kw in keywords):
                    memories.append({
                        'year': memory.get('year'),
                        'description': memory['description'],
                        'details': memory.get('details', ''),
                        'people': memory.get('people', [])
                    })
        
        return memories

class OptimizedMemoryAgent:
    """Optimized agent combining adaptive dialogue and memory search"""
    
    def __init__(self, patient_id: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.memory_graph = MemoryGraphManager(patient_id)
        self.weights = SearchWeights()

    def process_query(self, dialogue: str, current_time: datetime) -> Dict[str, Any]:
        """Process query and return relevant information"""
        # Get current context
        current_activity = self.memory_graph.get_current_activity(current_time)
        context = DialogueContext(dialogue, current_time, current_activity)
        
        # Analyze dialogue for search strategy
        search_strategy = self._analyze_dialogue(context)
        
        # Initial search with current weights
        results = self._execute_search(context, search_strategy)
        
        # Evaluate and adjust if needed
        effectiveness = self._evaluate_results(context, search_strategy, results)
        
        if effectiveness['score'] < 0.7:  # Threshold for acceptable results
            self.weights.adjust(self.weights.story_weight + 0.2)
            results = self._execute_search(context, search_strategy)
        
        return self._format_response(context, search_strategy, results, effectiveness)

    def _expand_semantic_context(self, keywords: Dict[str, List[str]]) -> SemanticContext:
        """Expand semantic context using LLM"""
        prompt = f"""Given these keywords and the patient's background, expand the semantic context:
        Original Keywords: {json.dumps(keywords)}
        Patient Profile: {json.dumps(self.memory_graph.profile)}
        
        For each category, provide:
        1. Related terms (e.g., "son" -> ["Tom", "my boy", "my child"])
        2. Semantic relationships (e.g., "Italy" -> {{"location": "vacation spot", "memory": "family trip"}})
        
        Consider:
        - Family relationships (son, daughter, etc.)
        - Locations and associated activities
        - Time references and routines
        - Common synonyms and related terms
        
        Return a JSON with:
        {{
            "expanded_keywords": {{
                "people": [],
                "activities": [],
                "time_refs": [],
                "locations": []
            }},
            "relationships": [
                {{"term": "", "category": "", "related_to": "", "relationship_type": ""}}
            ]
        }}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Expand semantic context for memory search."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            expansion = json.loads(response.choices[0].message.content)
            return SemanticContext(
                original_keywords=keywords,
                expanded_keywords=expansion["expanded_keywords"],
                relationships=self._process_relationships(expansion["relationships"])
            )
        except Exception as e:
            print(f"Error in semantic expansion: {e}")
            return SemanticContext(
                original_keywords=keywords,
                expanded_keywords=keywords,
                relationships={}
            )

    def _process_relationships(self, relationships: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """Process and organize semantic relationships"""
        processed = {}
        for rel in relationships:
            term = rel["term"]
            if term not in processed:
                processed[term] = []
            processed[term].append({
                "category": rel["category"],
                "related_to": rel["related_to"],
                "type": rel["relationship_type"]
            })
        return processed

    def _analyze_dialogue(self, context: DialogueContext) -> Dict[str, Any]:
        """Enhanced dialogue analysis with semantic expansion"""
        prompt = f"""Analyze this dialogue and extract key elements:
        Dialogue: "{context.text}"
        Current Time: {context.current_time.strftime("%H:%M")}
        Current Activity: {context.current_activity['activity'] if context.current_activity else 'None'}
        
        Return a JSON with initial keywords:
        {{
            "keywords": {{
                "people": [],
                "activities": [],
                "time_refs": [],
                "locations": []
            }},
            "focus": "routine" or "memory" or "both",
            "temporal_context": "current" or "past" or "both"
        }}"""
        
        try:
            # Get initial analysis
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Analyze dialogue for search strategy."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            initial_analysis = json.loads(response.choices[0].message.content)
            
            # Expand semantic context
            semantic_context = self._expand_semantic_context(initial_analysis["keywords"])
            
            # Merge original analysis with semantic expansions
            return {
                "keywords": semantic_context.expanded_keywords,
                "semantic_relationships": semantic_context.relationships,
                "focus": initial_analysis["focus"],
                "temporal_context": initial_analysis["temporal_context"]
            }
            
        except Exception as e:
            print(f"Error in dialogue analysis: {e}")
            return {
                "keywords": {"people": [], "activities": [], "time_refs": [], "locations": []},
                "semantic_relationships": {},
                "focus": "both",
                "temporal_context": "both"
            }

    def _execute_search(self, context: DialogueContext,
                       strategy: Dict[str, Any]) -> SearchResult:
        """Enhanced search using expanded semantic context"""
        results = SearchResult(
            activity_info=None,
            people_info=[],
            related_memories=[]
        )
        
        # Get expanded keywords and relationships
        keywords = strategy['keywords']
        relationships = strategy.get('semantic_relationships', {})
        
        # Search current activity with expanded terms
        if strategy['focus'] in ['routine', 'both']:
            if context.current_activity:
                results.activity_info = self._search_current_activity(
                    context.current_activity,
                    keywords,
                    relationships
                )
                
                # Get information about involved people
                for person in context.current_activity.get('participants', []):
                    if self._is_person_relevant(person['name'], keywords, relationships):
                        person_activities = self._get_person_activities(
                            person['name'],
                            context.current_time
                        )
                        if person_activities:
                            results.people_info.append(person_activities)
        
        # Search memories with expanded terms
        if strategy['focus'] in ['memory', 'both']:
            # Combine all relevant search terms
            search_terms = set()
            for category in keywords.values():
                search_terms.update(category)
            
            # Add related terms from relationships
            for term, rels in relationships.items():
                for rel in rels:
                    search_terms.add(rel['related_to'])
            
            memories = self.memory_graph.search_memories(list(search_terms))
            if memories:
                results.related_memories = self._rank_memories(
                    memories,
                    keywords,
                    relationships
                )
        
        return results

    def _is_person_relevant(self, person_name: str,
                          keywords: Dict[str, List[str]],
                          relationships: Dict[str, List[Dict[str, str]]]) -> bool:
        """Check if a person is relevant based on keywords and relationships"""
        # Direct keyword match
        if person_name in keywords.get('people', []):
            return True
            
        # Relationship match
        for term, rels in relationships.items():
            for rel in rels:
                if rel['related_to'] == person_name:
                    return True
        
        return False

    def _search_current_activity(self, 
                               activity: Dict[str, Any],
                               keywords: Dict[str, List[str]],
                               relationships: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
        """Enhanced activity search with semantic context"""
        activity_info = {
            'name': activity['activity'],
            'time': f"{activity['time_start']} - {activity['time_end']}",
            'location': activity['location'],
            'participants': []
        }
        
        # Process participants with semantic relationships
        for participant in activity.get('participants', []):
            if self._is_person_relevant(participant['name'], keywords, relationships):
                activity_info['participants'].append(participant)
        
        return activity_info

    def _get_person_activities(self, 
                             person: str, 
                             current_time: datetime) -> Optional[Dict[str, Any]]:
        """Get activities related to a person"""
        activities = []
        
        for activity in self.memory_graph.routine_data['activities']:
            participants = [p['name'] for p in activity.get('participants', [])]
            if person in participants:
                activities.append({
                    'name': activity['activity'],
                    'time': f"{activity['time_start']} - {activity['time_end']}",
                    'location': activity['location']
                })
        
        if activities:
            return {
                'person': person,
                'activities': activities
            }
        return None

    def _rank_memories(self,
                      memories: List[Dict[str, Any]],
                      keywords: Dict[str, List[str]],
                      relationships: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        """Rank memories based on semantic relevance"""
        ranked_memories = []
        
        for memory in memories:
            relevance_score = 0
            memory_text = f"{memory['description']} {memory.get('details', '')}".lower()
            
            # Score direct keyword matches
            for category in keywords.values():
                for term in category:
                    if term.lower() in memory_text:
                        relevance_score += 1
            
            # Score semantic relationship matches
            for term, rels in relationships.items():
                for rel in rels:
                    if rel['related_to'].lower() in memory_text:
                        relevance_score += 0.5  # Lower weight for related terms
            
            ranked_memories.append({
                **memory,
                'relevance_score': relevance_score
            })
        
        # Sort by relevance score
        ranked_memories.sort(key=lambda x: x['relevance_score'], reverse=True)
        return ranked_memories



    def _evaluate_results(self,
                         context: DialogueContext,
                         strategy: Dict[str, Any],
                         results: SearchResult) -> Dict[str, Any]:
        """Evaluate search results"""
        prompt = f"""Evaluate these search results:
        Dialogue: "{context.text}"
        Strategy: {json.dumps(strategy)}
        Results: {json.dumps(results.__dict__)}
        
        Return a JSON with:
        {{
            "score": float (0-1),
            "coverage": ["covered_aspects"],
            "gaps": ["missing_aspects"],
            "focus_recommendation": "routine" or "memory" or "balanced"
        }}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Evaluate search results"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error in results evaluation: {e}")
            return {
                "score": 0.5,
                "coverage": [],
                "gaps": [],
                "focus_recommendation": "balanced"
            }

    def _format_response(self,
                        context: DialogueContext,
                        strategy: Dict[str, Any],
                        results: SearchResult,
                        effectiveness: Dict[str, Any]) -> Dict[str, Any]:
        """Format final response"""
        return {
            "query_analysis": {
                "dialogue": context.text,
                "time": context.current_time.strftime("%H:%M"),
                "strategy": strategy
            },
            "search_results": {
                "current_activity": results.activity_info,
                "people_info": results.people_info,
                "memories": results.related_memories
            },
            "search_performance": {
                "effectiveness_score": effectiveness['score'],
                "coverage": effectiveness['coverage'],
                "gaps": effectiveness['gaps'],
                "weights": {
                    "routine": self.weights.routine_weight,
                    "memory": self.weights.story_weight
                }
            }
        }

def main():
    # Test the optimized agent
    agent = OptimizedMemoryAgent(
        patient_id="P001",
        api_key="your-api-key"
    )
    
    current_time = datetime.strptime("06:15", "%H:%M")
    dialogue = "I enjoy our morning exercises, but I can't remember what time they start today. Could you help me find out when it begins??"
    
    results = agent.process_query(dialogue, current_time)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()