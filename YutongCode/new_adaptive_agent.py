from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import networkx as nx
import pathlib
from openai import OpenAI

@dataclass
class SearchWeights:
    routine_weight: float = 0.5
    story_weight: float = 0.5
    min_weight: float = 0.2
    max_weight: float = 0.8

    def adjust(self, effectiveness: Dict[str, Any]):
        if effectiveness['focus_recommendation'] == 'routine':
            self.routine_weight = min(self.max_weight, self.routine_weight + 0.1)
        elif effectiveness['focus_recommendation'] == 'memory':
            self.story_weight = min(self.max_weight, self.story_weight + 0.1)
        
        total = self.routine_weight + self.story_weight
        self.routine_weight /= total
        self.story_weight /= total

@dataclass
class SearchPriorities:
    people: float = 0.4
    events: float = 0.3
    time: float = 0.2
    location: float = 0.1

@dataclass
class RelevanceThresholds:
    min_score: float = 0.5
    strong_match: float = 0.7
    keyword_overlap: float = 0.3

@dataclass
class DialogueContext:
    text: str
    current_time: datetime
    current_activity: Optional[Dict[str, Any]]

@dataclass
class SearchResult:
    nodes: List[Dict[str, Any]]
    effectiveness: Dict[str, Any]

class MemoryGraphManager:
    def __init__(self, patient_id: str, base_path: str = "Patient_Data"):
        self.base_path = pathlib.Path(base_path)
        self.patient_id = patient_id
        self.load_patient_data()
        self.memory_network = self.construct_memory_network()

    def load_patient_data(self):
        try:
            profile_path = self.base_path / "Profiles" / f"{self.patient_id}_profile.json"
            story_path = self.base_path / "Life_Stories" / f"{self.patient_id}_story.json"
            routine_path = self.base_path / "Daily_Routines" / f"{self.patient_id}_routine.json"
            
            with open(profile_path) as f:
                self.profile = json.load(f)
            with open(story_path) as f:
                self.story_data = json.load(f)
            with open(routine_path) as f:
                self.routine_data = json.load(f)
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def construct_memory_network(self) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()
        
        for activity in self.routine_data['activities']:
            activity_id = f"activity_{activity['activity']}"
            G.add_node(activity_id, type='activity', **activity)
            
            for person in activity.get('participants', []):
                person_id = f"person_{person['name']}"
                G.add_node(person_id, type='person', **person)
                G.add_edge(activity_id, person_id, type='involves')

        for memory in self.story_data.get('memories', []):
            memory_id = f"memory_{memory.get('id', hash(str(memory)))}"
            G.add_node(memory_id, type='memory', **memory)
            
            for person in memory.get('people', []):
                person_id = f"person_{person}"
                if person_id in G:
                    G.add_edge(memory_id, person_id, type='involves')
        
        return G

    def get_activity_at_time(self, time: datetime) -> Optional[Dict[str, Any]]:
        time_str = time.strftime("%H:%M")
        for activity in self.routine_data['activities']:
            if activity['time_start'] <= time_str <= activity['time_end']:
                return activity
        return None

class ReflectiveMemoryAgent:
    def __init__(self, patient_id: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.memory_graph = MemoryGraphManager(patient_id)
        self.weights = SearchWeights()
        self.priorities = SearchPriorities()
        self.thresholds = RelevanceThresholds()
        self.search_history = []


    def _analyze_dialogue(self, context: DialogueContext) -> Dict[str, Any]:
        prompt = f"""Analyze this dialogue and return ONLY a valid JSON object with no additional text:
        Text: "{context.text}"
        Time: {context.current_time.strftime("%H:%M")}
        Current Activity: {context.current_activity['activity'] if context.current_activity else 'None'}

        The JSON must have exactly this structure:
        {{
            "keywords": {{
                "people": [],
                "events": [],
                "time": [],
                "location": []
            }},
            "focus": "routine",
            "temporal_context": "current"
        }}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a JSON generator. Only output valid JSON without any other text."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            analysis = json.loads(response.choices[0].message.content)
            expanded = self._expand_keywords(analysis['keywords'])
            return {
                "keywords": expanded,
                "focus": analysis['focus'],
                "temporal_context": analysis['temporal_context']
            }
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Raw response: {response.choices[0].message.content}")
            return {
                "keywords": {
                    "people": [],
                    "events": [],
                    "time": [],
                    "location": []
                },
                "focus": "both",
                "temporal_context": "both"
            }

    def _expand_keywords(self, keywords: Dict[str, List[str]]) -> Dict[str, List[str]]:
        prompt = f"""Expand these keywords and return ONLY a valid JSON object with no additional text:
        Keywords: {json.dumps(keywords)}
        Profile: {json.dumps(self.memory_graph.profile)}

        Return a JSON object with exactly this structure:
        {{
            "people": [],
            "events": [],
            "time": [],
            "location": []
        }}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a JSON generator. Only output valid JSON without any other text."},
                    {"role": "user", "content": prompt}
                ]
            )
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Raw response: {response.choices[0].message.content}")
            return keywords


    # def _calculate_relevance(self, content: Dict[str, Any], 
    #                        keywords: Dict[str, List[str]]) -> float:
    #     score = 0.0
    #     text = json.dumps(content).lower()
        
    #     for category, terms in keywords.items():
    #         weight = getattr(self.priorities, category)
    #         matches = sum(1 for term in terms if term.lower() in text)
    #         score += matches * weight
            
    #     return score

    def _calculate_relevance(self, content: Dict[str, Any], keywords: Dict[str, List[str]]) -> float:
        score = 0.0
        text = json.dumps(content).lower()

        for category, terms in keywords.items():
            weight = getattr(self.priorities, category)
            matches = 0
            for term in terms:
                # Ensure term is a string
                if isinstance(term, dict):
                    # Extract the actual term if term is a dictionary
                    term_str = term.get('term') or term.get('name') or ''
                elif isinstance(term, str):
                    term_str = term
                else:
                    term_str = ''
                if term_str.lower() in text:
                    matches += 1
            score += matches * weight

        return score


    def _evaluate_effectiveness(self, context: DialogueContext, 
                              results: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = f"""Evaluate search results:
        Query: "{context.text}"
        Results: {json.dumps(results)}
        
        Return JSON:
        {{
            "score": float (0-1),
            "sufficient": boolean,
            "missing_aspects": [],
            "focus_recommendation": "routine/memory/balanced",
            "reasoning": "explanation"
        }}"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def _search_knowledge_graphs(self, keywords: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        results = []
        
        # Apply weights to different graph types
        for node in self.memory_graph.memory_network.nodes(data=True):
            relevance = self._calculate_relevance(node[1], keywords)
            source_type = 'routine' if node[1]['type'] == 'activity' else 'story'
            weight = (self.weights.routine_weight if source_type == 'routine' 
                     else self.weights.story_weight)
            
            weighted_relevance = relevance * weight
            
            if weighted_relevance >= self.thresholds.min_score:
                results.append({
                    "type": node[1]['type'],
                    "content": node[1],
                    "relevance": weighted_relevance,
                    "source": source_type
                })
        
        return sorted(results, key=lambda x: x['relevance'], reverse=True)[:4]

    def process_query(self, text: str, current_time: datetime) -> Dict[str, Any]:
        # Initialize with current activity
        current_activity = self.memory_graph.get_activity_at_time(current_time)
        context = DialogueContext(text, current_time, current_activity)
        
        # Analyze dialogue for search strategy
        strategy = self._analyze_dialogue(context)
        
        results = []
        # First check current activity relevance
        if current_activity:
            relevance = self._calculate_relevance(current_activity, strategy['keywords'])
            if relevance >= self.thresholds.min_score:
                results.append({
                    "type": "activity",
                    "content": current_activity,
                    "relevance": relevance,
                    "source": "routine"
                })
        
        # Evaluate current activity results
        initial_effectiveness = self._evaluate_effectiveness(context, results)
        
        # Continue search if needed
        if not initial_effectiveness['sufficient']:
            graph_results = self._search_knowledge_graphs(strategy['keywords'])
            results.extend(graph_results)
            initial_effectiveness = self._evaluate_effectiveness(context, results)
        
        # Final reflection and weight adjustment
        self.weights.adjust(initial_effectiveness)
        
        self.search_history.append({
            "query": text,
            "effectiveness": initial_effectiveness['score'],
            "weights": {
                "routine": self.weights.routine_weight,
                "story": self.weights.story_weight
            }
        })
        
        return {
            "query_time": current_time.strftime("%H:%M"),
            "current_activity": current_activity,
            "keyword_analysis": {
                "original": strategy['keywords'],
                "expanded": strategy['keywords']
            },
            "results": results[:8],
            "search_performance": {
                "effectiveness": initial_effectiveness,
                "weights": {
                    "routine": self.weights.routine_weight,
                    "story": self.weights.story_weight
                }
            }
        }



def main():
    agent = ReflectiveMemoryAgent(
        patient_id="P001",
        api_key="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"
    )
    


    query = "Will Emily come today?"
    current_time = datetime.strptime("14:30", "%H:%M")
    
    result = agent.process_query(query, current_time)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()