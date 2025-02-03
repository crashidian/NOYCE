import openai
import json
from typing import List, Dict, Optional, Tuple
import random
import time
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import logging

class IntegratedPatientGenerator:
    """Generate both personal stories and daily routines for dementia patients."""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('generation_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create directory structure
        self.base_dir = pathlib.Path("Patient_Data")
        self.stories_dir = self.base_dir / "Life_Stories"
        self.routines_dir = self.base_dir / "Daily_Routines"
        self.graphs_dir = self.base_dir / "Routine_Graphs"
        self.profiles_dir = self.base_dir / 'Profiles'
        
        for directory in [self.stories_dir, self.routines_dir, self.graphs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        

        self.dialogues_dir = self.base_dir / "Ground_Truth"
        for directory in [self.stories_dir, self.routines_dir, self.graphs_dir, self.dialogues_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize reference data
        self.career_categories = {
            "Education": ["Teacher", "Professor", "Principal"],
            "Healthcare": ["Doctor", "Nurse", "Pharmacist"],
            "Business": ["Business Owner", "Manager", "Accountant"],
            "Government": ["Civil Servant", "Police Officer", "Government Official"],
            "Technical": ["Engineer", "Architect", "Technician"],
            "Service": ["Shop Owner", "Restaurant Owner", "Salesperson"]
        }
        
        self.hobbies = [
            "Gardening", "Reading", "Walking", "Cooking", 
            "Chess", "Fishing", "Photography", "Painting",
            "Music", "Travel", "Writing", "Golf"
        ]
        
        # Visual styling for graphs
        self.node_colors = {
            'person': '#66c2a5',
            'event': '#fc8d62'
        }

    def generate_patient_profile(self, patient_id: int) -> Dict:
        """Generate basic profile for a patient."""
        current_year = 2024
        age = random.randint(40, 80)
        birth_year = current_year - age
        
        # Generate random gender and appropriate name
        gender = random.choice(["male", "female"])
        if gender == "male":
            first_name = random.choice([
                # Traditional names
                "James", "John", "Robert", "William", "David", "Richard", "Thomas",
                "Charles", "Michael", "Joseph", "George", "Edward", "Henry", "Frank",
                "Walter", "Harold", "Albert", "Arthur", "Fred", "Donald", "Paul",
                # Additional common names
                "Daniel", "Peter", "Mark", "Anthony", "Steven", "Andrew", "Kenneth",
                "Stephen", "Christopher", "Kevin", "Brian", "Ronald", "Timothy", "Gary",
                # Classic names
                "Raymond", "Lawrence", "Ralph", "Howard", "Eugene", "Philip", "Carl",
                "Roger", "Bruce", "Gerald", "Samuel", "Benjamin", "Theodore", "Russell",
                # More variations
                "Jack", "Louis", "Roy", "Dennis", "Stanley", "Leonard", "Vincent",
                "Gordon", "Norman", "Bernard", "Frederick", "Martin", "Douglas", "Leo"
            ])
        else:
            first_name = random.choice([
                # Traditional names
                "Mary", "Elizabeth", "Patricia", "Barbara", "Margaret", "Susan",
                "Dorothy", "Helen", "Betty", "Ruth", "Sarah", "Alice", "Florence",
                "Marie", "Anna", "Grace", "Rose", "Louise", "Martha", "Emma",
                # Additional common names
                "Catherine", "Frances", "Shirley", "Virginia", "Carol", "Judith",
                "Nancy", "Joyce", "Janet", "Linda", "Carolyn", "Sandra", "Sharon",
                "Diane", "Joan", "Jane", "Karen", "Gloria", "Evelyn", "Jean",
                # Classic names
                "Irene", "Marilyn", "Phyllis", "Eleanor", "Doris", "Beverly",
                "Charlotte", "Agnes", "Mildred", "Katherine", "Edith", "Rita",
                # More variations
                "Josephine", "Theresa", "Clara", "Gladys", "Ethel", "Esther",
                "Ellen", "Beatrice", "Audrey", "Anne", "Lillian", "Julia", "Sylvia"
            ])
        
        career_category = random.choice(list(self.career_categories.keys()))
        career = random.choice(self.career_categories[career_category])
        patient_hobbies = random.sample(self.hobbies, random.randint(2, 3))
        
        return {
            "patient_id": f"P{patient_id:03d}",
            "name": first_name,
            "gender": gender,
            "age": age,
            "birth_year": birth_year,
            "career": career,
            "career_category": career_category,
            "education_level": random.choice(["High School", "College", "Bachelor", "Master"]),
            "hobbies": patient_hobbies,
            "marital_status": random.choice(["Married", "Widowed", "Divorced"]),
            "children_count": random.randint(0, 3)
        }

    def generate_life_story(self, profile: Dict) -> Optional[Dict]:
        """Generate life story based on patient profile."""
        system_prompt = """You are a professional oral history recorder. Generate interview records in JSON format with three interviews for each patient.
        Each interview must follow this structure:
        {
            "interviews": [
                {
                    "interviewee": {"name": "", "relationship": "", "age": 0, "background": ""},
                    "memories": [
                        {
                            "year": 0,
                            "title": "", # A short 1-3 word summary (e.g., "Wedding Day", "University Graduation", "Career Promotion")
                            "description": "", # Original longer description
                            "details": "",
                            "impact": ""
                        }
                    ],
                    "relationship_story": "",
                    "observations": "",
                    "recent_changes": ""
                }
            ]
        }"""
        
        user_prompt = f"""Generate three interviews for a patient with the following profile:
        - Age: {profile['age']} years old (born in {profile['birth_year']})
        - Career: {profile['career']}
        - Education: {profile['education_level']}
        - Hobbies: {', '.join(profile['hobbies'])}
        - Marital Status: {profile['marital_status']}
        - Number of Children: {profile['children_count']}

        Include:
        1. At least two close family members (spouse or child)
        2. At least two friends or colleagues
        3. Another family member (sibling or relative)

        Each interview must have exactly 5 memories that span across different periods of the patient's life:
        - Early life/childhood (before 18)
        - Young adult years (18-30)
        - Middle age years (30-50)
        - Later adult years (50+)
        - Recent memories (within last 5 years)
        
        For each memory:
        1. Add a short title (1-3 words) that clearly identifies the event (e.g., "Wedding Day", "House Purchase", "Career Award")
        2. Make sure the title is distinct and specific to that memory
        3. The title should be easily readable in a network visualization
        4. Provide full details in the description field
        
        Make sure the memories are significant life events that showcase the patient's personality, relationships, career progression, and important life changes.
        Return only the JSON data."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            story_data = json.loads(response.choices[0].message.content)
            return{
                "profile": profile,
                "interviews": story_data.get("interviews", [])
            }


        
        except Exception as e:
            self.logger.error(f"Error generating life story: {str(e)}")
            return None

    def generate_daily_routine(self, profile: Dict) -> Optional[Dict]:
        """Generate daily routine based on patient profile."""
        system_prompt = f"""Create a detailed daily schedule for a {profile['age']}-year-old dementia patient 
        with background as a {profile['career']} and interests in {', '.join(profile['hobbies'])}.
        
        The schedule should:
        1. Cover activities from 6:00 AM to 22:00 PM
        2. Include interactions with family, medical staff, and other patients
        3. Include medical routines, meals, and activities related to their background
        4. Include activities that match their interests and past career
        
        Return the schedule as a JSON object with an 'activities' array containing:
        {{
            "time_start": "HH:MM",
            "time_end": "HH:MM",
            "activity": "",
            "location": "",
            "participants": [
                {{"role": "", "name": ""}}
            ],
            "details": ""
        }}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Create a personalized daily routine for patient {profile['patient_id']}"}
                ],
                temperature=0.8,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Error generating daily routine: {str(e)}")
            return None



    # def generate_ground_truth_dialogues(self, profile: Dict, story_data: Dict, routine_data: Dict) -> Optional[Dict]:
    #     """Generate dialogues with a realistic mix of clear and confused communication."""
        
    #     def get_current_activity(time_str: str, activities: List[Dict]) -> Optional[Dict]:
    #         """Find the activity occurring at a given time."""
    #         hour = int(time_str.split(':')[0])
    #         minute = int(time_str.split(':')[1])
    #         current_time = hour * 60 + minute
            
    #         for activity in activities:
    #             start_hour, start_minute = map(int, activity['time_start'].split(':'))
    #             end_hour, end_minute = map(int, activity['time_end'].split(':'))
                
    #             start_time = start_hour * 60 + start_minute
    #             end_time = end_hour * 60 + end_minute
                
    #             if start_time <= current_time <= end_time:
    #                 return activity
    #         return None

    #     system_prompt = """Generate 10 dialogues simulating conversations with a dementia patient. Each dialogue should:

    #     1. TIMING AND CONTEXT:
    #     - Include a specific time (06:00-22:00)
    #     - Reference the current or upcoming activities
    #     - Consider the location and present people
        
    #     2. DIALOGUE DISTRIBUTION:
    #     - 80% Clear Communication:
    #         * Ask about current activities
    #         * Discuss daily schedule
    #         * Share memories correctly
    #         * Express current needs/wants
    #         * Ask about family/friends
    #         * Comment on surroundings
    #         * Request assistance
    #         * Make observations
        
    #     - 20% Confused Communication (choose from):
    #         * Confuse past/present events
    #         * Mix up scheduled activities
    #         * Reference non-existent appointments
    #         * Show time displacement
    #         * Mix up family members
    #         * Show location uncertainty
    #         * Express age/life stage confusion
        
    #     3. EMOTIONAL ELEMENTS (throughout all dialogues):
    #     - Show anxiety in unclear situations
    #     - Express occasional frustration
    #     - Display comfort with familiar topics
    #     - Show concern in unfamiliar settings
    #     - Express joy when discussing happy memories
    #     - Show appreciation for help
        
    #     Return JSON format:
    #     {
    #         "dialogues": [
    #             {
    #                 "time": "HH:MM",
    #                 "context": {
    #                     "current_activity": {
    #                         "name": "",
    #                         "location": "",
    #                         "participants": []
    #                     }
    #                 },
    #                 "dialogue_type": "clear/confused",
    #                 "query": {
    #                     "text": "Patient's question or statement",
    #                     "emotional_state": "",
    #                     "confusion_pattern": null  # Only for confused dialogues
    #                 },
    #                 "expected_response": {
    #                     "content": "appropriate response",
    #                     "referenced_data": {
    #                         "actual_facts": ["relevant information"],
    #                         "patient_confusion": []  # Only for confused dialogues
    #                     },
    #                     "emotional_support": "validation and comfort approach"
    #                 }
    #             }
    #         ]
    #     }"""

    #     user_prompt = f"""Create authentic patient dialogues using this patient's information:
        
    #     PATIENT PROFILE:
    #     {json.dumps(profile, indent=2)}
        
    #     DAILY SCHEDULE:
    #     {json.dumps(routine_data['activities'], indent=2)}
        
    #     LIFE STORY INTERVIEWS:
    #     {json.dumps(story_data['interviews'], indent=2)}
        
    #     Instructions:
    #     1. Generate 8 clear dialogues showing:
    #     - Normal questions about activities
    #     - Correct memory sharing
    #     - Regular conversation
    #     - Appropriate emotional expression
        
    #     2. Generate 2 confused dialogues with:
    #     - Specific confusion patterns
    #     - Clear contrast between confusion and reality
    #     - Appropriate emotional response
        
    #     3. For all dialogues:
    #     - Use real names from their life story
    #     - Reference actual scheduled activities
    #     - Include emotional elements
    #     - Make responses supportive and appropriate
    #     """

    #     try:
    #         response = self.client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": system_prompt},
    #                 {"role": "user", "content": user_prompt}
    #             ],
    #             response_format={"type": "json_object"},
    #             max_tokens=8000
    #         )
            
    #         dialogues_data = json.loads(response.choices[0].message.content)
            
    #         # Enrich dialogues with current activity context
    #         for dialogue in dialogues_data["dialogues"]:
    #             time_str = dialogue["time"]
    #             current_activity = get_current_activity(time_str, routine_data['activities'])
                
    #             if current_activity:
    #                 dialogue["context"]["current_activity"] = {
    #                     "name": current_activity["activity"],
    #                     "location": current_activity["location"],
    #                     "participants": current_activity["participants"]
    #                 }
    #             else:
    #                 dialogue["context"]["current_activity"] = {
    #                     "name": "Free Time",
    #                     "location": "Unknown",
    #                     "participants": []
    #                 }
            
    #         return {
    #             "patient_id": profile["patient_id"],
    #             "dialogues": dialogues_data["dialogues"]
    #         }
            
    #     except Exception as e:
    #         self.logger.error(f"Error in dialogue generation: {str(e)}")
    #         return None


    def generate_ground_truth_dialogues(self, profile: Dict, story_data: Dict, routine_data: Dict) -> Optional[Dict]:
        """Generate dialogues with balanced lengths between queries and responses."""
        
        def get_current_activity(time_str: str, activities: List[Dict]) -> Optional[Dict]:
            """Find the activity occurring at a given time."""
            hour = int(time_str.split(':')[0])
            minute = int(time_str.split(':')[1])
            current_time = hour * 60 + minute
            
            for activity in activities:
                start_hour, start_minute = map(int, activity['time_start'].split(':'))
                end_hour, end_minute = map(int, activity['time_end'].split(':'))
                
                start_time = start_hour * 60 + start_minute
                end_time = end_hour * 60 + end_minute
                
                if start_time <= current_time <= end_time:
                    return activity
            return None

        system_prompt = """Generate 10 dialogues between a dementia patient and a CAREGIVER. Each dialogue should show:

        PATIENT QUERIES (simulate detailed patient's questions/statements):
        1. Clear Communications (80%):
        - Questions about activities with context: 
        "I enjoy our morning exercises, but I can't remember what time they start today.  Could you help me find out when it begins?"
        
        - Sharing memories with details:
        "When I was teaching mathematics at the high school, we had this wonderful end-of-year tradition where students would solve puzzles in teams. Do you think I could help with any math activities here?"
        
        2. Confused Communications (20%):
        - Confuse past and present events
        - Misremember scheduled activities
        - Ask about non-existent appointments
        - Reference incorrect times/dates
        - Show uncertainty about location/context
        - Repeat questions with slight variations
        - Show time displacement (e.g., thinking it's a different year)
        - Confusion about current age/life stage

        Guidelines:
        1. Patient queries must:
        - Be 1-2 sentences long


        
        CAREGIVER RESPONSES (<=1000 tokens) must:
        - two times long of query
        - Gently correct misconceptions using real data
        - Reference actual schedule/people
        - Provide reassurance with facts
        - Use positive memories as anchors
        - Validate emotions while redirecting
        - Include specific comfort strategies

        Example questions:
        - "Is Mary coming today? She usually brings cookies..." 
        - "Don't we have choir practice this afternoon?" 
        - "Did Tom say he's visiting after lunch?" 
        - "I think I'm supposed to meet Emily after cooking lecture. 

        Return JSON format:
        {
            "dialogues": [
                {
                    "time": "HH:MM",
                    "context": {
                        "current_activity": {
                            "name": "",
                            "location": "",
                            "participants": []
                        }
                    },
                    "dialogue_type": "clear/confused",
                    "query": {
                        "text": "Detailed patient question/statement (1-2 sentences)",
                        "confusion_pattern": null  # Only for confused communications
                    },
                    "expected_response": {
                        "content": "Balanced response (2x query length)"
                    }
                }
            ]
        }"""

        user_prompt = f"""Create authentic dialogues using this patient's background:
        
        PATIENT PROFILE:
        {json.dumps(profile, indent=2)}
        
        DAILY SCHEDULE:
        {json.dumps(routine_data['activities'], indent=2)}
        
        LIFE STORY:
        {json.dumps(story_data['interviews'], indent=2)}
        

        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=2000
            )
            
            dialogues_data = json.loads(response.choices[0].message.content)
            
            # Enrich dialogues with current activity context
            for dialogue in dialogues_data["dialogues"]:
                time_str = dialogue["time"]
                current_activity = get_current_activity(time_str, routine_data['activities'])
                
                if current_activity:
                    dialogue["context"]["current_activity"] = {
                        "name": current_activity["activity"],
                        "location": current_activity["location"],
                        "participants": current_activity["participants"]
                    }
                else:
                    dialogue["context"]["current_activity"] = {
                        "name": "Free Time",
                        "location": "Unknown",
                        "participants": []
                    }
            
            return {
                "patient_id": profile["patient_id"],
                "dialogues": dialogues_data["dialogues"]
            }
            
        except Exception as e:
            self.logger.error(f"Error in dialogue generation: {str(e)}")
            return None


    # def generate_ground_truth_dialogues(self, profile: Dict, story_data: Dict, routine_data: Dict) -> Optional[Dict]:
    #     """Generate dialogues with concise but empathetic AI responses."""
        
    #     def get_current_activity(time_str: str, activities: List[Dict]) -> Optional[Dict]:
    #         """Find the activity occurring at a given time."""
    #         hour = int(time_str.split(':')[0])
    #         minute = int(time_str.split(':')[1])
    #         current_time = hour * 60 + minute
            
    #         for activity in activities:
    #             start_hour, start_minute = map(int, activity['time_start'].split(':'))
    #             end_hour, end_minute = map(int, activity['time_end'].split(':'))
                
    #             start_time = start_hour * 60 + start_minute
    #             end_time = end_hour * 60 + end_minute
                
    #             if start_time <= current_time <= end_time:
    #                 return activity
    #         return None

    #     system_prompt = """Generate 10 dialogues between a dementia patient and an AI assistant. Each dialogue should show:

    #     PATIENT QUERIES (simulate dementia patient's questions/statements):
    #     1. Clear Communications (80%):
    #     - Questions about daily activities
    #     - Requests for assistance
    #     - Sharing memories
    #     - Basic needs and wants
        
    #     2. Confused Communications (20%):
    #     - Time confusion
    #     - Identity mixing
    #     - Activity displacement
    #     - Location uncertainty

    #     EMOTIONAL ELEMENTS IN QUERIES:
    #     - Anxiety
    #     - Confusion
    #     - Joy
    #     - Frustration
    #     - Contentment

    #     AI ASSISTANT RESPONSES SHOULD:
    #     1. Be concise but caring
    #     2. Include one clear validation
    #     3. Provide one practical solution/guidance
    #     4. Reference personal details when relevant
        
    #     Example Response Format:
    #     "[Validation] + [Practical Information/Guidance]"
    #     e.g., "I understand you're looking forward to seeing Mary. Sarah will be visiting today at 2 PM, and she always brings your favorite cookies."

    #     Return JSON format:
    #     {
    #         "dialogues": [
    #             {
    #                 "time": "HH:MM",
    #                 "context": {
    #                     "current_activity": {
    #                         "name": "",
    #                         "location": "",
    #                         "participants": []
    #                     }
    #                 },
    #                 "dialogue_type": "clear/confused",
    #                 "query": {
    #                     "text": "Patient's question or statement",
    #                     "emotional_state": "anxiety/confusion/joy/frustration/content"
    #                 },
    #                 "expected_response": {
    #                     "content": "Concise, empathetic response with one validation and one guidance point"
    #                 }
    #             }
    #         ]
    #     }"""

    #     user_prompt = f"""Create authentic dialogues using this patient's background:
        
    #     PATIENT PROFILE:
    #     {json.dumps(profile, indent=2)}
        
    #     DAILY SCHEDULE:
    #     {json.dumps(routine_data['activities'], indent=2)}
        
    #     LIFE STORY:
    #     {json.dumps(story_data['interviews'], indent=2)}
        
    #     Guidelines:
    #     1. Patient queries should be:
    #     - Natural and authentic
    #     - Based on their background
    #     - Reference real people and events
        
    #     2. AI responses should be:
    #     - One or two sentences only
    #     - Include emotional validation
    #     - Provide clear guidance
    #     - Reference relevant personal details
    #     """

    #     try:
    #         response = self.client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": system_prompt},
    #                 {"role": "user", "content": user_prompt}
    #             ],
    #             response_format={"type": "json_object"},
    #             max_tokens=8000
    #         )
            
    #         dialogues_data = json.loads(response.choices[0].message.content)
            
    #         # Enrich dialogues with current activity context
    #         for dialogue in dialogues_data["dialogues"]:
    #             time_str = dialogue["time"]
    #             current_activity = get_current_activity(time_str, routine_data['activities'])
                
    #             if current_activity:
    #                 dialogue["context"]["current_activity"] = {
    #                     "name": current_activity["activity"],
    #                     "location": current_activity["location"],
    #                     "participants": current_activity["participants"]
    #                 }
    #             else:
    #                 dialogue["context"]["current_activity"] = {
    #                     "name": "Free Time",
    #                     "location": "Unknown",
    #                     "participants": []
    #                 }
            
    #         return {
    #             "patient_id": profile["patient_id"],
    #             "dialogues": dialogues_data["dialogues"]
    #         }
            
    #     except Exception as e:
    #         self.logger.error(f"Error in dialogue generation: {str(e)}")
    #         return None


    def create_routine_graph(self, routine_data: Dict, profile: Dict) -> nx.Graph:
        """Create knowledge graph from daily routine data."""
        G = nx.Graph()
        
        # Add patient node
        G.add_node(profile['patient_id'],
                  node_type='person',
                  role='patient',
                  display_name=f"Patient {profile['patient_id']}")
        
        added_people = set()
        
        for idx, activity in enumerate(routine_data['activities']):
            event_id = f"E{idx:02d}_{activity['time_start']}_{activity['time_end']}"
            display_name = f"{activity['time_start']}-{activity['time_end']}\n{activity['activity']}"
            
            G.add_node(event_id,
                      node_type='event',
                      time_start=activity['time_start'],
                      time_end=activity['time_end'],
                      activity=activity['activity'],
                      location=activity['location'],
                      details=activity['details'],
                      display_name=display_name)
            
            G.add_edge(profile['patient_id'], event_id, relation='participates_in')
            
            for participant in activity['participants']:
                person_id = f"{participant['role']}_{participant['name']}"
                
                if person_id not in added_people:
                    G.add_node(person_id,
                             node_type='person',
                             role=participant['role'],
                             name=participant['name'],
                             display_name=f"{participant['name']}\n({participant['role']})")
                    added_people.add(person_id)
                
                G.add_edge(person_id, event_id, relation='involved_in')
        
        return G

    def save_patient_data(self, patient_id: str, story_data: Dict, routine_data: Dict, 
                        routine_graph: nx.Graph, dialogues: List[Dict]) -> bool:
        try:
            # Save patient profile
            profile_file = self.profiles_dir / f"{patient_id}_profile.json"
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(story_data['profile'], f, ensure_ascii=False, indent=2)

            # Save life story data
            story_file = self.stories_dir / f"{patient_id}_story.json"
            with open(story_file, 'w', encoding='utf-8') as f:
                # Remove profile from story data to avoid duplication
                story_data_without_profile = {
                    'interviews': story_data['interviews']
                }
                json.dump(story_data_without_profile, f, ensure_ascii=False, indent=2)

            # Save routine data
            routine_file = self.routines_dir / f"{patient_id}_routine.json"
            with open(routine_file, 'w', encoding='utf-8') as f:
                json.dump(routine_data, f, ensure_ascii=False, indent=2)

            # Save dialogues
            dialogue_file = self.dialogues_dir / f"{patient_id}_dialogues.json"
            with open(dialogue_file, 'w', encoding='utf-8') as f:
                json.dump({"dialogues": dialogues}, f, ensure_ascii=False, indent=2)

            # Create and save graph data
            graph_dir = self.graphs_dir / patient_id
            graph_dir.mkdir(exist_ok=True)

            # Save nodes and edges
            pd.DataFrame([
                {'node_id': node, **attrs}
                for node, attrs in routine_graph.nodes(data=True)
            ]).to_csv(graph_dir / 'nodes.csv', index=False)

            pd.DataFrame([
                {'source': u, 'target': v, 'relation': attrs.get('relation', '')}
                for u, v, attrs in routine_graph.edges(data=True)
            ]).to_csv(graph_dir / 'edges.csv', index=False)

            return True

        except Exception as e:
            self.logger.error(f"Error saving data for {patient_id}: {str(e)}")
            return False

    def generate_patient_data(self, patient_id: int, max_retries: int = 3) -> Tuple[bool, Optional[Dict]]:
        """Generate patient data without routine graph."""
        profile = self.generate_patient_profile(patient_id)
        patient_id_str = profile['patient_id']
        
        self.logger.info(f"Generating data for patient {patient_id_str}")
        
        for attempt in range(max_retries):
            try:
                story_data = self.generate_life_story(profile)
                if not story_data:
                    raise Exception("Failed to generate life story")
                
                # Add profile to story_data for saving
                story_data['profile'] = profile
                
                routine_data = self.generate_daily_routine(profile)
                if not routine_data:
                    raise Exception("Failed to generate daily routine")
                
                dialogues = self.generate_ground_truth_dialogues(profile, story_data, routine_data)
                if not dialogues:
                    raise Exception("Failed to generate dialogues")
                
                # Save the data
                try:
                    # Save patient profile
                    profile_file = self.profiles_dir / f"{patient_id_str}_profile.json"
                    with open(profile_file, 'w', encoding='utf-8') as f:
                        json.dump(story_data['profile'], f, ensure_ascii=False, indent=2)

                    # Save life story data
                    story_file = self.stories_dir / f"{patient_id_str}_story.json"
                    with open(story_file, 'w', encoding='utf-8') as f:
                        # Remove profile from story data to avoid duplication
                        story_data_without_profile = {
                            'interviews': story_data['interviews']
                        }
                        json.dump(story_data_without_profile, f, ensure_ascii=False, indent=2)

                    # Save routine data
                    routine_file = self.routines_dir / f"{patient_id_str}_routine.json"
                    with open(routine_file, 'w', encoding='utf-8') as f:
                        json.dump(routine_data, f, ensure_ascii=False, indent=2)

                    # Save dialogues
                    dialogue_file = self.dialogues_dir / f"{patient_id_str}_dialogues.json"
                    with open(dialogue_file, 'w', encoding='utf-8') as f:
                        json.dump({"dialogues": dialogues['dialogues']}, f, ensure_ascii=False, indent=2)

                    self.logger.info(f"Successfully generated data for {patient_id_str}")
                    return True, {
                        "patient_id": patient_id_str,
                        "profile": profile,
                        "has_story": True,
                        "has_routine": True,
                        "has_dialogues": True
                    }

                except Exception as e:
                    self.logger.error(f"Error saving data for {patient_id_str}: {str(e)}")
                    if attempt < max_retries - 1:
                        continue
                    return False, None

            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Attempt {attempt + 1} failed for {patient_id_str}, retrying...")
                    time.sleep(2)
                else:
                    self.logger.error(f"Failed all attempts for {patient_id_str}: {str(e)}")
            
        return False, None



    def generate_multiple_patients(self, count: int = 100) -> List[Dict]:
        """Generate complete datasets for multiple patients."""
        successful_generations = []
        
        for i in range(count):
            success, patient_info = self.generate_patient_data(i + 1)
            if success:
                successful_generations.append(patient_info)
            
            # Create/update index every 10 patients
            if (i + 1) % 10 == 0:
                self.create_patient_index(successful_generations)
                self.logger.info(f"Updated index with {len(successful_generations)} patients")
        
        # Final index update
        self.create_patient_index(successful_generations)
        return successful_generations

    def create_patient_index(self, patients_info: List[Dict]):
        """Create index file with status of generated data."""
        try:
            index_file = self.base_dir / "patient_index.json"
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(patients_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error creating index: {str(e)}")


