    def generate_ground_truth_dialogues(self, profile: Dict, story_data: Dict, routine_data: Dict) -> Optional[Dict]:
        system_prompt = """Generate 10 dialogues simulating a dementia patient. Questions should show:

        
        CONFUSION PATTERNS:
        - Confuse past and present events
        - Misremember scheduled activities
        - Ask about non-existent appointments
        - Reference incorrect times/dates
        - Show uncertainty about location/context
        - Repeat questions with slight variations
        - Mix up family members' identities
        - Show time displacement (e.g., thinking it's a different year)
        - Confusion about current age/life stage

        EMOTIONAL ELEMENTS:
        - Show anxiety about unclear situations
        - Express frustration with memory issues
        - Display comfort when discussing clear memories
        - Show concern about unfamiliar surroundings

        Example questions:
        - "Is Mary coming today? She usually brings cookies..." (when Sarah is the actual visitor)
        - "Don't we have choir practice this afternoon?" (mixing up days/activities)
        - "Did Tom say he's visiting after lunch?" (when Tom visits in mornings)
        - "I think I'm supposed to meet someone..." (vague memory of scheduled activity)
        - "Where did I put my glasses? Someone must have moved them..."
        - "I need to pick up the children from school..." (when children are grown)
        - "Why hasn't my mother visited?" (when mother is deceased)

        CAREGIVER RESPONSES (<=500 tokens) must:
        - Gently correct misconceptions using real data
        - Reference actual schedule/people
        - Provide reassurance with facts
        - Use positive memories as anchors
        - Validate emotions while redirecting
        - Include specific comfort strategies

        Return JSON format:
        {
            "dialogues": [
                {
                    "type": "daily_routine/memory/random",
                    "query": "confused question (<=100 tokens)",
                    "expected_response": {
                        "content": "gentle correction with facts",
                        "referenced_data": {
                            "actual_facts": ["correct_information"],
                            "patient_confusion": ["misremembered_elements"]
                        }
                    }
                }
            ]
        }"""

        user_prompt = f"""Create authentic dementia patient dialogues using patient data:
        {json.dumps({'profile': profile, 'schedule': routine_data['activities'], 'memories': story_data['interviews']}, indent=2)}
        
        
        Remember:
        1. 80% of dialogues should show clear communication
        2. Only 20% should show confusion
        3. All responses should be detailed and empathetic
        4. Include personal details from the patient's history
        
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=8000
            )
            
            dialogues_data = json.loads(response.choices[0].message.content)
            
            
            return {
                "patient_id": profile["patient_id"],
                "dialogues": dialogues_data["dialogues"]
            }
        except Exception as e:
            self.logger.error(f"Error in dialogue generation: {str(e)}")
            return None



        system_prompt = """Generate 10 dialogues between a dementia patient and an AI assistant. Each dialogue should show:

        PATIENT QUERIES (simulate dementia patient's questions/statements):
        1. Clear Communications (80%):
        - Questions about current activities: "What time is lunch today?"
        - Requests for help: "Can you help me find my book?"
        - Sharing memories: "I remember when I used to teach mathematics..."
        - Daily needs: "I'd like to go for a walk in the garden."
        
        2. Confused Communications (20%):
        - Time confusion: "Is it time for breakfast?" (when it's afternoon)
        - Identity mixing: "When is my sister Mary coming?" (when Mary passed away)
        - Activity displacement: "I need to prepare lessons for tomorrow" (when retired)
        - Location uncertainty: "When are we going home?" (when in their home)

        EMOTIONAL ELEMENTS IN QUERIES:
        - Anxiety: "I'm not sure where I am..."
        - Confusion: "Everything feels strange today..."
        - Joy: "I remember my wedding day clearly!"
        - Frustration: "Why can't I remember?"

        AI ASSISTANT RESPONSES MUST:
        1. Validation and Empathy:
        - Acknowledge emotions: "I can see this is frustrating for you..."
        - Validate feelings: "It's perfectly natural to feel uncertain..."
        - Show understanding: "I understand why you might feel this way..."

        2. Gentle Reorientation:
        - Use facts gently: "Let's look at today's schedule together..."
        - Reference familiar elements: "Remember how you enjoy your morning coffee..."
        - Provide clear information: "It's currently 2:30 PM, and we're in your favorite chair..."

        3. Positive Engagement:
        - Reference happy memories: "You mentioned teaching mathematics. Tell me about your favorite teaching moments..."
        - Encourage participation: "Would you like to join the garden group? I know how much you enjoy flowers..."
        - Focus on strengths: "Your memory of your wedding day is so vivid..."

        4. Comfort and Support:
        - Offer reassurance: "You're safe here, and we're here to help..."
        - Provide structure: "Let's go through your day together..."
        - Give choices: "Would you prefer to rest or join the activity?"

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
                        "text": "Patient's question or statement",
                        "emotional_state": "anxiety/confusion/joy/frustration/content",
                        "confusion_pattern": null  # Only for confused communications
                    },
                    "expected_response": {
                        "content": "Detailed, empathetic response with validation and gentle guidance",
                        "response_strategy": {
                            "validation": "How the response validates feelings",
                            "reorientation": "How the response provides orientation",
                            "engagement": "How the response encourages positive interaction",
                            "support": "Specific comfort strategies used"
                        }
                    }
                }
            ]
        }"""