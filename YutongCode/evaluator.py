from typing import Dict, List, Any
from datetime import datetime
import json
import pathlib
from dataclasses import dataclass
from openai import OpenAI
import numpy as np
from tqdm import tqdm

@dataclass
class TestResult:
    """Test result data structure"""
    query: str
    expected_response: Dict[str, Any]
    generated_response: str
    metrics: Dict[str, float]
    facts_verified: Dict[str, List[str]]

class ModelEvaluator:
    def __init__(self, api_key: str, test_data_path: str):
        self.api_key = api_key
        self.test_data_path = pathlib.Path(test_data_path)
        self.client = OpenAI(api_key=api_key)
        
    def run_evaluation(self, patient_id: str) -> Dict[str, Any]:
        """Run complete evaluation process"""
        # Load test data
        test_cases = self._load_test_cases(patient_id)
        
        # Run model and get responses
        results = []
        for test_case in tqdm(test_cases, desc="Evaluating responses"):
            result = self._evaluate_single_case(test_case)
            results.append(result)
            
        # Summarize evaluation results
        summary = self._summarize_results(results)
        
        return {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "test_cases": len(results),
            "detailed_results": results,
            "summary": summary
        }
        
    def _load_test_cases(self, patient_id: str) -> List[Dict[str, Any]]:
        """Load test cases"""
        file_path = self.test_data_path / f"{patient_id}.json"
        with open(file_path) as f:
            data = json.load(f)
        return data["dialogues"]["dialogues"]

    def _evaluate_single_case(self, test_case: Dict[str, Any]) -> TestResult:
        """Evaluate single test case"""
        # Generate response
        current_time = datetime.now()  # Can be set to specific time if needed
        response = integrated_memory_search(
            patient_id="P001",  # Use actual patient_id
            api_key=self.api_key,
            dialogue=test_case["query"],
            current_time=current_time
        )
        
        generated_response = response["response"]
        
        # Evaluate response
        metrics = self._evaluate_response(
            generated_response,
            test_case["expected_response"],
            test_case["query"]
        )
        
        # Verify facts
        facts_verified = self._verify_facts(
            generated_response,
            test_case["expected_response"]["referenced_data"]["actual_facts"]
        )
        
        return TestResult(
            query=test_case["query"],
            expected_response=test_case["expected_response"],
            generated_response=generated_response,
            metrics=metrics,
            facts_verified=facts_verified
        )

    def _evaluate_response(self, 
                          generated: str, 
                          expected: Dict[str, Any], 
                          query: str) -> Dict[str, float]:
        """Evaluate response quality"""
        prompt = f"""Evaluate the quality of this dementia care dialogue response:

Original Query: {query}
Expected Response: {expected["content"]}
Generated Response: {generated}

Please evaluate the following dimensions (score 0-1):
1. Factual Accuracy: Are the facts in the response accurate?
2. Empathy Level: Does it show understanding and support for the patient's emotions?
3. Communication Clarity: Is the expression clear and easy to understand?
4. Confusion Handling: How well does it address patient's confusion?
5. Supportiveness: Does it provide appropriate support and guidance?

Return evaluation in JSON format:
{{
    "factual_accuracy": float,
    "empathy": float,
    "clarity": float,
    "confusion_handling": float,
    "supportiveness": float,
    "analysis": string
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional dementia care dialogue evaluation expert"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            results = json.loads(response.choices[0].message.content)
            return {k: v for k, v in results.items() if isinstance(v, (int, float))}
        except Exception as e:
            print(f"Error in evaluation process: {e}")
            return {
                "factual_accuracy": 0.0,
                "empathy": 0.0,
                "clarity": 0.0,
                "confusion_handling": 0.0,
                "supportiveness": 0.0
            }

    def _verify_facts(self, 
                     generated: str, 
                     expected_facts: List[str]) -> Dict[str, List[str]]:
        """Verify facts in response"""
        prompt = f"""Verify the factual accuracy of the generated response:

Generated Response: {generated}
Expected Facts: {expected_facts}

Please identify:
1. Correctly mentioned facts
2. Incorrect facts
3. Missing facts

Return results in JSON format:
{{
    "correct_facts": [str],
    "incorrect_facts": [str],
    "missing_facts": [str]
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional fact verification expert"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error in fact verification process: {e}")
            return {
                "correct_facts": [],
                "incorrect_facts": [],
                "missing_facts": expected_facts
            }

    def _summarize_results(self, results: List[TestResult]) -> Dict[str, Any]:
        """Summarize evaluation results"""
        metrics_summary = {
            "factual_accuracy": [],
            "empathy": [],
            "clarity": [],
            "confusion_handling": [],
            "supportiveness": []
        }
        
        fact_verification = {
            "correct_facts": 0,
            "incorrect_facts": 0,
            "missing_facts": 0
        }
        
        for result in results:
            # Summarize evaluation metrics
            for metric, value in result.metrics.items():
                if metric in metrics_summary:
                    metrics_summary[metric].append(value)
            
            # Summarize fact verification results
            fact_verification["correct_facts"] += len(result.facts_verified["correct_facts"])
            fact_verification["incorrect_facts"] += len(result.facts_verified["incorrect_facts"])
            fact_verification["missing_facts"] += len(result.facts_verified["missing_facts"])
        
        # Calculate average scores
        metrics_avg = {
            metric: np.mean(values) for metric, values in metrics_summary.items()
        }
        
        return {
            "average_metrics": metrics_avg,
            "fact_verification_summary": fact_verification,
            "overall_score": np.mean(list(metrics_avg.values()))
        }

def main():
    # Setup parameters
    API_KEY = "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"
    TEST_DATA_PATH = "Patient_Data/Ground_Truth/"
    PATIENT_ID = "P001_dialogues"
    
    # Initialize evaluator
    evaluator = ModelEvaluator(API_KEY, TEST_DATA_PATH)
    
    # Run evaluation
    results = evaluator.run_evaluation(PATIENT_ID)
    
    # Save results
    output_path = f"evaluation_results_{PATIENT_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, indent=2, ensure_ascii=False, fp=f)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(json.dumps(results["summary"], indent=2))

if __name__ == "__main__":
    main()
