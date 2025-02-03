import os
from pathlib import Path
from make_kg_data import IntegratedPatientGenerator

def generate_patients(count: int):

    api_key = "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"
    os.environ["OPENAI_API_KEY"] = api_key

    generator = IntegratedPatientGenerator(api_key=api_key)
    print(f"\nGenerating {count} patients...")
    results = generator.generate_multiple_patients(count=count)
    print("\nGeneration Summary:")
    print(f"Total patients requested: {count}")
    print(f"Successfully generated: {len(results)}")
    print(f"Success rate: {(len(results)/count)*100:.2f}%")
    
    return results


results = generate_patients(count=2)