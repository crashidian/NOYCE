"""
Test script for memory agent
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add the parent directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# First, try to verify the import path is correct
logger.info("Python path: %s", sys.path)
logger.info("Current directory: %s", current_dir)

try:
    # Try importing factory directly first to debug
    from memory_agent.factory import create_agent
    logger.info("Successfully imported create_agent from factory")
    
    def test_agent():
        # Create agent
        agent = create_agent(
            patient_id="P001",
            api_key="your-openai-api-key-here"
        )
        
        # Test dialogue
        current_time = datetime.strptime("08:00", "%H:%M")
        dialogue = "Is my daughter coming for lunch today?"
        
        logger.info("Testing agent with dialogue: %s", dialogue)
        
        try:
            results = agent.analyze_dialogue(dialogue, current_time)
            
            # Print results in a formatted way
            logger.info("\nAnalysis Results:")
            logger.info("Keywords: %s", results.get('keywords', {}))
            logger.info("Current Activity: %s", results.get('current_activity', {}))
            logger.info("Search Adaptation: %s", results.get('search_adaptation', {}))
            logger.info("\nRelevant Information:")
            for key, value in results.get('relevant_info', {}).items():
                logger.info(f"{key}: {value}")
                
            return True
            
        except Exception as e:
            logger.error("Error during dialogue analysis: %s", str(e))
            return False

    if __name__ == "__main__":
        logger.info("Starting agent test...")
        success = test_agent()
        if success:
            logger.info("Test completed successfully")
        else:
            logger.error("Test failed")

except ImportError as e:
    logger.error("Import error: %s", e)
    
    # Additional debugging information
    logger.info("\nChecking package structure:")
    memory_agent_dir = current_dir / 'memory_agent'
    
    if memory_agent_dir.exists():
        logger.info("memory_agent directory exists")
        logger.info("Contents of memory_agent directory:")
        for item in memory_agent_dir.iterdir():
            logger.info(f"- {item.name}")
        
        init_file = memory_agent_dir / '__init__.py'
        if init_file.exists():
            logger.info("__init__.py exists")
            with open(init_file) as f:
                logger.info("Contents of __init__.py:")
                logger.info(f.read())
        else:
            logger.error("__init__.py is missing!")
            
        factory_file = memory_agent_dir / 'factory.py'
        if factory_file.exists():
            logger.info("factory.py exists")
            with open(factory_file) as f:
                logger.info("First few lines of factory.py:")
                logger.info("\n".join(f.readlines()[:10]))
        else:
            logger.error("factory.py is missing!")
    else:
        logger.error("memory_agent directory not found!")
    
    raise