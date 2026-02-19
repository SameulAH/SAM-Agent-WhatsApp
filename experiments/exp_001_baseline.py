"""
Experiment Harness - Baseline (EXP-001)

Controlled experiment with 50 fixed prompts to measure intelligence metrics
deterministically and safely.

This baseline establishes:
1. Fact extraction accuracy on known inputs
2. Memory ranking quality
3. Tool integration safety
4. Guardrail enforcement
5. Agent decision stability

All metrics are saved to experiments/results/exp_001_baseline.csv
"""

import csv
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# TODO: Import when agent is fully integrated
# from agent.langgraph_orchestrator import SAMAgentOrchestrator
# from agent.intelligence.metrics import get_metrics_collector
# from agent.intelligence.guardrails import get_guardrails


class BaselineExperimentPrompt:
    """Single prompt in baseline experiment."""
    
    def __init__(
        self,
        prompt_id: str,
        user_input: str,
        expected_fact_extraction: List[str],
        expected_tool_use: bool,
        expected_memory_write: bool,
        description: str,
    ):
        self.prompt_id = prompt_id
        self.user_input = user_input
        self.expected_fact_extraction = expected_fact_extraction
        self.expected_tool_use = expected_tool_use
        self.expected_memory_write = expected_memory_write
        self.description = description


class BaselineExperiment:
    """
    EXP-001: Baseline Intelligence Measurement
    
    50 carefully designed prompts to test:
    1. User preference extraction
    2. Personal fact tracking
    3. Goal identification
    4. Task tracking
    5. Tool integration safety
    6. Memory guardrails
    """
    
    def __init__(self):
        self.prompts = self._generate_prompts()
        self.results = []
    
    def _generate_prompts(self) -> List[BaselineExperimentPrompt]:
        """Generate 50 baseline prompts."""
        
        prompts = [
            # Category 1: User Preferences (Prompts 1-10)
            BaselineExperimentPrompt(
                prompt_id="exp_001_001",
                user_input="I prefer coffee over tea",
                expected_fact_extraction=["coffee preference"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Simple preference statement",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_002",
                user_input="I don't like spicy food",
                expected_fact_extraction=["food preference", "no spicy"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Negative preference",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_003",
                user_input="My favorite programming language is Python",
                expected_fact_extraction=["programming language", "Python"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Language preference",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_004",
                user_input="I work best in the morning",
                expected_fact_extraction=["work schedule", "morning"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Time preference",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_005",
                user_input="I prefer email communication",
                expected_fact_extraction=["communication preference", "email"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Communication preference",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_006",
                user_input="What time is it?",
                expected_fact_extraction=[],
                expected_tool_use=False,
                expected_memory_write=False,
                description="Question with no preference",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_007",
                user_input="I like working with teams",
                expected_fact_extraction=["work style", "team"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Work style preference",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_008",
                user_input="I usually sleep 8 hours",
                expected_fact_extraction=["sleep habit"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Personal habit",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_009",
                user_input="I enjoy reading books",
                expected_fact_extraction=["hobby", "reading"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Hobby/interest",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_010",
                user_input="I usually prefer written instructions",
                expected_fact_extraction=["learning preference"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Learning preference",
            ),
            
            # Category 2: Personal Facts (Prompts 11-20)
            BaselineExperimentPrompt(
                prompt_id="exp_001_011",
                user_input="I live in San Francisco",
                expected_fact_extraction=["location", "San Francisco"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Location fact",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_012",
                user_input="I'm a software engineer",
                expected_fact_extraction=["profession", "engineer"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Profession",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_013",
                user_input="I have 5 years of experience",
                expected_fact_extraction=["experience", "5 years"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Experience level",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_014",
                user_input="I speak English and Spanish",
                expected_fact_extraction=["languages", "English", "Spanish"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Language skills",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_015",
                user_input="I'm interested in AI and machine learning",
                expected_fact_extraction=["interests", "AI", "machine learning"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Research interests",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_016",
                user_input="I graduated from Stanford University",
                expected_fact_extraction=["education", "Stanford"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Educational background",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_017",
                user_input="I'm married with 2 kids",
                expected_fact_extraction=["family", "married"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Family status",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_018",
                user_input="I'm originally from New York",
                expected_fact_extraction=["origin", "New York"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Origin fact",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_019",
                user_input="I play tennis on weekends",
                expected_fact_extraction=["hobby", "tennis", "weekend"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Recreation activity",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_020",
                user_input="I'm currently learning Japanese",
                expected_fact_extraction=["learning", "Japanese"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Learning goal",
            ),
            
            # Category 3: Goals (Prompts 21-30)
            BaselineExperimentPrompt(
                prompt_id="exp_001_021",
                user_input="I want to become a machine learning engineer",
                expected_fact_extraction=["goal", "machine learning"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Career goal",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_022",
                user_input="I'm trying to improve my public speaking",
                expected_fact_extraction=["goal", "public speaking"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Skill development goal",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_023",
                user_input="I want to read 50 books this year",
                expected_fact_extraction=["goal", "reading", "50 books"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Reading goal",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_024",
                user_input="I'm aiming to run a marathon",
                expected_fact_extraction=["goal", "marathon"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Fitness goal",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_025",
                user_input="I want to start my own company",
                expected_fact_extraction=["goal", "company", "entrepreneur"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Entrepreneurship goal",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_026",
                user_input="I'm trying to save 20% of my income",
                expected_fact_extraction=["goal", "savings"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Financial goal",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_027",
                user_input="I want to travel to 10 countries",
                expected_fact_extraction=["goal", "travel"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Travel goal",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_028",
                user_input="I'm aiming for a promotion this year",
                expected_fact_extraction=["goal", "promotion"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Career advancement goal",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_029",
                user_input="I want to improve my health",
                expected_fact_extraction=["goal", "health"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Health goal",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_030",
                user_input="I'm learning to code from scratch",
                expected_fact_extraction=["goal", "programming"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Learning goal",
            ),
            
            # Category 4: Tasks (Prompts 31-40)
            BaselineExperimentPrompt(
                prompt_id="exp_001_031",
                user_input="I need to write a report by Friday",
                expected_fact_extraction=["task", "report", "Friday"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Time-bound task",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_032",
                user_input="I have to prepare slides for a presentation",
                expected_fact_extraction=["task", "presentation"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Presentation task",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_033",
                user_input="I need to review the code changes",
                expected_fact_extraction=["task", "code review"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Code review task",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_034",
                user_input="I want to schedule a meeting",
                expected_fact_extraction=["task", "meeting"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Meeting task",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_035",
                user_input="I need to call the client tomorrow",
                expected_fact_extraction=["task", "call", "client"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Call task",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_036",
                user_input="I have to debug this issue",
                expected_fact_extraction=["task", "debug"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Debugging task",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_037",
                user_input="I need to send an email to the team",
                expected_fact_extraction=["task", "email"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Email task",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_038",
                user_input="I want to backup my files",
                expected_fact_extraction=["task", "backup"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Backup task",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_039",
                user_input="I need to update the documentation",
                expected_fact_extraction=["task", "documentation"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Documentation task",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_040",
                user_input="I want to install new software",
                expected_fact_extraction=["task", "install", "software"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Installation task",
            ),
            
            # Category 5: Mixed & Edge Cases (Prompts 41-50)
            BaselineExperimentPrompt(
                prompt_id="exp_001_041",
                user_input="Hello",
                expected_fact_extraction=[],
                expected_tool_use=False,
                expected_memory_write=False,
                description="Minimal greeting",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_042",
                user_input="Tell me a joke",
                expected_fact_extraction=[],
                expected_tool_use=False,
                expected_memory_write=False,
                description="Entertainment request",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_043",
                user_input="What is 2+2?",
                expected_fact_extraction=[],
                expected_tool_use=False,
                expected_memory_write=False,
                description="Math question",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_044",
                user_input="I love Python and I work on AI projects",
                expected_fact_extraction=["Python preference", "AI interests"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Compound statement",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_045",
                user_input="My name is Alex and I'm from Boston",
                expected_fact_extraction=["name", "location"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Identity statement",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_046",
                user_input="I don't know anything about that",
                expected_fact_extraction=[],
                expected_tool_use=False,
                expected_memory_write=False,
                description="Negation statement",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_047",
                user_input="Thanks for your help",
                expected_fact_extraction=[],
                expected_tool_use=False,
                expected_memory_write=False,
                description="Gratitude statement",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_048",
                user_input="I previously mentioned I like coffee",
                expected_fact_extraction=["coffee preference"],
                expected_tool_use=False,
                expected_memory_write=False,
                description="Reference to past fact",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_049",
                user_input="Can you help me with Python programming?",
                expected_fact_extraction=["Python interest"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Help request with context",
            ),
            BaselineExperimentPrompt(
                prompt_id="exp_001_050",
                user_input="I changed my mind - I actually prefer tea",
                expected_fact_extraction=["tea preference", "preference change"],
                expected_tool_use=False,
                expected_memory_write=True,
                description="Preference change",
            ),
        ]
        
        return prompts
    
    def run(self, agent=None) -> List[Dict[str, Any]]:
        """
        Run baseline experiment.
        
        NOTE: This is a skeleton. Actual execution requires SAMAgentOrchestrator
        integration.
        """
        
        # TODO: Implement when agent is integrated
        
        if agent is None:
            print("Note: Agent not provided. Skipping baseline execution.")
            print(f"Baseline has {len(self.prompts)} prompts ready for testing.")
            return []
        
        results = []
        
        for i, prompt in enumerate(self.prompts, 1):
            print(f"Running prompt {i}/50: {prompt.prompt_id}")
            
            # TODO: Invoke agent with prompt
            # result = agent.invoke(prompt.user_input)
            
            # TODO: Collect metrics
            # metrics = get_metrics_collector().emit()
            
            # TODO: Verify guardrails
            # guardrails = get_guardrails()
            
            # result_dict = {
            #     "prompt_id": prompt.prompt_id,
            #     "description": prompt.description,
            #     "timestamp": datetime.utcnow().isoformat(),
            #     ...
            # }
            # results.append(result_dict)
        
        self.results = results
        return results
    
    def save_results(self, output_path: str = "experiments/results/exp_001_baseline.csv"):
        """Save results to CSV."""
        
        if not self.results:
            print("No results to save.")
            return
        
        # TODO: Implement CSV writing
        print(f"Would save {len(self.results)} results to {output_path}")
    
    def print_summary(self):
        """Print summary of baseline."""
        print("\n" + "=" * 60)
        print("EXP-001: Baseline Intelligence Measurement")
        print("=" * 60)
        print(f"Total Prompts: {len(self.prompts)}")
        print(f"\nPrompt Categories:")
        print(f"  - User Preferences (1-10)")
        print(f"  - Personal Facts (11-20)")
        print(f"  - Goals (21-30)")
        print(f"  - Tasks (31-40)")
        print(f"  - Mixed & Edge Cases (41-50)")
        print(f"\nMetrics to Collect:")
        print(f"  - Fact extraction accuracy")
        print(f"  - Memory write decisions")
        print(f"  - Tool use safety")
        print(f"  - Guardrail compliance")
        print(f"  - Response latency")
        print(f"  - Memory growth rate")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Initialize baseline
    baseline = BaselineExperiment()
    baseline.print_summary()
    
    # In actual use:
    # from agent.langgraph_orchestrator import SAMAgentOrchestrator
    # agent = SAMAgentOrchestrator()
    # baseline.run(agent)
    # baseline.save_results()
