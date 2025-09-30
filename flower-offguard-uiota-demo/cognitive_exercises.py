#!/usr/bin/env python3
"""
Cognitive Exercise Suite for Memory Guardian
Interactive exercises for memory, pattern recognition, problem solving, and reaction time
"""

import random
import time
from typing import Dict, List, Tuple
from datetime import datetime
import json


class MemoryExercise:
    """Memory recall and recognition exercises"""

    @staticmethod
    def sequence_memory(difficulty: int = 1) -> Dict:
        """
        Remember and recall a sequence of numbers/patterns
        Difficulty 1-5: length increases
        """
        length = 3 + (difficulty * 2)
        sequence = [random.randint(1, 9) for _ in range(length)]

        return {
            "type": "sequence_memory",
            "difficulty": difficulty,
            "sequence": sequence,
            "time_limit_seconds": 2 + (len(sequence) * 0.5),
            "instructions": f"Memorize this sequence: {' '.join(map(str, sequence))}",
            "scoring": {
                "perfect_match": 100,
                "per_correct_digit": 10,
                "time_bonus": "5 points per second under time limit"
            }
        }

    @staticmethod
    def image_grid_memory(difficulty: int = 1) -> Dict:
        """
        Remember positions of highlighted items in grid
        Difficulty 1-5: grid size and item count increases
        """
        grid_size = 3 + difficulty
        item_count = 2 + difficulty

        # Generate random positions
        all_positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        highlighted = random.sample(all_positions, min(item_count, len(all_positions)))

        return {
            "type": "image_grid_memory",
            "difficulty": difficulty,
            "grid_size": grid_size,
            "highlighted_positions": highlighted,
            "time_limit_seconds": 3 + (item_count * 0.8),
            "instructions": f"Memorize the positions of {item_count} highlighted squares in a {grid_size}x{grid_size} grid",
            "scoring": {
                "per_correct_position": 100 / item_count,
                "penalty_per_wrong": -10
            }
        }

    @staticmethod
    def word_pair_memory(difficulty: int = 1) -> Dict:
        """
        Remember word pairs and recall associations
        """
        word_pairs = [
            ("ocean", "wave"), ("mountain", "peak"), ("forest", "tree"),
            ("city", "building"), ("desert", "sand"), ("river", "flow"),
            ("garden", "flower"), ("sky", "cloud"), ("field", "grass"),
            ("lake", "water"), ("valley", "hill"), ("canyon", "rock")
        ]

        pair_count = 3 + difficulty
        selected_pairs = random.sample(word_pairs, min(pair_count, len(word_pairs)))

        return {
            "type": "word_pair_memory",
            "difficulty": difficulty,
            "word_pairs": selected_pairs,
            "time_limit_seconds": 5 + (pair_count * 2),
            "instructions": f"Memorize these {pair_count} word pairs",
            "scoring": {
                "per_correct_pair": 100 / pair_count
            }
        }


class PatternRecognitionExercise:
    """Pattern recognition and visual processing exercises"""

    @staticmethod
    def number_pattern(difficulty: int = 1) -> Dict:
        """
        Identify the pattern in a sequence and predict next number
        """
        patterns = {
            1: [lambda x: x + 2, "Add 2"],  # Arithmetic +2
            2: [lambda x: x * 2, "Multiply by 2"],  # Geometric x2
            3: [lambda x: x + (x // 2), "Add half of previous"],  # Add half
            4: [lambda x: x ** 2, "Square"],  # Squares
            5: [lambda x: x * 2 - 1, "Double minus 1"]  # Complex
        }

        pattern_func, pattern_desc = patterns[min(difficulty, 5)]
        start = random.randint(2, 5)

        sequence = [start]
        for _ in range(4):
            next_val = pattern_func(sequence[-1])
            if next_val > 10000:  # Prevent overflow
                break
            sequence.append(next_val)

        correct_answer = pattern_func(sequence[-1])

        return {
            "type": "number_pattern",
            "difficulty": difficulty,
            "sequence": sequence,
            "correct_answer": correct_answer,
            "instructions": "What comes next in this sequence?",
            "hint": pattern_desc if difficulty <= 2 else "Find the pattern",
            "scoring": {
                "correct": 100,
                "close_range": "50 points if within 10% of correct answer"
            }
        }

    @staticmethod
    def shape_pattern(difficulty: int = 1) -> Dict:
        """
        Identify missing shape in pattern
        """
        shapes = ["circle", "square", "triangle", "diamond", "hexagon", "star"]
        pattern_length = 4 + difficulty

        # Create repeating pattern with variations
        if difficulty <= 2:
            # Simple repeat
            base_pattern = random.sample(shapes, 2 + difficulty)
            full_pattern = (base_pattern * 3)[:pattern_length]
        else:
            # Alternating or complex
            base_pattern = random.sample(shapes, 3)
            full_pattern = []
            for i in range(pattern_length):
                full_pattern.append(base_pattern[i % len(base_pattern)])

        # Remove one element for user to guess
        missing_index = random.randint(1, len(full_pattern) - 2)
        correct_answer = full_pattern[missing_index]
        full_pattern[missing_index] = "?"

        return {
            "type": "shape_pattern",
            "difficulty": difficulty,
            "pattern": full_pattern,
            "missing_index": missing_index,
            "correct_answer": correct_answer,
            "options": random.sample(shapes, 4),
            "instructions": "Which shape is missing from the pattern?",
            "scoring": {
                "correct": 100
            }
        }

    @staticmethod
    def matrix_pattern(difficulty: int = 1) -> Dict:
        """
        3x3 matrix pattern recognition (like Raven's Progressive Matrices)
        """
        size = 3

        # Simplified matrix with progression patterns
        matrix = []
        for row in range(size):
            matrix_row = []
            for col in range(size):
                # Create pattern based on row and column
                value = (row + col + 1) % 6
                matrix_row.append(value)
            matrix.append(matrix_row)

        # Remove bottom-right for user to solve
        correct_answer = matrix[size-1][size-1]
        matrix[size-1][size-1] = "?"

        return {
            "type": "matrix_pattern",
            "difficulty": difficulty,
            "matrix": matrix,
            "correct_answer": correct_answer,
            "instructions": "What number completes the pattern?",
            "scoring": {
                "correct": 100
            }
        }


class ProblemSolvingExercise:
    """Logical reasoning and problem-solving exercises"""

    @staticmethod
    def math_puzzle(difficulty: int = 1) -> Dict:
        """
        Mathematical word problems and puzzles
        """
        puzzles = {
            1: {
                "question": "If you have 12 apples and give away 5, how many do you have left?",
                "answer": 7,
                "hint": "Subtraction problem"
            },
            2: {
                "question": "A train travels 60 miles in 1 hour. How far does it travel in 3 hours?",
                "answer": 180,
                "hint": "Distance = Speed × Time"
            },
            3: {
                "question": "If 4 workers can build a wall in 6 days, how many days for 6 workers?",
                "answer": 4,
                "hint": "Inverse proportion"
            },
            4: {
                "question": "A square garden has area 144 sq ft. What is the perimeter?",
                "answer": 48,
                "hint": "Side = sqrt(area), Perimeter = 4 × side"
            },
            5: {
                "question": "What is the next prime number after 17?",
                "answer": 19,
                "hint": "Test divisibility"
            }
        }

        puzzle = puzzles[min(difficulty, 5)]

        return {
            "type": "math_puzzle",
            "difficulty": difficulty,
            "question": puzzle["question"],
            "correct_answer": puzzle["answer"],
            "hint": puzzle["hint"] if difficulty <= 2 else None,
            "time_limit_seconds": 30 + (difficulty * 10),
            "scoring": {
                "correct": 100,
                "time_bonus": "1 point per second remaining"
            }
        }

    @staticmethod
    def logic_puzzle(difficulty: int = 1) -> Dict:
        """
        Logical deduction puzzles
        """
        puzzles = {
            1: {
                "question": "All cats are animals. Fluffy is a cat. Is Fluffy an animal?",
                "answer": "yes",
                "options": ["yes", "no", "cannot determine"]
            },
            2: {
                "question": "If it rains, the ground gets wet. The ground is wet. Did it rain?",
                "answer": "cannot determine",
                "options": ["yes", "no", "cannot determine"],
                "explanation": "Wet ground could have other causes"
            },
            3: {
                "question": "No birds are mammals. All bats are mammals. Are bats birds?",
                "answer": "no",
                "options": ["yes", "no", "cannot determine"]
            },
            4: {
                "question": "If A > B and B > C, which is smallest?",
                "answer": "C",
                "options": ["A", "B", "C", "cannot determine"]
            },
            5: {
                "question": "Three people: Alice, Bob, Carol. Alice is taller than Bob. Carol is shorter than Bob. Who is tallest?",
                "answer": "Alice",
                "options": ["Alice", "Bob", "Carol"]
            }
        }

        puzzle = puzzles[min(difficulty, 5)]

        return {
            "type": "logic_puzzle",
            "difficulty": difficulty,
            "question": puzzle["question"],
            "correct_answer": puzzle["answer"],
            "options": puzzle["options"],
            "explanation": puzzle.get("explanation"),
            "scoring": {
                "correct": 100
            }
        }

    @staticmethod
    def strategy_puzzle(difficulty: int = 1) -> Dict:
        """
        Strategic thinking and planning exercises
        """
        return {
            "type": "strategy_puzzle",
            "difficulty": difficulty,
            "scenario": "You need to cross a river with a boat that holds 2 items. You have: wolf, goat, cabbage. Wolf eats goat, goat eats cabbage.",
            "question": "What order do you transport them?",
            "correct_sequence": ["goat", "return", "wolf", "return_with_goat", "cabbage", "return", "goat"],
            "instructions": "Plan the sequence to get all across safely",
            "scoring": {
                "perfect_solution": 100,
                "valid_solution": 75
            }
        }


class ReactionTimeExercise:
    """Reaction time and processing speed exercises"""

    @staticmethod
    def simple_reaction(difficulty: int = 1) -> Dict:
        """
        Simple stimulus-response reaction time test
        """
        delay = random.uniform(1.0, 3.0)

        return {
            "type": "simple_reaction",
            "difficulty": difficulty,
            "delay_seconds": delay,
            "stimulus": "color_change",  # Screen changes color
            "instructions": "Click/tap as soon as you see the color change",
            "scoring": {
                "excellent": "< 200ms = 100 points",
                "good": "200-300ms = 80 points",
                "average": "300-400ms = 60 points",
                "slow": "> 400ms = 40 points"
            }
        }

    @staticmethod
    def choice_reaction(difficulty: int = 1) -> Dict:
        """
        Multiple choice reaction time (discriminate between stimuli)
        """
        choices = 2 + difficulty
        correct_choice = random.randint(0, choices - 1)

        return {
            "type": "choice_reaction",
            "difficulty": difficulty,
            "number_of_choices": choices,
            "correct_choice": correct_choice,
            "instructions": f"Click the highlighted button as fast as possible",
            "scoring": {
                "base_score": 100,
                "time_penalty": "-5 points per 100ms",
                "wrong_choice": "-50 points"
            }
        }

    @staticmethod
    def go_no_go(difficulty: int = 1) -> Dict:
        """
        Inhibitory control test - respond to some stimuli, not others
        """
        trial_count = 10 + (difficulty * 5)

        # 70% go, 30% no-go
        trials = []
        for _ in range(trial_count):
            if random.random() < 0.7:
                trials.append({"type": "go", "stimulus": "green_circle"})
            else:
                trials.append({"type": "no-go", "stimulus": "red_square"})

        return {
            "type": "go_no_go",
            "difficulty": difficulty,
            "trials": trials,
            "instructions": "Click for GREEN CIRCLES only, ignore RED SQUARES",
            "scoring": {
                "correct_go": 10,
                "correct_no_go": 15,
                "false_positive": -20,
                "miss": -10
            }
        }


class CognitiveExerciseSuite:
    """Main controller for cognitive exercise system"""

    def __init__(self):
        self.exercises = {
            "memory": MemoryExercise(),
            "pattern": PatternRecognitionExercise(),
            "problem_solving": ProblemSolvingExercise(),
            "reaction": ReactionTimeExercise()
        }

    def generate_daily_assessment(self, difficulty: int = 1) -> Dict:
        """
        Generate a complete daily cognitive assessment
        Includes exercises from all categories
        """
        assessment = {
            "assessment_id": f"daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "difficulty": difficulty,
            "exercises": []
        }

        # Memory exercises (2)
        assessment["exercises"].append(
            MemoryExercise.sequence_memory(difficulty)
        )
        assessment["exercises"].append(
            MemoryExercise.word_pair_memory(difficulty)
        )

        # Pattern recognition (2)
        assessment["exercises"].append(
            PatternRecognitionExercise.number_pattern(difficulty)
        )
        assessment["exercises"].append(
            PatternRecognitionExercise.shape_pattern(difficulty)
        )

        # Problem solving (2)
        assessment["exercises"].append(
            ProblemSolvingExercise.math_puzzle(difficulty)
        )
        assessment["exercises"].append(
            ProblemSolvingExercise.logic_puzzle(difficulty)
        )

        # Reaction time (2)
        assessment["exercises"].append(
            ReactionTimeExercise.simple_reaction(difficulty)
        )
        assessment["exercises"].append(
            ReactionTimeExercise.choice_reaction(difficulty)
        )

        return assessment

    def calculate_scores(self, assessment_results: Dict) -> Dict:
        """
        Calculate overall scores from completed assessment
        """
        category_scores = {
            "memory": [],
            "pattern_recognition": [],
            "problem_solving": [],
            "reaction_time": []
        }

        for result in assessment_results.get("completed_exercises", []):
            exercise_type = result["type"]
            score = result["score"]

            if "memory" in exercise_type:
                category_scores["memory"].append(score)
            elif "pattern" in exercise_type or "matrix" in exercise_type:
                category_scores["pattern_recognition"].append(score)
            elif "puzzle" in exercise_type:
                category_scores["problem_solving"].append(score)
            elif "reaction" in exercise_type or "go_no_go" in exercise_type:
                category_scores["reaction_time"].append(score)

        # Calculate averages
        final_scores = {}
        for category, scores in category_scores.items():
            if scores:
                final_scores[category] = sum(scores) / len(scores)
            else:
                final_scores[category] = 0

        # Calculate overall score
        overall = sum(final_scores.values()) / len(final_scores)

        return {
            "memory_score": final_scores["memory"],
            "pattern_recognition_score": final_scores["pattern_recognition"],
            "problem_solving_score": final_scores["problem_solving"],
            "reaction_time_ms": 500 - (final_scores["reaction_time"] * 3),  # Convert score to ms
            "overall_score": overall,
            "timestamp": datetime.now().isoformat()
        }

    def adaptive_difficulty(self, recent_scores: List[float]) -> int:
        """
        Adjust difficulty based on recent performance
        """
        if not recent_scores:
            return 1

        avg_score = sum(recent_scores) / len(recent_scores)

        if avg_score >= 90:
            return min(5, 3 + 1)  # Increase difficulty
        elif avg_score >= 75:
            return 3
        elif avg_score >= 60:
            return 2
        else:
            return 1  # Stay at easier level


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("COGNITIVE EXERCISE SUITE")
    print("=" * 80)

    suite = CognitiveExerciseSuite()

    # Generate daily assessment
    print("\n📝 Generating Daily Assessment...")
    assessment = suite.generate_daily_assessment(difficulty=2)

    print(f"\nAssessment ID: {assessment['assessment_id']}")
    print(f"Difficulty Level: {assessment['difficulty']}")
    print(f"Total Exercises: {len(assessment['exercises'])}\n")

    # Display sample exercises
    for i, exercise in enumerate(assessment['exercises'][:3], 1):
        print(f"\n--- Exercise {i}: {exercise['type']} ---")
        print(f"Difficulty: {exercise['difficulty']}")
        print(f"Instructions: {exercise.get('instructions', 'N/A')}")
        if 'question' in exercise:
            print(f"Question: {exercise['question']}")
        if 'sequence' in exercise:
            print(f"Sequence: {exercise['sequence']}")
        print(f"Scoring: {exercise['scoring']}")

    # Simulate assessment results
    print("\n\n📊 Simulating Assessment Results...")
    simulated_results = {
        "assessment_id": assessment['assessment_id'],
        "completed_exercises": [
            {"type": "sequence_memory", "score": 85},
            {"type": "word_pair_memory", "score": 90},
            {"type": "number_pattern", "score": 88},
            {"type": "shape_pattern", "score": 92},
            {"type": "math_puzzle", "score": 82},
            {"type": "logic_puzzle", "score": 87},
            {"type": "simple_reaction", "score": 75},
            {"type": "choice_reaction", "score": 80}
        ]
    }

    final_scores = suite.calculate_scores(simulated_results)

    print("\n🎯 Final Scores:")
    print(f"   Memory: {final_scores['memory_score']:.1f}")
    print(f"   Pattern Recognition: {final_scores['pattern_recognition_score']:.1f}")
    print(f"   Problem Solving: {final_scores['problem_solving_score']:.1f}")
    print(f"   Reaction Time: {final_scores['reaction_time_ms']:.0f}ms")
    print(f"   Overall Score: {final_scores['overall_score']:.1f}/100")

    # Test adaptive difficulty
    print("\n\n🎚️  Testing Adaptive Difficulty...")
    test_scores = [85, 88, 90, 92, 91]
    new_difficulty = suite.adaptive_difficulty(test_scores)
    print(f"   Recent scores: {test_scores}")
    print(f"   Average: {sum(test_scores)/len(test_scores):.1f}")
    print(f"   Recommended difficulty: {new_difficulty}")

    print("\n✅ Cognitive Exercise Suite Demo Complete!")