
# CODE Outline 
    #0. SETUP
    #   0.1 imports
    #   0.2 clingo_check
    #   0.3 values_to_asp
    #1. CORE LOGIC
    #   1.1 Dataclass [derived step] 
    #   1.2 Asp functions
    #   1.3 Dataclass [scratchpad]
    #   1.4 Build Prompt 
    #   1.5 Parse 
    #   1.6 Build Reasoning loop 
    #   1.7 Dataclass [Test Case]
    #2. EXPERIMENTAL CONDITIONS
    #   2.1 Baseline
    #   2.2 Chain of Thought (COT)
    #   2.3 Hybrid (Neuro-symbolic) 
    #3. LLM INTERFACE
    #   3.1 Call LLM
    #4. TEST CASES 
    #5. RUN EXPERIMENTS 


# -------------------------
########################### 0. SETUP
# -------------------------

# -------------------------
# Imports 
# -------------------------

from dataclasses import dataclass, field
from typing import Dict, Union, List, Optional
import clingo
import json
import re
import os 
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# CLINGO Check
# -------------------------

## Basic clingo test 
ctl = clingo.Control() #create solver instance 
ctl.add("base", [], "a :- not b.") #add rule to solver
ctl.ground([("base", [])]) #ground rules
print(ctl.solve().satisfiable) #solve and print result

#clingo check function -> checks if program is consistent
def clingo_check(program: str) -> bool:
    ctl = clingo.Control() #create solver instance 
    ctl.add("base", [], program) #add program to solver
    ctl.ground([("base", [])]) #ground rules
    result = ctl.solve() #solve
    return result.satisfiable #return result

#encode current values into ASP format -> encoding Python state into logic facts so Clingo can reason over it!
def values_to_asp(values: Dict[str, int]) -> str:
    lines = []
    for var, val in values.items():
        lines.append(f"value({var},{val}).")
    return "\n".join(lines)



# -------------------------
########################### 1. CORE LOGIC
# -------------------------

# -------------------------
# Derived Step
# -------------------------

@dataclass
#data class for a derived step -> a step in the reasoning process
class DerivedStep:
    step_id: str
    op: str
    # allow either variable names (str) or ints
    inputs: List[Union[str, int]]
    relation: str
    result_var: str
    result_value: int

# -------------------------
# ASP Functions
# -------------------------

#encode step into ASP format -> encoding the step into logic facts so Clingo can reason over it!
def operand_to_asp(name, symbol):
    if isinstance(name, int):
        return f"{symbol} = {name}."
    else:
        return f"value({name},{symbol})."

def step_to_asp(step: DerivedStep) -> str:
    op_map = {
        "add": "+",
        "subtract": "-",  
        "multiply": "*",
        "divide": "/",    
    }

    if step.op not in op_map:
        raise ValueError(f"Unsupported op: {step.op}")

    operator = op_map[step.op]

    body_atoms = []
    exprs = []

    for i, inp in enumerate(step.inputs):
        if isinstance(inp, str):
            var_name = f"V{i}"
            body_atoms.append(f"value({inp},{var_name})")
            exprs.append(var_name)
        else:
            exprs.append(str(inp))

    # Only multi-input sum 
    if step.op == "add":
        expr = " + ".join(exprs)
    elif step.op == "multiply":
        expr = " * ".join(exprs)
    else:
        # keep binary operations 
        if len(exprs) != 2:
            raise ValueError(f"{step.op} must have exactly 2 inputs")
        expr = f"{exprs[0]} {operator} {exprs[1]}"

    body_atoms.append(f"proposed({step.result_var},C)")
    body_atoms.append(f"C = {expr}")
    body_atoms.append(f"C = {step.result_value}")

    body = ",\n    ".join(body_atoms) + "."

    return f"""
proposed({step.result_var},{step.result_value}).

valid :-
    {body}

:- not valid.
"""

# -------------------------
# SCRATCHPAD
# -------------------------

## Scratchpad class -> tracks the current state of the reasoning process

@dataclass
class Scratchpad:
    problem: str
    facts: Dict[str, int] #ground facts from the problem/initial known values
    goal: str

    derived_steps: List[DerivedStep] = field(default_factory=list) #accepted reasoning steps 
    rejected_steps: List[DerivedStep] = field(default_factory=list) #rejected reasoning steps 
    answer: Optional[int] = None #final answer to the problem
    log: List[Dict] = field(default_factory=list) #log of the reasoning process


    def current_values(self) -> Dict[str, int]: #get the current values of the variables
        values = dict(self.facts)
        for step in self.derived_steps: #add the values of the derived steps to the current values
            values[step.result_var] = step.result_value
        return values

    def apply_step(self, step: DerivedStep):
        # Assign unique step_id automatically
        step.step_id = f"s{len(self.derived_steps) + 1}"

        current_vals = self.current_values()

        log_entry = {
            "step_id": step.step_id,
            "op": step.op,
            "inputs": step.inputs,
            "result_var": step.result_var,
            "result_value": step.result_value,
            "depth_before": len(self.derived_steps)
        }

        # Check all inputs are defined
        for inp in step.inputs:
            if isinstance(inp, str) and inp not in current_vals:
                log_entry["accepted"] = False
                log_entry["error"] = "undefined_variable"
                self.log.append(log_entry)
                self.rejected_steps.append(step)
                return False, "undefined_variable"

        # Check that at least one derived variable is used (if any exist)
        derived_vars = {s.result_var for s in self.derived_steps}
        # Only enforce if there is at least 1 previously derived step
        if len(self.derived_steps) > 0:
            derived_vars = {s.result_var for s in self.derived_steps}
            if not any(inp in derived_vars for inp in step.inputs if isinstance(inp, str)):
                log_entry["accepted"] = False
                log_entry["error"] = "no_derived_variable_used"
                self.log.append(log_entry)
                self.rejected_steps.append(step)
                return False, "no_derived_variable_used"
        if derived_vars:
            if not any(inp in derived_vars for inp in step.inputs if isinstance(inp, str)):
                log_entry["accepted"] = False
                log_entry["error"] = "no_derived_variable_used"
                self.log.append(log_entry)
                self.rejected_steps.append(step)
                return False, "no_derived_variable_used"

        # Generate ASP program for arithmetic validation
        asp_program = values_to_asp(current_vals) + "\n" + step_to_asp(step)
        print("\n--- ASP PROGRAM ---")
        print(asp_program)

        # Check arithmetic correctness via Clingo
        if not clingo_check(asp_program):
            log_entry["accepted"] = False
            log_entry["error"] = "arithmetic_error"
            self.log.append(log_entry)
            self.rejected_steps.append(step)
            return False, "arithmetic_error"

        # Add step
        self.derived_steps.append(step)

        # Check structural validity incrementally
        if not check_incremental_trace(self):
            self.derived_steps.pop()
            log_entry["accepted"] = False
            log_entry["error"] = "structural_error"
            self.log.append(log_entry)
            self.rejected_steps.append(step)
            return False, "structural_error"

        # Update answer if this step produces the goal
        if step.result_var == self.goal:
            self.answer = step.result_value

        log_entry["accepted"] = True
        log_entry["error"] = None
        log_entry["depth_after"] = len(self.derived_steps)
        self.log.append(log_entry)

        return True, None
# -------------------------
# Build Prompt 
# -------------------------

def build_prompt(scratch: Scratchpad) -> str:
    values = scratch.current_values()

    value_lines = "\n".join(f"{k} = {v}" for k, v in values.items())

    return f"""#prompt template
You are a formal arithmetic reasoning engine.

STRICT RULES:
- Output ONLY valid JSON (no explanations, no markdown)
- Inputs MUST be variable names from Current values (no raw numbers)
- Each step MUST use at least one previously derived variable if any exist
- Do NOT recompute values that can be derived from existing variables
- Propose steps that move toward the goal: {scratch.goal}

Format:
{{
  "op": "add|subtract|multiply|divide",
  "inputs": [...],
  "result_var": "...",
  "result_value": integer
}}

Problem:
{scratch.problem}

Current values:
{value_lines}

Goal: {scratch.goal}

Propose the next step that moves toward the goal using derived variables whenever possible.
"""


# -------------------------
# PARSER
# -------------------------

## Parse llm output

#turning text into program -> first safety layer 
def parse_llm_output(output: str) -> DerivedStep:
    try:
        # extract JSON block if extra text exists
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found")

        json_str = match.group(0)
        data = json.loads(json_str)

    except Exception as e:
        raise ValueError(f"Invalid JSON: {output}")
    
    return DerivedStep(
        step_id="",  # placeholder
        op=data["op"],
        inputs=data["inputs"],
        relation="",
        result_var=data["result_var"],
        result_value=data["result_value"]
    )

# -------------------------
# Reasoning loop
# -------------------------
def run_llm_reasoning(scratch: Scratchpad, llm_call, max_steps=10):
    accepted_steps = 0
    total_attempts = 0

    while accepted_steps < max_steps:
        if scratch.answer is not None:
            break

        prompt = build_prompt(scratch)
        output = llm_call(prompt)
        print("\n--- LLM OUTPUT ---")
        print(output)

        try:
            step = parse_llm_output(output)
        except Exception as e:
            print("\n--- PARSE ERROR ---", str(e))
            total_attempts += 1
            continue

        # Assign step_id if missing
        if not hasattr(step, "step_id") or not step.step_id:
            step.step_id = f"s{len(scratch.derived_steps) + 1}"
            print(f"[DEBUG] Assigned step_id: {step.step_id}")

        accepted, error = scratch.apply_step(step)
        print(f"Step {step.step_id} accepted? {accepted}, error={error}")

        total_attempts += 1
        if accepted:
            accepted_steps += 1
        else:
            # Retry immediately without incrementing accepted_steps
            continue

        # safety break if too many attempts
        if total_attempts > max_steps * 2:
            print("Too many failed attempts, stopping loop.")
            break

    return scratch.log


#converts reasoning trace to graph structure 
def trace_to_asp(scratch: Scratchpad) -> str:
    lines = []

    # Encode initial facts
    for var in scratch.facts.keys():
        lines.append(f"fact({var}).")

    # Goal fact
    lines.append(f"goal({scratch.goal}).")

    for step in scratch.derived_steps:
        lines.append(f"step({step.step_id}).")
        lines.append(f"produces({step.step_id},{step.result_var}).")

        for inp in step.inputs:
            if isinstance(inp, str):
                lines.append(f"uses({step.step_id},{inp}).")

    return "\n".join(lines)

# structuring reasoning as a DAG (Directed Acyclic Graph).
STRUCTURAL_RULES = """
% A variable is derived if some step produces it
derived(V) :- produces(_,V).

% No overwriting initial facts
:- produces(_,V), fact(V).

% Direct dependency
depends(X,Y) :- produces(S,X), uses(S,Y).

% Transitive closure
reachable(X,Y) :- depends(X,Y).
reachable(X,Y) :- depends(X,Z), reachable(Z,Y).

% No cycles
:- reachable(V,V).

% Used variable
used(V) :- uses(_,V).
"""
# Final constraints
FINAL_STRUCTURAL_RULES = STRUCTURAL_RULES + """
% Goal must be derived
:- goal(G), not derived(G).

% No dead steps (unless goal)
:- produces(_,V), not used(V), not goal(V).
"""
#checks reasoning traces along the way -> applies structural rules
def check_incremental_trace(scratch: Scratchpad) -> bool:
    asp_program = trace_to_asp(scratch) + "\n" + STRUCTURAL_RULES
    return clingo_check(asp_program)

#checks full trace -> applies final structural rules 
def validate_trace(scratch: Scratchpad, debug=True) -> bool:
    asp_program = trace_to_asp(scratch) + "\n" + FINAL_STRUCTURAL_RULES
    result = clingo_check(asp_program)

    if debug:
        print("\n--- FINAL TRACE VALIDATION ---")
        print("Derived steps:", [s.step_id for s in scratch.derived_steps])
        print("Goal:", scratch.goal)
        print("Answer:", scratch.answer)
        print("Final structural valid?", result)

    return result
# -------------------------
# Test Case
# -------------------------

@dataclass
class TestCase:
    name: str
    problem: str                 
    facts: Dict[str, int]
    goal: str
    expected_answer: int
    steps: List[DerivedStep] = None  


# -------------------------
########################### 2. EXPERIMENTAL CONDITIONS
# -------------------------

# Baseline condition -> standard LLM prompting (no structure, constraints, verification)

def run_baseline(test_case: TestCase, llm_call):
    prompt = f"""
Solve the following arithmetic word problem.
Show your reasoning and give the final answer.

Problem:
{test_case.problem}
"""

    output = llm_call(prompt)
    print("\n--- BASELINE OUTPUT ---")
    print(output)

    return {
        "condition": "baseline",
        "raw_output": output
    }

# Standard COT condition -> self-refinement loop w/ COT (prompted to critique self, fix mistakes, improve reasoning, NO external verification)
def run_iterative_cot(test_case: TestCase, llm_call, max_rounds=3):
    prompt = f"""
Solve the following problem step-by-step.

Problem:
{test_case.problem}
"""

    reasoning = llm_call(prompt)

    for i in range(max_rounds - 1):
        refine_prompt = f"""
Here is your reasoning:

{reasoning}

Check for arithmetic or logical errors and correct them.
Give improved reasoning and final answer.
"""
        reasoning = llm_call(refine_prompt)

    print("\n--- ITERATIVE COT OUTPUT ---")
    print(reasoning)
    return {
        "condition": "iterative_cot",
        "final_output": reasoning
    }
# Hybrid (Neuro-symbolic) scratchpad system -> LLM + ASP verification loop
def run_hybrid(test_case: TestCase, llm_call, max_steps=5):
    scratch = Scratchpad(
        problem=test_case.problem,
        facts=test_case.facts,
        goal=test_case.goal
    )

    log = run_llm_reasoning(scratch, llm_call, max_steps=max_steps)

    final_valid = validate_trace(scratch, debug=False)

    return {
    "condition": "hybrid",
    "answer": scratch.answer,
    "expected": test_case.expected_answer,
    "trace_valid": final_valid,
    "log": scratch.log,
    "rejections": len(scratch.rejected_steps),
    "steps_taken": len(scratch.derived_steps)   
}

# Experiment runner
def extract_number(text):
    if text is None:
        return None
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None

# -------------------------
########################### 4. LLM INTERFACE
# -------------------------

#LLM interface 
def llm_call(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  #justifcation: 
        #model="gpt-4.1", #justifcation:
        messages=[
            {"role": "system", "content": "You are a formal arithmetic reasoning engine."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        #response_format={"type": "json_object"} #removed so as to not force baseline + COT into json format
    )

    return response.choices[0].message.content

def run_experiment(test_case: TestCase, llm_call):
    print("\n--- MODEL 1: gpt-4.1-mini  ---")
    print("\n--- BASELINE ---")
    baseline = run_baseline(test_case, llm_call)

    print("\n--- ITERATIVE COT ---")
    iterative = run_iterative_cot(test_case, llm_call)

    print("\n--- HYBRID ---")
    hybrid = run_hybrid(test_case, llm_call)

    # Extract answers
    baseline_answer = extract_number(baseline["raw_output"])
    iterative_answer = extract_number(iterative["final_output"])
    hybrid_answer = hybrid["answer"]

    return {
        "baseline": {
            "answer": baseline_answer,
            "correct": baseline_answer == test_case.expected_answer
        },
        "iterative": {
            "answer": iterative_answer,
            "correct": iterative_answer == test_case.expected_answer
        },
        "hybrid": {
            "answer": hybrid_answer,
            "correct": hybrid_answer == test_case.expected_answer,
            "trace_valid": hybrid["trace_valid"],
            "steps": hybrid["steps_taken"],
            "rejections": hybrid["rejections"]
        }
    }
# -------------------------
########################### 5. TEST CASES
# -------------------------
'''
test_cases = [
    TestCase(
        name="james_letters",
        problem="James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        facts={"pages_per_letter": 3, "num_friends": 2, "letters_per_week": 2, "weeks_per_year": 52},
        goal="total_pages",
        expected_answer=624,
        steps=[
            DerivedStep("s1", "multiply", ["pages_per_letter", "num_friends"], "", "pages_per_session", 6),
            DerivedStep("s2", "multiply", ["pages_per_session", "letters_per_week"], "", "pages_per_week", 12),
            DerivedStep("s3", "multiply", ["pages_per_week", "weeks_per_year"], "", "total_pages", 624),
        ]
    )
]
'''

test_cases = [
    TestCase(
        name="julie_book",
        problem="Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        facts={"total_pages": 120, "pages_yesterday": 12, "const_2": 2},
        goal="half_remaining",
        expected_answer=42,
        steps=[
            DerivedStep("s1", "multiply", ["pages_yesterday", "const_2"], "", "pages_today", 24),
            DerivedStep("s2", "add", ["pages_yesterday", "pages_today"], "", "pages_read", 36),
            DerivedStep("s3", "subtract", ["total_pages", "pages_read"], "", "remaining_pages", 84),
            DerivedStep("s4", "divide", ["remaining_pages", "const_2"], "", "half_remaining", 42),
        ]
    ),
    TestCase(
        name="ken_package",
        problem="Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. Finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?",
        facts={"initial_weight": 0, "first_jellybeans": 2, "triple_multiplier": 3, "second_jellybeans": 2, "double_multiplier": 2},
        goal="final_weight",
        expected_answer=16,
        steps=[
            DerivedStep("s1", "add", ["initial_weight", "first_jellybeans"], "", "after_first_jb", 2),
            DerivedStep("s2", "multiply", ["after_first_jb", "triple_multiplier"], "", "after_brownies", 6),
            DerivedStep("s3", "add", ["after_brownies", "second_jellybeans"], "", "after_second_jb", 8),
            DerivedStep("s4", "multiply", ["after_second_jb", "double_multiplier"], "", "final_weight", 16),
        ]
    ),

    TestCase(
        name="natalia_clips",
        problem="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        facts={"april_clips": 48, "const_half": 2},
        goal="total_clips",
        expected_answer=72,
        steps=[
            DerivedStep("s1", "divide", ["april_clips", "const_half"], "", "may_clips", 24),
            DerivedStep("s2", "add", ["april_clips", "may_clips"], "", "total_clips", 72),
        ]   
    ),
    TestCase(
        name="betty_wallet",
        problem="Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        facts={"wallet_cost": 100, "half_wallet": 50, "parents_contrib": 15, "grandparents_multiplier": 2},
        goal="money_needed",
        expected_answer=5,
        steps=[
            DerivedStep("s1", "multiply", ["parents_contrib", "grandparents_multiplier"], "", "grandparents_contrib", 30),
            DerivedStep("s2", "add", ["half_wallet", "parents_contrib"], "", "partial_total", 65),
            DerivedStep("s3", "add", ["partial_total", "grandparents_contrib"], "", "total_savings", 95),
            DerivedStep("s4", "subtract", ["wallet_cost", "total_savings"], "", "money_needed", 5),
        ]
    )
]

def compute_overall_success(all_results):
    summary = {
        "baseline": {"correct": 0, "total": 0, "accuracy": 0.0},
        "iterative": {"correct": 0, "total": 0, "accuracy": 0.0},
        "hybrid": {"correct": 0, "total": 0, "accuracy": 0.0},
    }

    for result in all_results:
        for condition in ["baseline", "iterative", "hybrid"]:
            if condition in result:
                summary[condition]["total"] += 1
                if result[condition].get("correct"):
                    summary[condition]["correct"] += 1

    # Compute accuracy
    for condition in summary.keys():
        if summary[condition]["total"] > 0:
            summary[condition]["accuracy"] = (
                summary[condition]["correct"] / summary[condition]["total"]
            )

    return summary

# -------------------------
########################### 6. RUN EXPERIMENTS 
# -------------------------

all_results = []

for tc in test_cases:
    print(f"\n===== {tc.name} =====")

    result = run_experiment(tc, llm_call)
    all_results.append(result)
    print(result)

overall_summary = compute_overall_success(all_results)
print("\n===== OVERALL SUCCESS RATE =====")
for cond, stats in overall_summary.items():
    print(f"{cond}: {stats['correct']}/{stats['total']} correct → accuracy={stats['accuracy']:.2%}")
