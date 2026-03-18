"""
Prompt templates for SQL grading.
Contains the actual rubrics and instructions used for grading.

Calibration notes (v1.2):
  - Added per-question deduction caps to prevent over-penalizing
  - Softened rubric language: deductions are MAXIMUMS, not automatic stacks
  - Expanded calibration examples to better reflect human grader leniency
  - Partial credit guidance added for near-correct queries

v1.3 o4-mini compatibility:
  - Added create_system_prompt() for OpenAI system role (brief, not the full rubric)
  - Added create_grading_prompt_full() alias — identical to create_grading_prompt()
  - create_grading_prompt() is UNCHANGED from v1.2 — full rubric preserved
"""

from typing import Dict


def create_system_prompt() -> str:
    """
    Return a minimal system role message for OpenAI API (o4-mini, gpt-4o, etc).
    The full grading rubric and calibration examples live in create_grading_prompt().
    For o4-mini: passed as role='system'. For Gemini: not used separately.
    """
    return (
        "You are an expert SQL grader. Grade student SQL queries with the same leniency "
        "as a human TA. Follow all rubric instructions exactly. Return only valid JSON."
    )


def create_grading_prompt_full(student_queries: Dict[str, str]) -> str:
    """
    Full combined prompt — identical to create_grading_prompt().
    Used by Gemini (single-turn API with no separate system message).
    """
    return create_grading_prompt(student_queries)


def create_grading_prompt(student_queries: Dict[str, str]) -> str:
    """
    Create a comprehensive grading prompt for all SQL questions.

    Args:
        student_queries: Dictionary mapping question numbers to SQL queries

    Returns:
        str: Formatted grading prompt for the LLM
    """

    prompt = f"""You are an expert SQL grader, grading SQL queries for 5 questions.
Each question is worth 10 points. You must grade with the SAME LENIENCY as a human TA.

================================================================================
CRITICAL GRADING PHILOSOPHY — READ CAREFULLY
================================================================================

1. FUNCTIONAL EQUIVALENCE WINS: If the query produces the correct result for any
   reasonable dataset, award 10/10 immediately. Accept all equivalent SQL styles:
   JOIN vs comma-join, aliases, subqueries, CTEs, different column orderings.

2. DEDUCTIONS ARE CAPS, NOT STACKS: Each listed deduction is a MAXIMUM penalty
   for that type of error. Do NOT add multiple deductions for the same underlying
   mistake. Pick the ONE most relevant deduction.

3. PARTIAL CREDIT IS THE NORM: A query that shows correct understanding but has
   one missing piece should score 7-8. Only penalize what is clearly wrong.

4. NEVER GO BELOW THESE FLOORS UNLESS THE ANSWER IS EMPTY OR COMPLETELY RANDOM:
   - Shows some correct tables/columns: minimum 3 points
   - Shows correct overall approach with one error: minimum 6 points
   - Nearly correct (minor issue): minimum 7 points

5. MAXIMUM TOTAL DEDUCTION PER QUESTION: 8 points (minimum score 2 unless empty)
   Exception: completely nonsensical or empty answer = 0.

6. DO NOT penalize for:
   - Case differences (SELECT vs select)
   - Extra whitespace or formatting
   - Semicolons present or absent
   - Table aliases that differ from the sample answer
   - DISTINCT where not strictly required but not harmful
   - Attribute name capitalization

DATABASE SCHEMA:
PART (P_PARTKEY, P_NAME, P_BRAND, P_TYPE, P_SIZE) key:P_PARTKEY
PARTSUPP (PS_PARTKEY, PS_SUPPKEY, PS_AVAILQTY, PS_SUPPLYCOST) key:PS_PARTKEY, PS_SUPPKEY
SUPPLIER (S_SUPPKEY, S_NAME, S_ADDRESS, S_PHONE, S_ACCTBAL) key:S_SUPPKEY

================================================================================
QUESTION 4.1:
How many quantities of the part named 'blush thistle blue yellow saddle'
are available in total?
================================================================================
STUDENT ANSWER: {student_queries['4.1']}

CORRECT ANSWERS (any equivalent):
1. SELECT SUM(PS_AVAILQTY)
   FROM PART P JOIN PARTSUPP PS ON P.P_PARTKEY = PS.PS_PARTKEY
   WHERE P.P_NAME = 'blush thistle blue yellow saddle';
2. SELECT SUM(ps_availqty) FROM Part, PartSupp WHERE ps_partkey = p_partkey AND p_name = 'blush thistle blue yellow saddle';

SPECIFIC DEDUCTIONS - APPLY ONLY THE MOST RELEVANT ONE PER ERROR TYPE:
(Start at 10. Total deductions capped at 8.)

Core Logic (pick ONE per issue, do not stack):
- Missing SUM entirely and using COUNT instead: -2
- SUM of completely wrong attribute (e.g., ps_partkey): -3
- Missing join between PART and PARTSUPP: -3 (max; reduce if partial join attempt)
- Missing PARTSUPP or PART in FROM: -2
- Missing WHERE filter on p_name: -2
- Correct logic but unnecessary GROUP BY that doesn't break result: -1

Syntax/Minor:
- SUM without parentheses (SUM ps_availqty): -1
- Minor typo not affecting logic: -1

Severe (use ONLY if query is essentially random/empty):
- Empty answer: 0 points
- Completely incorrect logic with no salvageable elements: 2 points max

CALIBRATION EXAMPLES FOR Q4.1:
- Full SUM with correct join and filter -> 10
- Correct logic but GROUP BY ps_partkey added unnecessarily -> 8
- SUM(ps_availqty) but missing PART table, uses only PARTSUPP with wrong filter -> 6
- Uses COUNT instead of SUM, otherwise correct -> 8
- Has SUM and filter but missing the join condition (tables listed) -> 7
- Totally wrong (e.g., SELECT P_NAME FROM PART) -> 2

================================================================================
QUESTION 4.2:
Return the distinct partkeys of parts that are supplied by suppliers
whose account balance (S_ACCTBAL) is greater than 1000.
================================================================================
STUDENT ANSWER: {student_queries['4.2']}

CORRECT ANSWERS:
1. SELECT DISTINCT PS_PARTKEY
   FROM PARTSUPP PS JOIN SUPPLIER S ON PS.PS_SUPPKEY = S.S_SUPPKEY
   WHERE S.S_ACCTBAL > 1000;
2. SELECT DISTINCT ps_partkey FROM PartSupp, Supplier WHERE ps_suppkey = s_suppkey AND s_acctbal > 1000

SPECIFIC DEDUCTIONS - APPLY ONLY THE MOST RELEVANT ONE PER ERROR TYPE:
(Start at 10. Total deductions capped at 8.)

Core Logic:
- Missing DISTINCT: -2 (only if result would have duplicates)
- Missing join predicate ps_suppkey = s_suppkey: -3
- Missing PARTSUPP in FROM: -2
- Wrong projection (returns s_suppkey instead of ps_partkey): -3
- Missing filter s_acctbal > 1000: -2

Incorrect Patterns (pick ONE):
- Unnecessary GROUP BY that still returns correct columns: -1
- Self-join on same table unnecessarily: -4
- HAVING max(s_acctbal) > 1000 instead of WHERE: -2

Severe:
- No answer: 0 points
- Completely wrong tables/logic: 2 points max

CALIBRATION EXAMPLES FOR Q4.2:
- SELECT DISTINCT PS_PARTKEY with correct join and filter -> 10
- Correct query but missing DISTINCT -> 8
- Correct tables and filter but wrong projected column (S_SUPPKEY) -> 7
- Missing join condition but correct tables and filter -> 7
- Only one table in FROM, missing join entirely -> 5
- Totally wrong -> 2

================================================================================
QUESTION 4.3:
Return the number of distinct parts that are supplied by each supplier.
Output: S_NAME, COUNT
================================================================================
STUDENT ANSWER: {student_queries['4.3']}

CORRECT ANSWERS:
1. SELECT S.S_NAME, COUNT(DISTINCT PS.PS_PARTKEY)
   FROM SUPPLIER S JOIN PARTSUPP PS ON S.S_SUPPKEY = PS.PS_SUPPKEY
   GROUP BY S.S_NAME;
2. SELECT S_NAME, COUNT(*)
   FROM PARTSUPP, SUPPLIER
   WHERE PARTSUPP.PS_SUPPKEY = SUPPLIER.S_SUPPKEY
   GROUP BY S_NAME

SPECIFIC DEDUCTIONS - APPLY ONLY THE MOST RELEVANT ONE PER ERROR TYPE:
(Start at 10. Total deductions capped at 8.)

Core Logic:
- Missing GROUP BY entirely: -3
- Grouping by s_suppkey instead of s_name (minor semantic issue): -1
- Missing COUNT (no aggregation at all): -2
- Missing s_name in SELECT: -2
- Missing join between SUPPLIER and PARTSUPP: -3
- Missing join predicate ps_suppkey = s_suppkey: -2

Aggregation (pick ONE):
- DISTINCT(COUNT(...)) syntax instead of COUNT(DISTINCT ...): -1 (intent is clear)
- SUM used instead of COUNT: -3
- Missing DISTINCT in COUNT when question asks for distinct: -1

Syntax:
- Minor typo (missing comma, capitalization): -1

Severe:
- Empty answer: 0 points
- No aggregation, no join, no grouping: 2 points max

CALIBRATION EXAMPLES FOR Q4.3:
- S_NAME + COUNT(DISTINCT ps_partkey) + correct join + GROUP BY s_name -> 10
- COUNT(*) instead of COUNT(DISTINCT ps_partkey), otherwise correct -> 9
- Correct query but GROUP BY s_suppkey instead of s_name -> 8
- Correct join and GROUP BY but missing DISTINCT in COUNT -> 8
- Has GROUP BY and COUNT but missing S_NAME in SELECT -> 7
- Missing GROUP BY but has join and COUNT -> 6
- Only one table, no join, some aggregation -> 3

================================================================================
QUESTION 4.4:
Return the maximal number of distinct parts that are supplied by any supplier.
(Nested queries required)
================================================================================
STUDENT ANSWER: {student_queries['4.4']}

CORRECT ANSWERS:
1. SELECT MAX(part_count)
   FROM (
        SELECT COUNT(DISTINCT PS_PARTKEY) AS part_count
        FROM PARTSUPP
        GROUP BY PS_SUPPKEY
   ) t;
2. WITH PartCounts(ps_suppkey, cnt) AS
    (
    SELECT ps_suppkey, COUNT(DISTINCT ps_partkey) as cnt
    FROM PartSupp
    GROUP BY ps_suppkey
    )
    SELECT MAX(cnt) as maxDistinctParts
    FROM PartCounts

SPECIFIC DEDUCTIONS - APPLY ONLY THE MOST RELEVANT ONE PER ERROR TYPE:
(Start at 10. Total deductions capped at 8.)

NOTE: This is the hardest question. Be generous with partial credit for students
who demonstrate they understand the two-step approach (count per supplier, then max).

Core Requirements (each is a separate concern):
- Missing nested query or CTE (flat query): -3
- Missing COUNT(DISTINCT ps_partkey) - using COUNT(*) is acceptable: -1
- Missing GROUP BY ps_suppkey in inner query: -3
- Missing MAX in outer query: -3

Logic Errors (pick ONE):
- Using SUM instead of COUNT for counting parts: -3
- Returning max(ps_availqty) instead of counting distinct parts: -2
- MAX of wrong attribute (ps_partkey, p_name, etc.): -2

Overcomplication:
- Correct logic but unnecessarily complex (extra joins, etc.): -1

Syntax:
- MAX(subquery) directly (non-standard): -2
- Minor syntax errors that don't change logic: -1

Severe:
- Empty or meaningless answer: 0 points
- Completely wrong with no two-step logic visible: 2 points max

CALIBRATION EXAMPLES FOR Q4.4:
- Correct nested query with MAX(COUNT(DISTINCT)) and GROUP BY -> 10
- CTE approach with correct logic -> 10
- Correct inner query (COUNT + GROUP BY) but MAX is missing in outer -> 7
- Has MAX and nested query but COUNT(*) instead of COUNT(DISTINCT) -> 9
- Has nested query and MAX but missing GROUP BY ps_suppkey -> 7
- Flat query with just MAX(ps_availqty) - no nesting -> 4
- Only outer SELECT MAX with no inner grouping logic -> 3
- Completely wrong -> 2

================================================================================
QUESTION 4.5:
Return the keys of suppliers who have supplied at least two different parts.
================================================================================
STUDENT ANSWER: {student_queries['4.5']}

CORRECT ANSWERS:
1. SELECT PS_SUPPKEY FROM PARTSUPP GROUP BY PS_SUPPKEY HAVING COUNT(DISTINCT PS_PARTKEY) >= 2;
2. SELECT DISTINCT ps1.ps_suppkey FROM PartSupp ps1, PartSupp ps2
   WHERE ps1.ps_suppkey = ps2.ps_suppkey AND ps1.ps_partkey != ps2.ps_partkey

SPECIFIC DEDUCTIONS - APPLY ONLY THE MOST RELEVANT ONE PER ERROR TYPE:
(Start at 10. Total deductions capped at 8.)

Core Logic (GROUP BY / HAVING approach):
- Missing HAVING or wrong HAVING condition: -3
- Missing GROUP BY ps_suppkey: -3
- HAVING COUNT = 2 instead of >= 2: -2
- Returns wrong key (ps_partkey instead of ps_suppkey): -3

Core Logic (Self-join approach):
- Missing self-join on PARTSUPP: -5
- Missing ps1.ps_partkey != ps2.ps_partkey: -3
- Missing ps1.ps_suppkey = ps2.ps_suppkey: -3

Aggregation Misuse (pick ONE):
- COUNT without GROUP BY: -3
- HAVING without GROUP BY: -2

Severe:
- No answer: 0 points
- Completely incorrect logic (no grouping, no self-join, wrong tables): 2 points max

CALIBRATION EXAMPLES FOR Q4.5:
- GROUP BY ps_suppkey HAVING COUNT(DISTINCT ps_partkey) >= 2 -> 10
- Self-join with both conditions -> 10
- GROUP BY ps_suppkey HAVING COUNT(*) >= 2 (no DISTINCT) -> 9
- HAVING COUNT >= 2 but missing GROUP BY -> 6
- GROUP BY with HAVING COUNT = 2 (not >=) -> 8
- Correct grouping but returns ps_partkey instead of ps_suppkey -> 7
- No grouping at all, just WHERE clause -> 3

================================================================================
CALIBRATED SCORING REFERENCE (align your scores to these human grader benchmarks)
================================================================================

These examples show what HUMAN graders actually gave. Match this leniency:

Question 4.1 (Score: 10)
SELECT SUM(PS_AVAILQTY) FROM PART, PARTSUPP
WHERE P_PARTKEY = PS_PARTKEY AND P_NAME = 'blush thistle blue yellow saddle'
Why 10? Functionally equivalent. Comma-join is acceptable.

Question 4.1 (Score: 8)
SELECT SUM(PS_AVAILQTY) FROM PARTSUPP
WHERE PS_PARTKEY IN (SELECT P_PARTKEY FROM PART WHERE P_NAME = 'blush thistle blue yellow saddle')
GROUP BY PS_PARTKEY
Why 8? Unnecessary GROUP BY but result is still the total sum per key. Minor overcomplication.

Question 4.2 (Score: 10)
SELECT DISTINCT ps_partkey FROM partsupp, supplier
WHERE ps_suppkey = s_suppkey AND s_acctbal > 1000
Why 10? Functionally correct. Comma-join style accepted.

Question 4.2 (Score: 8)
SELECT ps_partkey FROM partsupp JOIN supplier ON ps_suppkey = s_suppkey
WHERE s_acctbal > 1000
Why 8? Missing DISTINCT only.

Question 4.3 (Score: 9)
SELECT S_NAME, COUNT(*) FROM SUPPLIER, PARTSUPP
WHERE S_SUPPKEY = PS_SUPPKEY GROUP BY S_NAME
Why 9? COUNT(*) instead of COUNT(DISTINCT ps_partkey). Minor issue since each row is a unique part supply.

Question 4.4 (Score: 10)
SELECT MAX(cnt) FROM
  (SELECT PS_SUPPKEY, COUNT(*) AS cnt FROM PARTSUPP GROUP BY PS_SUPPKEY) t
Why 10? COUNT(*) here is equivalent since each row is a unique (suppkey, partkey) combination.

Question 4.4 (Score: 7)
SELECT MAX(COUNT(DISTINCT PS_PARTKEY)) FROM PARTSUPP GROUP BY PS_SUPPKEY
Why 7? Correct intent but MAX(aggregate) is non-standard SQL. Shows two-step understanding.

Question 4.5 (Score: 9)
SELECT PS_SUPPKEY FROM PARTSUPP GROUP BY PS_SUPPKEY HAVING COUNT(*) >= 2
Why 9? COUNT(*) acceptable if (PS_SUPPKEY, PS_PARTKEY) pairs are unique. Only -1 for missing DISTINCT.

Question 4.1 (Score: 5)
SELECT P_NAME, SUM(PS_AVAILQTY) FROM PART, PARTSUPP
WHERE P_PARTKEY = PS_PARTKEY GROUP BY P_NAME
Why 5? Has SUM and join but groups by P_NAME instead of filtering. Returns all parts not just the named one.

Question 4.2 (Score: 5)
SELECT P_PARTKEY FROM Suppliers WHERE S_ACCTBAL > 1000
Why 5? Missing join to PARTSUPP, missing DISTINCT, only partially correct logic.

Question 4.3 (Score: 2)
SELECT S_NAME FROM Supplier WHERE COUNT(PART)
Why 2? Invalid aggregation, missing GROUP BY, incorrect SQL structure.

Question 4.4 (Score: 2)
SELECT S_NAME FROM Supplier WHERE SIZE=MAX()
Why 2? No nested query, incorrect aggregation logic, completely incorrect structure.

Question 4.5 (Score: 2)
SELECT S_SUPPKEY FROM Supplier WHERE P_PARTKEY = 2
Why 2? No grouping, no comparison of two parts, completely incorrect logic.

----------------------------------------------------------------------

IMPORTANT: Your scores must reflect the same leniency shown above.
When in doubt between two scores, choose the HIGHER one.

================================================================================
OUTPUT FORMAT
================================================================================
Return ONLY a JSON object (no markdown, no extra text):
{{
    "question_4_1": {{
        "score": <0-10>,
        "deduction_details": "<details or 'Full credit'>",
        "feedback": "<brief explanation>",
        "needs_review": <true/false>
    }},
    "question_4_2": {{
        "score": <0-10>,
        "deduction_details": "<details or 'Full credit'>",
        "feedback": "<brief explanation>",
        "needs_review": <true/false>
    }},
    "question_4_3": {{
        "score": <0-10>,
        "deduction_details": "<details or 'Full credit'>",
        "feedback": "<brief explanation>",
        "needs_review": <true/false>
    }},
    "question_4_4": {{
        "score": <0-10>,
        "deduction_details": "<details or 'Full credit'>",
        "feedback": "<brief explanation>",
        "needs_review": <true/false>
    }},
    "question_4_5": {{
        "score": <0-10>,
        "deduction_details": "<details or 'Full credit'>",
        "feedback": "<brief explanation>",
        "needs_review": <true/false>
    }}
}}"""
    return prompt


# Per-question rubric data used to build single-question prompts
_QUESTION_DATA = {
    '4.1': {
        'question': "How many quantities of the part named 'blush thistle blue yellow saddle' are available in total?",
        'correct_answers': """1. SELECT SUM(PS_AVAILQTY)
   FROM PART P JOIN PARTSUPP PS ON P.P_PARTKEY = PS.PS_PARTKEY
   WHERE P.P_NAME = 'blush thistle blue yellow saddle';
2. SELECT SUM(ps_availqty) FROM Part, PartSupp WHERE ps_partkey = p_partkey AND p_name = 'blush thistle blue yellow saddle';""",
        'rubric': """DEDUCTIONS ARE CAPS - pick the ONE most relevant deduction per issue. Do not stack.
Total deductions capped at 8. When in doubt, choose the higher score.

Core Logic:
- Missing SUM entirely, using COUNT instead: -2
- SUM of completely wrong attribute: -3
- Missing join between PART and PARTSUPP: -3 (reduce if partial attempt)
- Missing PARTSUPP or PART in FROM: -2
- Missing WHERE filter on p_name: -2
- Unnecessary GROUP BY that doesn't break result: -1

Severe (only if essentially random/empty):
- Empty answer: 0 points
- Completely wrong with no salvageable elements: 2 points

Calibration: COUNT instead of SUM but otherwise correct -> 8.
Missing join condition but tables present -> 7.""",
        'key': 'question_4_1',
    },
    '4.2': {
        'question': "Return the distinct partkeys of parts that are supplied by suppliers whose account balance (S_ACCTBAL) is greater than 1000.",
        'correct_answers': """1. SELECT DISTINCT PS_PARTKEY
   FROM PARTSUPP PS JOIN SUPPLIER S ON PS.PS_SUPPKEY = S.S_SUPPKEY
   WHERE S.S_ACCTBAL > 1000;
2. SELECT DISTINCT ps_partkey FROM PartSupp, Supplier WHERE ps_suppkey = s_suppkey AND s_acctbal > 1000""",
        'rubric': """DEDUCTIONS ARE CAPS - pick the ONE most relevant deduction per issue. Do not stack.
Total deductions capped at 8. When in doubt, choose the higher score.

Core Logic:
- Missing DISTINCT: -2
- Missing join predicate ps_suppkey = s_suppkey: -3
- Missing PARTSUPP in FROM: -2
- Wrong projection (returns s_suppkey instead of ps_partkey): -3
- Missing filter s_acctbal > 1000: -2

Incorrect Patterns: Unnecessary GROUP BY: -1, Self-join unnecessarily: -4

Severe: No answer: 0 points. Completely wrong: 2 points.

Calibration: Missing DISTINCT only -> 8. Wrong column projected -> 7.""",
        'key': 'question_4_2',
    },
    '4.3': {
        'question': "Return the number of distinct parts that are supplied by each supplier. Output: S_NAME, COUNT",
        'correct_answers': """1. SELECT S.S_NAME, COUNT(DISTINCT PS.PS_PARTKEY)
   FROM SUPPLIER S JOIN PARTSUPP PS ON S.S_SUPPKEY = PS.PS_SUPPKEY
   GROUP BY S.S_NAME;
2. SELECT S_NAME, COUNT(*) FROM PARTSUPP, SUPPLIER WHERE PARTSUPP.PS_SUPPKEY = SUPPLIER.S_SUPPKEY GROUP BY S_NAME""",
        'rubric': """DEDUCTIONS ARE CAPS - pick the ONE most relevant deduction per issue. Do not stack.
Total deductions capped at 8. When in doubt, choose the higher score.

Core Logic:
- Missing GROUP BY entirely: -3
- Grouping by s_suppkey instead of s_name: -1
- Missing COUNT (no aggregation): -2
- Missing s_name in SELECT: -2
- Missing join between SUPPLIER and PARTSUPP: -3
- Missing join predicate: -2

Aggregation: DISTINCT(COUNT) instead of COUNT(DISTINCT): -1, SUM instead of COUNT: -3
Severe: Empty: 0, No aggregation/no join/no grouping: 2 points.

Calibration: COUNT(*) instead of COUNT(DISTINCT) -> 9. Missing DISTINCT in COUNT -> 8.""",
        'key': 'question_4_3',
    },
    '4.4': {
        'question': "Return the maximal number of distinct parts that are supplied by any supplier. (Nested queries required)",
        'correct_answers': """1. SELECT MAX(part_count) FROM (
        SELECT COUNT(DISTINCT PS_PARTKEY) AS part_count FROM PARTSUPP GROUP BY PS_SUPPKEY) t;
2. WITH PartCounts AS (SELECT ps_suppkey, COUNT(DISTINCT ps_partkey) as cnt FROM PartSupp GROUP BY ps_suppkey)
   SELECT MAX(cnt) FROM PartCounts""",
        'rubric': """DEDUCTIONS ARE CAPS - pick the ONE most relevant deduction per issue. Do not stack.
Total deductions capped at 8. When in doubt, choose the higher score.

This is the hardest question. Be generous with partial credit for students who demonstrate
the two-step approach (count per supplier, then take max).

Core Requirements:
- Missing nested query/CTE entirely (flat query): -3
- COUNT(*) instead of COUNT(DISTINCT ps_partkey): -1 (minor; often equivalent)
- Missing GROUP BY ps_suppkey in inner query: -3
- Missing MAX in outer query: -3

Logic Errors (pick ONE): SUM instead of COUNT: -3, max(ps_availqty) instead of counting: -2
Syntax: MAX(subquery) directly: -2, Minor syntax: -1
Severe: Empty: 0. Completely wrong with no two-step logic: 2 points.

Calibration: COUNT(*) instead of COUNT(DISTINCT) -> 9. Missing GROUP BY only -> 7.
MAX without nesting -> 3-4.""",
        'key': 'question_4_4',
    },
    '4.5': {
        'question': "Return the keys of suppliers who have supplied at least two different parts.",
        'correct_answers': """1. SELECT PS_SUPPKEY FROM PARTSUPP GROUP BY PS_SUPPKEY HAVING COUNT(DISTINCT PS_PARTKEY) >= 2;
2. SELECT DISTINCT ps1.ps_suppkey FROM PartSupp ps1, PartSupp ps2
   WHERE ps1.ps_suppkey = ps2.ps_suppkey AND ps1.ps_partkey != ps2.ps_partkey""",
        'rubric': """DEDUCTIONS ARE CAPS - pick the ONE most relevant deduction per issue. Do not stack.
Total deductions capped at 8. When in doubt, choose the higher score.

GROUP BY / HAVING approach:
- Missing HAVING or wrong HAVING condition: -3
- Missing GROUP BY ps_suppkey: -3
- HAVING COUNT = 2 instead of >= 2: -2
- Returns wrong key (ps_partkey instead of ps_suppkey): -3

Self-join approach:
- Missing self-join on PARTSUPP: -5
- Missing ps1.ps_partkey != ps2.ps_partkey: -3
- Missing ps1.ps_suppkey = ps2.ps_suppkey: -3

Aggregation: COUNT without GROUP BY: -3, HAVING without GROUP BY: -2
Severe: No answer: 0, No grouping/join/wrong tables: 2 points.

Calibration: COUNT(*) instead of COUNT(DISTINCT) -> 9. HAVING COUNT = 2 -> 8. Missing GROUP BY -> 6.""",
        'key': 'question_4_5',
    },
}


def create_single_question_prompt(q_num: str, student_query: str) -> str:
    """
    Create a grading prompt for a single SQL question.
    Used for models that struggle to produce all 5 answers in one response (e.g. mistral).

    Args:
        q_num: Question number string, e.g. '4.1'
        student_query: The student's SQL query for this question

    Returns:
        str: Formatted grading prompt for the single question
    """
    qd = _QUESTION_DATA[q_num]
    key = qd['key']

    return f"""You are an expert SQL grader. Grade the following SQL query for ONE question.
The question is worth 10 points.

DATABASE SCHEMA:
PART (P_PARTKEY, P_NAME, P_BRAND, P_TYPE, P_SIZE) key:P_PARTKEY
PARTSUPP (PS_PARTKEY, PS_SUPPKEY, PS_AVAILQTY, PS_SUPPLYCOST) key:PS_PARTKEY, PS_SUPPKEY
SUPPLIER (S_SUPPKEY, S_NAME, S_ADDRESS, S_PHONE, S_ACCTBAL) key:S_SUPPKEY

GRADING PHILOSOPHY - READ CAREFULLY:
1. If the query is functionally equivalent to a correct answer, give 10/10 immediately.
2. Deductions are CAPS, not automatic stacks. Pick the ONE most relevant deduction per issue.
3. When in doubt between two scores, choose the HIGHER one.
4. Only go below 3 if the answer is empty or completely random/nonsensical.
5. Total deductions capped at 8 (minimum score 2, unless empty = 0).

QUESTION {q_num}: {qd['question']}

STUDENT ANSWER:
{student_query}

CORRECT ANSWERS (any equivalent accepted):
{qd['correct_answers']}

RUBRIC DEDUCTIONS:
{qd['rubric']}

Return ONLY this JSON (no markdown, no extra text):
{{
    "{key}": {{
        "score": <0-10>,
        "deduction_details": "<list each applied deduction with points, or 'Full credit'>",
        "feedback": "<brief explanation>",
        "needs_review": <true/false>
    }}
}}"""