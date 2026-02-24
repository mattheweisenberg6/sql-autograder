"""
Prompt templates for SQL grading.
Contains the actual rubrics and instructions used for grading.
"""

from typing import Dict


def create_grading_prompt(student_queries: Dict[str, str]) -> str:
    """
    Create a comprehensive grading prompt for all SQL questions.

    Args:
        student_queries: Dictionary mapping question numbers to SQL queries

    Returns:
        str: Formatted grading prompt for the LLM
    """

    prompt = f"""You are an expert SQL grader, grading SQL queries for 5 questions. 
Each question is worth 10 points. Follow these rules consistently for ALL questions:

GENERAL GRADING INSTRUCTIONS:
1. ALWAYS check functional equivalence FIRST
   - If the query produces the correct logical result, give 10 points (full credit) immediately
   - Accept different syntax: JOIN vs comma-separated tables, subqueries, aliases
2. ONLY apply deductions if the query is NOT functionally equivalent
3. Be LENIENT with syntax if intent and logic are correct
4. Focus on correctness of results, not formatting

UNIVERSAL DEDUCTIONS:
- Empty answer or "[NO ANSWER PROVIDED]": 0 points
- Completely incorrect approach: max 2–4 points

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

SPECIFIC DEDUCTIONS (Q4.1 CANVAS RUBRIC ALIGNMENT):

Core Logic Errors:
- Missing SUM(ps_availqty): -3
- Using COUNT instead of SUM: -2
- SUM(*) instead of SUM(ps_availqty): -2
- SUM of wrong attribute (e.g., ps_partkey, supplier.s_suppkey, p_name): -3
- Missing join PART.p_partkey = PARTSUPP.ps_partkey: -3
- Missing PARTSUPP in FROM: -2
- Missing PART in FROM: -2
- Missing filter p_name = 'blush thistle blue yellow saddle': -2
- Comparing P_PARTKEY to string instead of P_NAME: -2

GroupBy Misuse:
- Unnecessary GROUP BY: -4
- Missing SUM due to misuse of GROUP BY: -3
- Subquery with unnecessary GROUP BY: -3

Overcomplication:
- Unnecessary nested query: -3
- Overly complicated but partially correct logic: -2 to -4

Syntax Errors:
- Missing SELECT: -2
- SUM ps_availqty instead of SUM(ps_availqty): -1
- Using && instead of AND: -2
- Missing AND between predicates: -2
- SUM(m1) or SUM(subquery): -3
- SELECT P_PARTKEY, SUM(P_NAME): -4
- Minor typo (e.g., ^ instead of AND): -1

Severe:
- Empty answer: 0 points
- Completely incorrect logic: -8 to -10

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

SPECIFIC DEDUCTIONS (Q4.2 CANVAS RUBRIC ALIGNMENT):

Core Logic Errors:
- Missing DISTINCT: -2
- Missing ps_suppkey = s_suppkey join: -3
- Missing PARTSUPP in FROM: -2
- Wrong projection (e.g., SELECT s_suppkey): -3
- Missing filter s_acctbal > 1000: -2

Incorrect Logic Patterns:
- Using GROUP BY unnecessarily: -2
- Using SELF JOIN unnecessarily: -4
- HAVING max(s_acctbal) > 1000: -2
- Overly complicated logic: -4

Syntax Errors:
- Wrong JOIN syntax (missing ON): -2
- Using = instead of IN for subquery: -2

Severe:
- No answer: 0 points   

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

SPECIFIC DEDUCTIONS (Q4.3 CANVAS RUBRIC ALIGNMENT):

Core Logic Errors:
- Missing GROUP BY s_name: -3
- Grouping by s_suppkey instead of s_name: -2
- Missing COUNT(*) or COUNT(DISTINCT ...): -2
- Missing s_name in SELECT: -2
- Missing join between SUPPLIER and PARTSUPP: -3
- Missing join predicate ps_suppkey = s_suppkey: -3
- Significant logic error (not counting per supplier): -5

Aggregation Errors:
- DISTINCT(COUNT(...)) instead of COUNT(DISTINCT ...): -2
- SUM(ps_availqty as COUNT): -3
- Missing DISTINCT when needed: -1

Syntax Errors:
- Missing FROM: -2
- Missing AND in WHERE: -2
- JOIN syntax incorrect: -2
- Minor typo (missing comma etc.): -1

Severe:
- Empty answer: 0 points
- Incorrect/incomplete: -8

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

SPECIFIC DEDUCTIONS (Q4.4 CANVAS RUBRIC ALIGNMENT):

Core Requirements:
- Missing nested query: -3
- Missing COUNT(DISTINCT ps_partkey): -3
- Missing GROUP BY ps_suppkey: -3
- Missing MAX in outer query: -3

Logic Errors:
- Using SUM instead of COUNT: -3
- max(ps_availqty) instead of counting parts: -2
- Wrong attribute (e.g., max(p_name), max(ps_partkey)): -3
- Not returning maximal number: -4

Overcomplication:
- Overly complicated but partially correct: -2
- Unnecessary join complexity: -2

Syntax Errors:
- MAX(subquery) incorrect syntax: -3
- DISTINCT COUNT(*) syntax error: -2
- GROUP BY wrong attribute (e.g., SID, PID): -2
- Missing join predicate: -3
- Minor syntax errors: -2

Severe:
- Empty or meaningless: 0 points
- Most logic incorrect: -8 to -10

================================================================================
QUESTION 4.5:
Return the keys of suppliers who have supplied at least two different parts.
================================================================================
STUDENT ANSWER: {student_queries['4.5']}

CORRECT ANSWERS:
1. SELECT PS_SUPPKEY
   FROM PARTSUPP
   GROUP BY PS_SUPPKEY
   HAVING COUNT(DISTINCT PS_PARTKEY) >= 2;
2. SELECT DISTINCT ps1.ps_suppkey
   FROM PartSupp ps1, PartSupp ps2
   WHERE ps1.ps_suppkey = ps2.ps_suppkey
   AND ps1.ps_partkey != ps2.ps_partkey

SPECIFIC DEDUCTIONS (Q4.5 CANVAS RUBRIC ALIGNMENT):

Core Logic Errors:
- Missing self join on PARTSUPP: -5
- Missing ps1.ps_partkey != ps2.ps_partkey: -3
- Missing ps1.ps_suppkey = ps2.ps_suppkey: -3
- Wrong logic ps_partkey = ps_partkey: -3
- Using GROUP BY incorrectly: -2
- HAVING COUNT(ps_partkey) = 2 instead of >= 2: -2
- Wrong projection (not returning supplier key): -1

Aggregation Misuse:
- COUNT(distinct ps_partkey) without grouping: -7
- HAVING without GROUP BY: -4

Syntax Errors:
- Missing alias ps1 or ps2: -2
- count(nested query) syntax error: -3

Severe:
- No answer: 0 points
- Completely incorrect logic: -7 to -10

================================================================================
GRADING CALIBRATION EXAMPLES (IMPORTANT)
================================================================================
Use the following examples to calibrate scoring. Your grading must be
consistent with these standards.

----------------------------------------------------------------------
EXAMPLE A — FULL CREDIT RESPONSES (All 10 Points)
----------------------------------------------------------------------

Question 4.1 (Score: 10)
select sum(ps_availqty)
from partsupp, part
where p_partkey = ps_partkey 
and p_name = 'blush thistle blue yellow saddle'

Why 10?
- Correct join
- Correct filter
- Correct SUM aggregation
- Functionally equivalent

Question 4.2 (Score: 10)
select distinct(p_partkey)
from part, partsupp, supplier
where p_partkey = ps_partkey 
and ps_suppkey = s_suppkey 
and s_acctbal > 1000

Why 10?
- Correct joins
- Correct filter
- DISTINCT included
- Fully correct logic

Question 4.3 (Score: 10)
select s_name, count(*)
from supplier, partsupp
where s_suppkey = ps_suppkey
group by s_suppkey

Why 10?
- Correct grouping
- Correct join
- Proper aggregation

Question 4.4 (Score: 10)
select max(numParts) 
from (
    select count(*) as numParts 
    from partsupp 
    group by ps_suppkey
) as T

Why 10?
- Correct nested query
- Correct grouping
- Correct MAX aggregation

Question 4.5 (Score: 10)
select distinct(s_suppkey)
from supplier, partsupp as ps1, partsupp as ps2
where s_suppkey = ps1.ps_suppkey 
and ps1.ps_suppkey = ps2.ps_suppkey 
and ps1.ps_partkey != ps2.ps_partkey

Why 10?
- Correct self join logic
- Ensures at least two different parts
- DISTINCT included


----------------------------------------------------------------------
EXAMPLE B — LOW / INCORRECT RESPONSES
----------------------------------------------------------------------

Question 4.1 (Score: 4)
Select PS_AVAILQTY 
FROM PartSupplier 
WHERE P_NAME = 'blush thistle blue yellow saddle'

Why 4?
- No SUM aggregation
- No join between PART and PARTSUPP
- Incorrect schema usage

Question 4.2 (Score: 5)
Select P_PARTKEY 
FROM Suppliers 
WHERE S_ACCTBAL > 1000

Why 5?
- Missing join to PARTSUPP
- Missing DISTINCT
- Only partially correct logic

Question 4.3 (Score: 2)
Select S_NAME 
FROM Supplier 
WHERE COUNT(PART)

Why 2?
- Invalid aggregation
- Missing GROUP BY
- Incorrect SQL structure

Question 4.4 (Score: 2)
Select S_NAME 
FROM Supplier 
WHERE SIZE=MAX()

Why 2?
- No nested query
- Incorrect aggregation logic
- Completely incorrect structure

Question 4.5 (Score: 2)
Select S_SUPPKEY 
FROM Supplier 
WHERE P_PARTKEY = 2

Why 2?
- No grouping
- No comparison of two parts
- Completely incorrect logic

----------------------------------------------------------------------

IMPORTANT:
When grading student responses, align your scoring scale with these examples.
Be consistent with these standards across all submissions.

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
        'rubric': """Core Logic Errors:
- Missing SUM(ps_availqty): -3
- Using COUNT instead of SUM: -2
- SUM(*) instead of SUM(ps_availqty): -2
- SUM of wrong attribute: -3
- Missing join PART.p_partkey = PARTSUPP.ps_partkey: -3
- Missing PARTSUPP in FROM: -2
- Missing PART in FROM: -2
- Missing filter p_name = 'blush thistle blue yellow saddle': -2
- Comparing P_PARTKEY to string instead of P_NAME: -2
GroupBy Misuse: Unnecessary GROUP BY: -4
Syntax Errors: Missing SELECT: -2, SUM without parens: -1, && instead of AND: -2
Severe: Empty answer: 0 points, Completely incorrect logic: -8 to -10""",
        'key': 'question_4_1',
    },
    '4.2': {
        'question': "Return the distinct partkeys of parts that are supplied by suppliers whose account balance (S_ACCTBAL) is greater than 1000.",
        'correct_answers': """1. SELECT DISTINCT PS_PARTKEY
   FROM PARTSUPP PS JOIN SUPPLIER S ON PS.PS_SUPPKEY = S.S_SUPPKEY
   WHERE S.S_ACCTBAL > 1000;
2. SELECT DISTINCT ps_partkey FROM PartSupp, Supplier WHERE ps_suppkey = s_suppkey AND s_acctbal > 1000""",
        'rubric': """Core Logic Errors:
- Missing DISTINCT: -2
- Missing ps_suppkey = s_suppkey join: -3
- Missing PARTSUPP in FROM: -2
- Wrong projection (e.g., SELECT s_suppkey): -3
- Missing filter s_acctbal > 1000: -2
Incorrect Patterns: Unnecessary GROUP BY: -2, SELF JOIN unnecessarily: -4
Severe: No answer: 0 points""",
        'key': 'question_4_2',
    },
    '4.3': {
        'question': "Return the number of distinct parts that are supplied by each supplier. Output: S_NAME, COUNT",
        'correct_answers': """1. SELECT S.S_NAME, COUNT(DISTINCT PS.PS_PARTKEY)
   FROM SUPPLIER S JOIN PARTSUPP PS ON S.S_SUPPKEY = PS.PS_SUPPKEY
   GROUP BY S.S_NAME;
2. SELECT S_NAME, COUNT(*) FROM PARTSUPP, SUPPLIER WHERE PARTSUPP.PS_SUPPKEY = SUPPLIER.S_SUPPKEY GROUP BY S_NAME""",
        'rubric': """Core Logic Errors:
- Missing GROUP BY s_name: -3
- Grouping by s_suppkey instead of s_name: -2
- Missing COUNT(*) or COUNT(DISTINCT ...): -2
- Missing s_name in SELECT: -2
- Missing join between SUPPLIER and PARTSUPP: -3
- Missing join predicate ps_suppkey = s_suppkey: -3
Aggregation Errors: DISTINCT(COUNT(...)) instead of COUNT(DISTINCT ...): -2
Severe: Empty answer: 0 points, Incorrect/incomplete: -8""",
        'key': 'question_4_3',
    },
    '4.4': {
        'question': "Return the maximal number of distinct parts that are supplied by any supplier. (Nested queries required)",
        'correct_answers': """1. SELECT MAX(part_count) FROM (
        SELECT COUNT(DISTINCT PS_PARTKEY) AS part_count FROM PARTSUPP GROUP BY PS_SUPPKEY) t;
2. WITH PartCounts AS (SELECT ps_suppkey, COUNT(DISTINCT ps_partkey) as cnt FROM PartSupp GROUP BY ps_suppkey)
   SELECT MAX(cnt) FROM PartCounts""",
        'rubric': """Core Requirements:
- Missing nested query: -3
- Missing COUNT(DISTINCT ps_partkey): -3
- Missing GROUP BY ps_suppkey: -3
- Missing MAX in outer query: -3
Logic Errors: Using SUM instead of COUNT: -3, max(ps_availqty) instead of counting parts: -2
Severe: Empty or meaningless: 0 points, Most logic incorrect: -8 to -10""",
        'key': 'question_4_4',
    },
    '4.5': {
        'question': "Return the keys of suppliers who have supplied at least two different parts.",
        'correct_answers': """1. SELECT PS_SUPPKEY FROM PARTSUPP GROUP BY PS_SUPPKEY HAVING COUNT(DISTINCT PS_PARTKEY) >= 2;
2. SELECT DISTINCT ps1.ps_suppkey FROM PartSupp ps1, PartSupp ps2
   WHERE ps1.ps_suppkey = ps2.ps_suppkey AND ps1.ps_partkey != ps2.ps_partkey""",
        'rubric': """Core Logic Errors:
- Missing self join on PARTSUPP: -5
- Missing ps1.ps_partkey != ps2.ps_partkey: -3
- Missing ps1.ps_suppkey = ps2.ps_suppkey: -3
- HAVING COUNT(ps_partkey) = 2 instead of >= 2: -2
Aggregation Misuse: COUNT(distinct ps_partkey) without grouping: -7, HAVING without GROUP BY: -4
Severe: No answer: 0 points, Completely incorrect logic: -7 to -10""",
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

GRADING INSTRUCTIONS:
1. If the query is functionally equivalent to a correct answer, give 10 points immediately.
2. Otherwise, apply ONLY the deductions listed in the rubric below. Do not invent new deductions.
3. Start from 10, subtract deductions, minimum score is 0.

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