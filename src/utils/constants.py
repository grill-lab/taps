DSPY_PROMPT_SUFFIX = """

---

Follow the following format.

Dialogue:
<user_utterance>

Applicable Standing Instructions:
<applicable_standing_instructions>

API Calls: 
API calls to solve the user task

---


"""

DSPY_PROMPT_SUFFIX_SIMPLE_TAG = """

---

Follow the following format.

Dialogue:
<user_utterance>

Applicable Standing Instructions:
<applicable_standing_instructions>

Tagged Standing Instructions:
<tagged applicable standing instructions>

API Calls: 
API calls to solve the user task

---


"""


DSPY_PROMPT_SUFFIX_TAG = """

---

Follow the following format.

Dialogue:
<user_utterance>

Applicable Standing Instructions:
<applicable_standing_instructions>

Tagged Standing Instructions:
Tagged standing instructions

API Calls: 
API calls to solve the user task

---


"""