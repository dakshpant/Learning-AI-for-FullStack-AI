I have an existing restaurant order management agent implemented using Python, LangGraph, LangChain, Groq, and Pydantic.

Modify my existing main.py. Do not rewrite the project from scratch or remove existing features without a reason.

My most important architectural requirement is a status-driven workflow coordinated by the llm_node.

ARCHITECTURE:

1. user_input receives the user's next message.
2. llm_node extracts the order or classifies the user's intent.
3. order_confirm validates the order against MENU and updates order details and status.
4. After order_confirm, control must return to llm_node.
5. llm_node reads status, order details, and retry counters, then decides the next workflow status and generates an appropriate customer-facing message.
6. After cook, control must return to llm_node.
7. After serve, control must return to llm_node.
8. A Python conditional-edge function routes according to the status written by llm_node.
9. The graph ends only when final_result is SUCCESS or FAIL, or the user explicitly ends the session.

Use this status-driven pattern:

Operational node -> llm_node -> conditional routing -> next node.

IMPORTANT: The language model should interpret user intent and generate messages, but Python must enforce inventory and retry rules. The llm_node may use deterministic Python conditions to interpret the operational status and set the next status. Do not rely on free-form LLM output to enforce business rules.

FIX THESE ISSUES:

- Partial and unavailable orders must not terminate the graph immediately.
- The llm_node must ask the user whether to accept a partial quantity or place a different order.
- After the user replies, user_input must capture the response and preserve the complete state and conversation history.
- A rejected partial order followed by a new order must be processed as a new order.
- An unrelated question should receive a food-ordering-only response and should not consume an order retry unless explicitly defined otherwise.
- Connect the order retry counter to the actual order-confirmation flow.
- Ensure all status names match across nodes and routing functions.
- Ensure all conditional-edge destinations correspond to registered nodes or END.
- Prevent infinite loops.
- Do not reset retry counters when the user changes their order.
- Do not report success until serving succeeds.
- If serving fails, retry cooking and serving only when the relevant retry counters allow it.
- Never call cook again if cook retries are exhausted.
- Issue an apology and set final_result to FAIL when retries are exhausted.

RETRY RULES:

- Three total order attempts, including the initial attempt.
- Two additional cooking attempts after the initial cooking attempt.
- Two additional serving attempts after the initial serving attempt.
- Every failed retryable cooking attempt consumes one cook retry when another cooking attempt is scheduled.
- A serving failure that triggers re-cooking and re-serving consumes one serve retry and one cook retry.
- Counters must never become negative.
- Define precisely when each counter is decremented.

TESTING:

Add deterministic scripted outcomes for cook and serve so that test scenarios do not depend on random chance.

TC1:
- Unrelated question.
- User orders a partially available quantity and rejects it.
- User submits another order that is unavailable.
- Order attempts are exhausted.
- Expected result: FAIL, without cooking or serving.

TC2:
- Fully available order.
- Cook fails once and then succeeds.
- Serve fails once.
- Cook succeeds during recovery.
- Serve succeeds.
- Expected result: SUCCESS.

TC3:
- Partial order rejected.
- A new fully available order is submitted.
- Cook fails once, then succeeds.
- Serve fails.
- Recovery cook succeeds.
- Serve fails again.
- Cook retries are exhausted, so no more cooking is allowed.
- Expected result: FAIL.

For each test, assert the final result and the number of cook and serve calls. Verify that TC3 never reports that the order is complete.

Keep the interactive terminal mode working independently from automated tests.

Return:
1. The complete corrected main.py.
2. A concise explanation of the changes.
3. The exact commands to run the application and tests.
4. Any remaining assumptions or limitations.

Do not claim the tests pass unless you have actually executed them.