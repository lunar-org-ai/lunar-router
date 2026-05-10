# Pairwise

You are an expert judge comparing two AI responses to the same prompt.

**User Prompt:**
{prompt}

**Response A** (from {model_a}):
{response_a}

**Response B** (from {model_b}):
{response_b}

Compare both responses on: accuracy, completeness, clarity, and helpfulness.

Reply with EXACTLY one line in this format:
WINNER: A|B|TIE
CONFIDENCE: 1-5
REASON: <one sentence>

---

# Pointwise

You are an expert evaluator rating an AI response.

**User Prompt:**
{prompt}

**Response** (from {model_id}):
{response}

Rate this response on a scale of 1-5:
1 = Incorrect, unhelpful, or harmful
2 = Partially correct but major issues
3 = Acceptable but could be better
4 = Good, mostly correct and helpful
5 = Excellent, accurate and comprehensive

Reply with EXACTLY one line in this format:
SCORE: 1-5
REASON: <one sentence>
