"""Structured prompts for LLM financial news sentiment analysis."""

SENTIMENT_ANALYSIS_PROMPT = """You are a forex market analyst. Analyze this news article's impact on the currency pair {pair}.

Respond with ONLY a JSON object (no other text, no markdown, no explanation):
{{
  "direction": <float from -1.0 to 1.0, where -1=very bearish, 0=neutral, 1=very bullish>,
  "confidence": <float from 0.0 to 1.0>,
  "volatility": <float from 0.0 to 1.0, where 0=calm, 1=high volatility expected>,
  "currencies_affected": <list of currency codes, e.g. ["USD", "EUR"]>
}}

Article title: {title}
Article body: {body}"""

BATCH_SENTIMENT_PROMPT = """You are a forex market analyst. Analyze these {count} news articles' collective impact on the currency pair {pair}.

For EACH article, respond with a JSON object on a separate line. Output ONLY a JSON array (no other text, no markdown):
[
  {{"direction": <float -1 to 1>, "confidence": <float 0 to 1>, "volatility": <float 0 to 1>, "currencies_affected": [<str>], "index": 0}},
  {{"direction": <float -1 to 1>, "confidence": <float 0 to 1>, "volatility": <float 0 to 1>, "currencies_affected": [<str>], "index": 1}},
  ...
]

Articles:
{articles_text}"""

LIVE_SENTIMENT_PROMPT = """You are a forex market analyst. Based on the following recent news headlines for {pair}, provide an overall sentiment assessment.

Respond with ONLY a JSON object (no other text):
{{
  "direction": <float from -1.0 to 1.0>,
  "confidence": <float from 0.0 to 1.0>,
  "volatility": <float from 0.0 to 1.0>,
  "currencies_affected": [<str>],
  "summary": <one sentence summary of market outlook>
}}

Recent headlines:
{headlines}"""