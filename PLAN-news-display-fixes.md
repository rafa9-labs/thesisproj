# Plan: News Display Fixes (6 issues)

## Issue 1: Double-encoding of `&amp;` (Bug)
**Root cause:** Backend `_strip_html()` strips HTML tags but does NOT decode entities (`&amp;` → `&`). Frontend `escapeHtml()` then re-encodes, producing `&amp;amp;`.

**Files:**
- `news/scraper.py:82` — `_strip_html()` doesn't call `html.unescape()`
- `frontend/src/pages/News/NewsPage.tsx:204` — `escapeHtml()` always runs, even on pre-encoded text

**Fix (2 options, recommend Option A):**

**Option A — Fix backend `_strip_html` to decode entities:**
```python
# news/scraper.py
import html as _html

def _strip_html(text: str) -> str:
    raw = _MULTI_SPACE.sub(" ", _HTML_TAG.sub(" ", text or "")).strip()
    return _html.unescape(raw)
```
This ensures all downstream consumers (summary, body, highlighted_body) get clean text.

**Option B — Add `decodeHtml` to frontend:**
```typescript
function decodeHtml(text: string): string {
  const el = document.createElement("textarea");
  el.innerHTML = text;
  return el.value;
}
```
Call in `formatNewsBody()` before returning paragraphs. Less clean — mixes DOM with data processing.

**Verdict:** Option A is better. Fix at the source.

---

## Issue 2: Binary sentiment scores (±1.00) — Backend
**Root cause:** The LLM prompt returns `direction` as hard ±1.0, and VADER compound is also ±1.0 for most news. No calibration/scaling is applied.

**Evidence from backend (`pipeline/llm/sentiment.py:276`):**
```python
"direction": float(np.clip(parsed.get("direction", 0.0), -1.0, 1.0))
```
The LLM is being asked for a direction in [-1, 1] but the prompt likely encourages extreme outputs.

**Files:**
- `pipeline/llm/sentiment.py:276-279` — `_parse_llm_json()`
- `pipeline/llm/sentiment.py:320-325` — `_parse_llm_batch_json()`
- The LLM system prompt (need to find where it's defined)

**Fix:**
1. Find the LLM prompt template and add explicit instruction: "Return direction as a continuous value between -1.0 and 1.0, not just -1 or 1. Use 0.3 for mild bullish, 0.7 for strong bullish, etc."
2. Apply soft-clamping in `_parse_llm_json()`: multiply by 0.85 to pull extremes toward center, or use `tanh(direction * 2) * 0.9` for softer mapping.
3. Consider adding a sigmoid/tanh calibration in the merge step.

**Note:** This is a backend change. The frontend can't fix binary scores — it needs to come from the scorer.

---

## Issue 3: Inconsistent sentiment thresholds across components
**Root cause:** Each component defines its own threshold constants independently.

| Component | File | Bullish | Bearish |
|---|---|---|---|
| `ScoredArticleRow` | NewsPage.tsx:280 | > 0.05 | < -0.05 |
| `MacroFeedRow` | NewsPage.tsx:386 | > 0.05 | < -0.05 |
| `SentimentNewsWidget ArticleRow` | SentimentNewsWidget.tsx:159 | > 0.2 | < -0.2 |
| `MarketPulsePanel ArticleRow` | MarketPulsePanel.tsx:206 | > 0.2 | < -0.2 |
| `sentimentLabel` (badge) | NewsPage.tsx:252 | > 0.3 | < -0.3 |
| `getImpactLabel` | MarketPulsePanel.tsx:48 | abs*mag ≥ 0.6 → High | — |
| `bias_label` (backend) | scraper.py:117 | > 0.05 | < -0.05 |

**Fix:** Define shared threshold constants in a single location and import everywhere.

**File:** `frontend/src/lib/sentiment-thresholds.ts` (new)
```typescript
export const SENTIMENT_THRESHOLDS = {
  /** Border color / visual classification */
  bullish: 0.15,
  bearish: -0.15,
  /** Badge label classification */
  badge_bullish: 0.3,
  badge_bearish: -0.3,
  /** Impact tiers */
  impact_high: 0.6,
  impact_med: 0.3,
} as const;
```

Then update all 5 components to import from this file. Use 0.15 for border colors (compromise between 0.05 too sensitive and 0.2 too insensitive).

---

## Issue 4: No article truncation in expanded view (UX)
**Root cause:** `NewsParagraphs` renders the full article body with no character/line limit.

**Files:**
- `NewsPage.tsx:229-248` — `NewsParagraphs` component
- Used in both `ScoredArticleRow:371` and `MacroFeedRow:454`

**Fix:** Add a max-height + scrollable container to expanded article body.

```tsx
// In ScoredArticleRow and MacroFeedRow expanded view:
{expanded && hasBody && (
  <div className="pl-4 border-l-2 border-slate-700 ml-2 mt-3 max-h-64 overflow-y-auto">
    <NewsParagraphs text={...} />
  </div>
)}
```

`max-h-64` = 256px (about 8-10 lines of body text). Also add a "Show more" button for articles longer than 500 chars.

---

## Issue 5: Double-highlighting when `highlighted_body` is used
**Root cause:** Backend `_highlight_sentiment_phrases()` returns HTML with `<span>` tags. Frontend `NewsParagraphs` → `highlightPhrases()` → `escapeHtml()` strips those tags and re-applies its own highlighting.

**Flow:**
1. Backend: `_strip_html(body)` → `_highlight_sentiment_phrases(text)` → returns `<span class="text-emerald-400...">rally</span>`
2. Frontend `MacroFeedRow:454`: passes `hlBody` to `NewsParagraphs`
3. `NewsParagraphs:230`: calls `formatNewsBody(hlBody)` → `stripHtml()` removes the `<span>` tags
4. `highlightPhrases()` re-wraps with its own `<span>` tags

**Fix:** When `highlighted_body` is provided, render it directly as HTML without re-processing.

```tsx
function NewsParagraphs({ text, preHighlighted }: { text: string; preHighlighted?: boolean }) {
  if (preHighlighted) {
    // Backend already applied <span> highlighting — render directly
    const paragraphs = text.split(/\n+/).filter(s => s.trim().length > 1);
    return paragraphs.map((p, i) => (
      <p key={i} dangerouslySetInnerHTML={{ __html: p }} />
    ));
  }
  // ... existing logic
}
```

Then in `MacroFeedRow`:
```tsx
<NewsParagraphs text={hlBody} preHighlighted={!!hlBody} />
```

---

## Issue 6: `escapeHtml` runs on pre-highlighted HTML from backend
**Root cause:** Same as Issue 5 — `highlightPhrases()` calls `escapeHtml()` on every text segment, which destroys `<span>` tags that the backend already inserted.

**Fix:** Covered by Issue 5's fix. When `preHighlighted=true`, skip `highlightPhrases()` entirely.

---

## Implementation Order

1. **Issue 1** (backend `_strip_html`) — 1 line change in `news/scraper.py`
2. **Issue 5+6** (double-highlighting) — `NewsParagraphs` + `MacroFeedRow` changes
3. **Issue 3** (threshold constants) — Create `sentiment-thresholds.ts`, update 5 components
4. **Issue 4** (article truncation) — Add `max-h-64 overflow-y-auto` to 2 expanded views
5. **Issue 2** (binary scores) — Backend LLM prompt calibration (separate from frontend)

Issues 1, 3, 4, 5, 6 are frontend-near. Issue 2 is backend-only.
