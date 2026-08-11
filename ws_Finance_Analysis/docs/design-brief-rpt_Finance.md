# Design Brief — rpt_Finance (Overview / Transactions / Trends)

## Design Identity

**Tone:** *Institutional Fintech Ledger* — restrained, high-trust, numbers-first. Near-white surfaces, a single confident teal brand accent, tabular (lining) figures everywhere money is shown, generous whitespace so KPIs read instantly. This replaces the prior ad-hoc layout (KPI strip clipping, inconsistent spacing) with a formalized, repeatable system.

**Signature:** *Teal Ledger* — every KPI card shows an accent bar in its mapped color plus a compact context label; every chart tied to a measure reuses that measure's exact color (never a fresh hue); tables use a light teal-tint header row and zebra striping; the teal header band + white title text repeats identically on all three pages, anchoring the report identity.

- `current_tone`: ad-hoc / inconsistent (KPI clipping issue fixed in isolation, no formal system)
- `current_signature`: none formalized

**Mode:** brownfield (existing `rpt_Finance.Report`, 3 live pages: Overview, Transactions, Trends). Canvas preserved at `1280 x 720` per explicit user spec (not resized to FHD). Existing Lotusoftware custom theme (`RegisteredResources`, teal `#197278` primary) preserved — only page-level layout/visual work in this pass.

## Color Map (max 4 intentional + neutral/semantic, per CLAUDE.md Design rules)

| Measure family | Color | Tint | Rationale |
|---|---|---|---|
| Primary money (`Total Amount`, `Successful Amount`, `Total Fee & Tax Revenue`) | `#197278` (brand teal) | `#DCEFEF` | Primary financial throughput |
| Volume (`Total Transactions`, `Active Customers`, `Successful Transactions`) | `#4FA8B0` (light teal accent) | `#E6F4F5` | Distinguishes counts from currency without leaving the teal family |
| Risk (`% Fraud Rate`, `Fraud Transactions`, `Avg Risk Score`) | `#E8A33D` (amber) | `#FBEBD2` | Reserved semantic warning color |
| Negative / Failed (`Failed Amount`, `Failed Transactions`) | `#D13438` (red) | `#F9DBDC` | Reserved semantic bad color, used only for Failed |
| Neutral / chrome | `#5B6770` (slate grey) | `#F3F2F1` (= left panel) | Gridlines, secondary labels, non-highlighted context |

`color_strategy: measure_match` on every card/line; `gradient` (tint→base) on every bar breakdown of that same measure.

## Shared Page Skeleton (applies to all 3 pages)

```yaml
canvas: { width: 1280, height: 720, margin: 20, gutter: 16, snap: 8 }
regions:
  header:  { x: 0,   y: 0,  w: 1280, h: 56  }   # teal #0F6C74, full width, on top
  rail:    { x: 0,   y: 56, w: 200,  h: 664 }   # #F3F2F1, slicers/nav, below header
  content: { x: 220, y: 68, w: 1048, h: 640 }   # #FAFAFA, 12px inset from rail/header
```

- `header` always carries exactly one white textbox placement (`page_title`), left-anchored, 18pt SemiBold, plus a slim white subtitle/breadcrumb at 10pt where noted.
- `rail` always carries the page's slicers stacked vertically (56px tall each, 16px gaps, full 168px usable width), reset-friendly, top-anchored at y=72.
- No visual in `content` may start above y=68 or extend past x=1268 / y=708.

---

## Page 1 — Overview

**Archetype:** Executive Summary, **Variant A (Hero-Right)** — 4 KPIs of comparable weight plus one clear hero metric (Total Amount trend) with a meaningful monthly trend.
**Title:** "Where the Money Moved This Period"

### Rail slicers (2 — inline count keeps rail from feeling sparse against 664px height, so cards below reinforce it)
- Year (`Date[Year]`, dropdown)
- Channel (`Channel[channel]`, list)

### Content layout (1048 x 640)

| Row | y | h | Visuals |
|---|---|---|---|
| KPI strip | 68 | 104 | 4 cards, 248w each, 16 gutter |
| Hero + breakdown | 188 | 260 | Hero line 616w · Channel breakdown bar 416w (16 gutter) |
| Status + merchant | 464 | 224 | Status breakdown bar 616w · Top merchant categories bar 416w |

### Placements

- `kpi_total_amount` — cardVisual, `Total Amount`, x=220 y=68 w=248 h=104, color `#197278`
- `kpi_total_transactions` — cardVisual, `Total Transactions`, x=484 y=68 w=248 h=104, color `#4FA8B0`
- `kpi_avg_transaction_value` — cardVisual, `Avg Transaction Value`, x=748 y=68 w=248 h=104, color `#197278`
- `kpi_fraud_rate` — cardVisual, `% Fraud Rate`, x=1012 y=68 w=248 h=104, color `#E8A33D` (this is the risk pulse on an otherwise all-teal strip — earns its space as the one non-financial-throughput signal)
- `hero_amount_trend` — lineChart, purpose "How is total transaction value trending month over month?", Category `Date[MonthYear]` (sort by `MonthYearSort`), Y `_Measures[Total Amount]`, color `#197278`, x=220 y=188 w=616 h=260
- `channel_breakdown` — barChart (horizontal, sorted desc), purpose "Which channel drives the most transaction value?", Category `Channel[channel]`, Y `_Measures[Total Amount]`, gradient of `#197278`, x=852 y=188 w=416 h=260
- `status_breakdown` — barChart (horizontal, sorted desc), purpose "How much value clears successfully vs fails?", Category `Transactions[transaction_status]`, Y `_Measures[Total Amount]`, semantic per-category color (Success `#197278`, Failed `#D13438`, Pending `#5B6770`), x=220 y=464 w=616 h=224
- `merchant_breakdown` — barChart (horizontal, sorted desc, top 8), purpose "Which merchant categories generate the most spend?", Category `Merchant[merchant_category]`, Y `_Measures[Total Amount]`, gradient of `#197278`, x=852 y=464 w=416 h=224

**space_audit:** content 1048x640 fully allocated across 3 rows + gutters; empty_cell_pct ≈ 0; largest region (hero+breakdown row) = 260/640 ≈ 40% of content height, justified as the archetype's explanatory hero paired with a same-height breakdown so neither starves the other.

---

## Page 2 — Transactions

**Archetype:** Analytical Canvas, **Variant A (Filter-Rail)** — the persistent left rail is the filter rail; risk/status slicers justify it (4 slicers fill well over 50% of the 664px rail).
**Title:** "Transaction Risk & Status Breakdown"

> Model governance note: `amount`, `transaction_date`, `risk_score`, `is_fraud_bool`, etc. are intentionally `isHidden` raw columns per `sm_Finance` design (measures only, per CLAUDE.md). The "detail table" therefore aggregates by the visible categorical dimensions with measures as values — a transaction breakdown table, not a raw unhidden-column row list. This keeps the report compliant with the model's hidden-column governance.

### Rail slicers (4 — justifies filter rail)
- Transaction Status (`Transactions[transaction_status]`, list)
- Transaction Type (`Transactions[transaction_type]`, dropdown, search on)
- Channel (`Channel[channel]`, list)
- Year (`Date[Year]`, dropdown)

### Content layout (1048 x 640)

| Row | y | h | Visuals |
|---|---|---|---|
| KPI strip | 68 | 88 | 4 compact cards |
| Risk + status charts | 172 | 176 | Avg risk by merchant 512w · Status volume bar 520w |
| Detail breakdown table | 364 | 344 | Full width tableEx |

### Placements

- `kpi_total_tx` — cardVisual, `Total Transactions`, x=220 y=68 w=248 h=88, color `#4FA8B0`
- `kpi_successful_tx` — cardVisual, `Successful Transactions`, x=484 y=68 w=248 h=88, color `#197278`
- `kpi_failed_tx` — cardVisual, `Failed Transactions`, x=748 y=68 w=248 h=88, color `#D13438`
- `kpi_fraud_tx` — cardVisual, `Fraud Transactions`, x=1012 y=68 w=248 h=88, color `#E8A33D`
- `risk_by_merchant` — barChart (horizontal, sorted desc), purpose "Which merchant categories carry the highest average risk?", Category `Merchant[merchant_category]`, Y `_Measures[Avg Risk Score]`, gradient of `#E8A33D`, x=220 y=172 w=512 h=176
- `status_volume` — barChart (horizontal, sorted desc), purpose "How many transactions land in each status?", Category `Transactions[transaction_status]`, Y `_Measures[Total Transactions]`, semantic per-category (Success `#197278`, Failed `#D13438`, Pending `#5B6770`), x=748 y=172 w=520 h=176 (labels neutral dark `#252423`)
- `transaction_detail_table` — tableEx, purpose "What does transaction volume, value, and risk look like by type, status, channel, and merchant?", rows `Transactions[transaction_type]`, `Transactions[transaction_status]`, `Channel[channel]`, `Merchant[merchant_category]`; values `_Measures[Total Transactions]`, `_Measures[Total Amount]`, `_Measures[Total Fees]`, `_Measures[Avg Risk Score]`, `_Measures[% Fraud Rate]`; x=220 y=364 w=1048 h=344, zebra striping teal tint `#E6F4F5`, header fill `#DCEFEF`, tabular numerals, sortable

**space_audit:** content fully allocated: KPI strip (88h) + chart row (176h) + table (344h) + 2×16 gutters = 640; empty_cell_pct ≈ 0; largest region = table at 344/640 ≈ 54% — justified because it's a `tableEx` detail region (archetype explicitly puts precision-on-demand as the page's dominant zone, not a bare card).

---

## Page 3 — Trends

**Archetype:** Analytical Canvas, **Variant B (Inline-Slicers)** — only 2 slicers needed; content gets full width for the trend hero.
**Title:** "Monthly Value Is Up — Here's the Shape of the Year"

### Rail slicers (2)
- Year (`Date[Year]`, dropdown)
- Channel (`Channel[channel]`, list)

### Content layout (1048 x 640)

| Row | y | h | Visuals |
|---|---|---|---|
| Hero trend | 68 | 280 | Full-width monthly trend |
| YoY + volume | 364 | 156 | YoY column 512w · Transaction volume trend 520w |
| Weekday + quarter | 536 | 172 | Day-of-week bar 512w · Quarter seasonality 520w |

### Placements

- `hero_monthly_trend` — lineChart, purpose "How has total transaction value moved month over month across the full date range?", Category `Date[MonthYear]` (sort `MonthYearSort`), Y `_Measures[Total Amount]`, color `#197278`, reference line/annotation at prior-year-same-month if supported; x=220 y=68 w=1048 h=280
- `yoy_amount` — columnChart, purpose "Is each year growing or shrinking versus the year before?", Category `Date[Year]`, Y `_Measures[Total Amount YoY %]`, diverging conditional format (positive `#197278`, negative `#D13438`), reference line at 0%, x=220 y=364 w=512 h=156
- `transactions_trend` — lineChart, purpose "Does transaction count follow the same shape as transaction value?", Category `Date[MonthYear]` (sort `MonthYearSort`), Y `_Measures[Total Transactions]`, color `#4FA8B0`, x=748 y=364 w=520 h=156
- `weekday_pattern` — columnChart, purpose "Do weekdays or weekends carry more transaction value?", Category `Date[DayName]` (natural order: Mon–Sun), Y `_Measures[Total Amount]`, semantic split (weekend `#E8A33D` vs weekday `#197278` via `Date[IsWeekend]`), x=220 y=536 w=512 h=172
- `quarter_seasonality` — clusteredColumnChart, purpose "Which quarters are strongest, and is that consistent year over year?", Category `Date[Quarter]`, Legend `Date[Year]`, Y `_Measures[Total Amount]`, categorical palette (teal family, one shade per year), x=748 y=536 w=520 h=172

**space_audit:** content fully allocated: hero (280h) + 2×156/172 rows + 2×16 gutters ≈ 640; empty_cell_pct ≈ 0; largest region = hero trend at 280/640 = 44% — justified as the archetype's single explanatory hero for "how did value move over time," with 4 supporting panels covering YoY, volume, weekday, and seasonality so no other analytical angle is starved.

---

## Cross-page rules

- `personalizeVisuals: true` disabled (KPI/exec-leaning pages; keep surface clean) except Transactions page which may enable it (analytical archetype).
- All bar charts sorted descending by value except `weekday_pattern` (natural Mon–Sun order) and `quarter_seasonality` (natural Q1–Q4 order).
- All currency measures use the model's existing `₹#,0.00` format string; percentages use `0.0%`/`0.00%` as defined in `_Measures`. No re-formatting in visuals.
- Card visuals: `visualHeader.show = false`, `border.show = false`, tabular/lining numerals, category label on, accent bar per color map.
- No donut, gauge, 3D, or pie visuals anywhere in this report.
- Data labels on all charts use neutral dark `#252423`, never the accent color.

## Accessibility

- Every semantic color pairing (Success/Failed/Pending, positive/negative YoY, weekday/weekend) is also distinguished by category label/legend text, not color alone.
- Text on teal header (`#0F6C74`) is pure white `#FFFFFF` — verified ≥4.5:1 contrast.
- Text on `#FAFAFA` content and `#F3F2F1` rail uses the theme's dark foreground (`#252423`) — ≥4.5:1 contrast.

---

**Handoff:** implement via `powerbi-report-authoring` against the existing `rpt_Finance.Report` PBIR definition, preserving the current theme registration. Validate → reload Desktop → screenshot → review after each page.
