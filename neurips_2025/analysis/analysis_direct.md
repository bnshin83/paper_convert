

# Review: using LLMs + retrieval to find “trends / acceptable topics / next directions” per primary area

Your instinct is right: **an LLM is very good at synthesis**, but **very bad (and expensive) at reading everything** across thousands of papers. The “index once → retrieve on demand → LLM reads only evidence” design is the correct way to make LLMs useful for your goal.

That said, there are a few *critical* pitfalls to address so the results are actually meaningful and not just “pretty summaries”.

---

# 1) First sanity checks (otherwise “acceptable topics” can be misleading)

## A) Do you truly have accepted *and* rejected labels?
Your current NeurIPS analysis report shows:

- total: 5542  
- accepted: 5287  
- rejected: 255  

That implies an acceptance rate of:

$$
p_{\text{accept}} = \frac{5287}{5542} \approx 0.954
$$

That’s **not a realistic NeurIPS acceptance rate**, so something is likely true:

- **NeurIPS OpenReview doesn’t expose most rejected submissions publicly**, or
- your “all submissions” export still only contains mostly accepted/visible papers, or
- label inference (`venue`/`decision`) is not aligned with what OpenReview provides.

**Consequence:** if you don’t have true rejects, you can still do **trend mapping**, but you cannot reliably infer “acceptable topics” as *accepted vs rejected*.

## B) Do you have enough full text coverage for RAG?
The report also says only **52 papers with local PDF→MD** were found at the path used by analysis. If your conversion is “done”, then either:

- `--md-root` was pointed to the wrong folder, or
- the MD output isn’t on your local machine (e.g., it’s on scratch), or
- folder naming doesn’t match `<paper_id>__<title>`.

**Consequence:** content-based “direction prediction” (future work / limitations) will be weak if the index doesn’t actually cover the corpus.

---

# 2) What LLMs *should* do vs what your pipeline *should* do

## A) Let the pipeline do counting and ranking (deterministic)
For each primary area, compute:

- **[Topic prevalence]** top keywords/phrases by frequency (from keywords + abstracts).
- **[Topic “selectivity”]** if you have rejects: compare accepted vs rejected terms.
- **[Quality proxy]** if no rejects: compare oral/spotlight/poster buckets or review rating averages (if available).

A robust “accepted vs rejected term lift” can be done with log-odds / smoothed ratios, e.g.:

$$
\delta_w = \log \frac{(c_{w,A}+\alpha)/(N_A+\alpha V)}{(c_{w,R}+\alpha)/(N_R+\alpha V)}
$$

Where:
- \(c_{w,A}\), \(c_{w,R}\): counts of term \(w\) in accepted/rejected
- \(N_A\), \(N_R\): total token counts
- \(V\): vocab size
- \(\alpha\): smoothing

This gives you a ranked list of “terms more associated with acceptance” (again: only valid if rejects are real).

## B) Let the LLM do synthesis *only after retrieval*
LLM tasks that work well **when constrained to retrieved excerpts**:

- **[Taxonomy building]** group retrieved chunks into themes for the area.
- **[Trend narrative]** summarize what is emerging and why, with citations.
- **[Future directions]** extract “limitations / future work” statements into a structured list.
- **[Comparative analysis]** contrast approaches across multiple papers.

---

# 3) A practical workflow that matches your goal (per primary area)

## Step 1 — Candidate topics per primary area (metadata-first)
For each `primary_area`, generate:

- **[Top terms]** from keywords + abstract n-grams.
- **[Differential terms]** accepted vs rejected (if you truly have both), or oral vs poster as a proxy.
- **[Representative seed queries]**: 10–30 short queries per area (keywords + key phrases).

This step is fast, cheap, and gives you *what to retrieve*.

## Step 2 — Evidence retrieval (RAG/FTS)
Use your SQLite FTS index to pull **evidence chunks** for each seed query, but make it systematic:

- **[Coverage constraint]** require at least \(k\) distinct papers per theme (not just many chunks from 1 paper).
- **[Diversity constraint]** cap chunks per paper (e.g., max 1–2 chunks/paper per query).
- **[Section targeting]** query specifically for “limitations”, “future work”, “we leave for future”, etc. to gather direction signals.

This avoids the biggest RAG failure mode: **overfitting to a few highly matchy papers**.

## Step 3 — LLM synthesis with citations
Feed only retrieved `context.md` blocks to the LLM and ask for structured outputs:

Per primary area, produce:

- **[Current trend map]**
  - 5–10 themes
  - 2–4 citations each: `[paper_id:chunk_id]`
- **[“Accepted topics” / “selective topics”]**
  - only if labels are meaningful
  - otherwise phrase it as “dominant topics among accepted / oral / spotlight”
- **[Next directions]**
  - extracted future work / open problems
  - citations for each claim
- **[Risk / uncertainty]**
  - what evidence is thin (low paper coverage)
  - where conclusions might not generalize

---

# 4) Minimal improvements I’d recommend (high leverage)

## A) Metadata-aware retrieval (small code change, big value)
Right now `chunks` stores only `paper_id`. To do per-area retrieval cleanly, you want filtering by `primary_area`, `decision_bucket`, etc.

Two minimal options:

- **[Option 1: store `primary_area` in `chunks` rows]**
  - Add a column `primary_area UNINDEXED` to FTS table (or a side table keyed by `paper_id`)
  - Lets you do: “search diffusion” **within** a specific area

- **[Option 2: join against a `papers` table]**
  - Create `papers(paper_id PRIMARY KEY, primary_area, decision_bucket, accepted_label, …)`
  - Populate from [paper_features.csv](cci:7://file:///Users/bnsmac/paper_convert/neurips_2025/analysis/paper_features.csv:0:0-0:0)
  - Join/filter at query time

This makes your “per primary area” goal first-class.

## B) Consider semantic retrieval later (optional)
FTS5 is BM25 keyword search. For “direction prediction” and higher-level themes, semantic search can help. A hybrid later is ideal:
- BM25 (FTS) + embeddings (cosine)

---

# 5) My bottom line
- **Your idea (LLM + retrieval) is the right architecture** for scalable, evidence-grounded research trend analysis.
- The **two biggest blockers** are:
  - whether your dataset truly includes rejected papers (otherwise “acceptability” is not identifiable)
  - whether your fulltext index actually covers the converted corpus (otherwise “future directions” extraction will be sparse)

---

## One question that determines the next step
Do you currently have **rejected submissions** available in OpenReview for NeurIPS 2025 (with venues/decisions), or is NeurIPS effectively “accepted-only public”?

- If it’s **accepted-only**, we should reframe outputs as “trend mapping within accepted” + “next directions” (still valuable).
- If you truly have **accepted + rejected**, we can build “acceptability” metrics per area (term lift / selectivity) and make stronger claims.

Task status: **analysis strategy reviewed; next step depends on label coverage + md-root coverage.**