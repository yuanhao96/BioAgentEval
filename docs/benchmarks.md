# Biology Benchmarks in BioAgentEval

BioAgentEval supports 9 task suites covering 8 published biology benchmarks plus an internal biomedical core suite. The benchmarks span two categories: **text-based QA** (testing knowledge retrieval and reasoning) and **agentic** (testing multi-step computational workflows with tool use and code execution).

This document describes each benchmark, its evaluation design, and how BioAgentEval maps it to task suites with appropriate graders.

---

## Overview

| Suite File | Benchmark | Type | Tasks | Graders Used |
|---|---|---|---|---|
| `hle_bio_chem.yaml` | HLE Bio/Chem | Text-based | 149 | `mcq_answer`, `model` |
| `frontierscience_bio.yaml` | FrontierScience-Bio | Text-based | 6 | `exact_match`, `numeric_tolerance`, `keyword_coverage`, `model` |
| `lab_bench.yaml` | LAB-Bench | Text-based | 12 | `mcq_answer`, `model` |
| `biomni_eval1.yaml` | Biomni-Eval1 | Agentic | 3 | `tool_calls`, `code_valid`, `groundedness`, `keyword_coverage`, `model` |
| `bixbench.yaml` | BixBench | Agentic | 3 | `code_valid`, `keyword_coverage`, `model` |
| `bioml_bench.yaml` | BioML-Bench | Agentic | 3 | `code_valid`, `numeric_tolerance`, `set_similarity`, `keyword_coverage`, `model` |
| `spatialbench.yaml` | SpatialBench | Agentic | 3 | `set_similarity`, `code_valid`, `keyword_coverage`, `model` |
| `scbench.yaml` | scBench | Agentic | 3 | `set_similarity`, `code_valid`, `numeric_tolerance`, `keyword_coverage`, `model` |
| `bioagent_bench.yaml` | BioAgent Bench | Agentic | 3 | `tool_calls`, `code_valid`, `numeric_tolerance`, `keyword_coverage`, `model` |

All suite files live in `tasks/` and can be loaded via `bioagenteval.loader.load_suite()`.

---

## Text-Based Benchmarks

### HLE Bio/Chem (Humanity's Last Exam)

**Source**: Phan et al., 2025. [arXiv:2501.14249](https://arxiv.org/abs/2501.14249). Published in *Nature* (DOI: 10.1038/s41586-025-09962-4).

**What it is**: A crowdsourced benchmark of 2,500 expert-level questions across 100+ academic subjects, created by 1,100+ subject-matter experts. The biology/chemistry/health subset contains 321 questions designed to be at the frontier of human knowledge — questions that current AI systems struggle with.

**Task format**: Mix of multiple-choice and short-answer questions. Topics span bioinformatics, genetics, molecular biology, biochemistry, pharmacology, and clinical medicine.

**Key metrics**: Accuracy (%). Current SOTA is approximately 44.7% (Gemini 3.1 Pro Preview).

**BioAgentEval suite**: `hle_bio_chem.yaml` contains 149 tasks from the bio/chem subset. Each task uses:
- `mcq_answer` code grader for multiple-choice questions
- `model` grader with detailed rubrics including the expected answer, rationale, and 3-tier scoring

**Notable considerations**: A FutureHouse audit found approximately 29% of bio/chem answers may be incorrect (conflicting with peer-reviewed literature). The adversarial design ("impossible for AI") incentivized edge-case questions where even experts disagree. Users should interpret results with this in mind.

---

### FrontierScience-Bio

**Source**: OpenAI, 2025. [arXiv:2601.21165](https://arxiv.org/abs/2601.21165).

**What it is**: A benchmark of 160 expert-level science questions (100 Olympiad + 60 Research) created by 42 Olympiad medalists and 45 PhD scientists. It was designed to replace saturated benchmarks like GPQA (where GPT-5.2 scored 92% vs. 70% expert baseline). The biology portion covers protein folding, genetic regulatory networks, molecular biology, pharmacology, and immunology.

**Task format**: Two tracks:
- **Olympiad** — constrained short-answer questions with definitive numerical or categorical answers. Graded by exact match.
- **Research** — open-ended PhD-level tasks designed to take 3-5 hours for a human expert. Graded via 10-point expert rubrics.

**Key metrics**: Accuracy (Olympiad), rubric score 0-10 (Research). GPT-5.2 scores 77% Olympiad but only 25% Research, showing the gap between knowledge recall and deep reasoning.

**BioAgentEval suite**: `frontierscience_bio.yaml` contains 6 representative tasks (3 Olympiad + 3 Research). Graders:
- Olympiad tasks: `exact_match` + `numeric_tolerance` for deterministic answer checking, plus `model` grader
- Research tasks: `keyword_coverage` for topic coverage verification, plus `model` grader with detailed rubrics

---

### LAB-Bench (Language Agent Biology Benchmark)

**Source**: Laurent, Janizek, Ruzo et al. (FutureHouse), 2024. [arXiv:2407.10362](https://arxiv.org/abs/2407.10362).

**What it is**: A benchmark of 2,457 multiple-choice questions testing practical biology research capabilities. Created by FutureHouse to evaluate whether AI can perform tasks that biology PhD students encounter daily. Covers 8 categories across 30 narrower subtasks.

**Subcategories**:

| Category | Count | What it tests |
|---|---|---|
| LitQA2 | 248 | Literature comprehension and reasoning |
| SuppQA | 102 | Supplementary information extraction |
| FigQA | 226 | Scientific figure interpretation |
| TableQA | 305 | Data table reasoning |
| DbQA | 650 | Biological database retrieval |
| ProtocolQA | 135 | Laboratory protocol troubleshooting |
| SeqQA | 750 | DNA/protein sequence manipulation |
| CloningScenarios | 41 | Molecular cloning strategy design |

**Key metrics**: Accuracy (%). All models deeply underperform humans on most categories. FigQA is particularly challenging (near chance for most models).

**BioAgentEval suite**: `lab_bench.yaml` contains 12 representative MCQ tasks (3 per subcategory for LitQA2, CloningScenarios, ProtocolQA, SeqQA). Each task uses:
- `mcq_answer` code grader for answer validation
- `model` grader with rubric explaining the correct answer and reasoning

**Note on LAB-Bench 2**: No separate LAB-Bench 2 publication exists as of February 2026. The benchmark receives ongoing updates via its GitHub repository.

---

## Agentic Benchmarks

Agentic benchmarks require agents to perform multi-step computational workflows: writing code, using tools, analyzing datasets, and producing structured outputs. These benchmarks test capabilities beyond knowledge recall.

### Biomni-Eval1

**Source**: Huang, Zhang, Wang et al. (Stanford/Genentech), 2025. [bioRxiv:2025.05.30.656746](https://www.biorxiv.org/content/10.1101/2025.05.30.656746v1).

**What it is**: The evaluation suite for the Biomni generalist biomedical agent. Contains 433 test instances across 10 task types covering GWAS analysis, CRISPR delivery design, rare disease diagnosis, drug repurposing, and more.

**Task format**: Mixed formats including exact match, letter matching (MCQ-style), JSON parsing, and list intersection. Each task type has specific structured output requirements and a custom `_compute_reward()` function.

**Key metrics**: Binary scoring (1.0/0.0) per task. Biomni achieved 74.4% on LAB-Bench DbQA (matching human 74.7%) and 81.9% on SeqQA (exceeding human 78.8%).

**BioAgentEval suite**: `biomni_eval1.yaml` contains 3 representative tasks (2 DryLab + 1 WetLab). Graders:
- `tool_calls` — verifies the agent used appropriate tools (literature search, analysis tools)
- `code_valid` — validates generated Python code
- `groundedness` — checks for proper citations in literature review tasks
- `keyword_coverage` — verifies key scientific concepts are covered
- `model` grader for overall quality assessment

---

### BixBench

**Source**: Mitchener, Laurent, Andonian et al. (FutureHouse/ScienceMachine), 2025. [arXiv:2503.00096](https://arxiv.org/abs/2503.00096).

**What it is**: A benchmark of 53 analytical scenarios with 296 guiding research questions, built from real experimental datasets in computational biology. Agents must explore heterogeneous input data, write and execute multi-step analysis code, and interpret results.

**Task format**: Open-answer questions over real biological datasets. Each scenario provides input data files and natural-language research questions. A verified subset (BixBench-Verified-50) contains 50 curated questions across 33 data capsules.

**Key metrics**: Accuracy (%). Frontier models achieve only ~17% accuracy in open-answer format and perform no better than random in MCQ format — demonstrating the gap between text QA and real analytical work.

**BioAgentEval suite**: `bixbench.yaml` contains 3 representative tasks spanning sequence analysis, variant filtering, and phylogenetics. Graders:
- `code_valid` — every task requires valid Python code
- `keyword_coverage` — verifies use of correct analytical terms
- `model` grader for code correctness and scientific soundness

---

### BioML-Bench

**Source**: Miller, Greenig, Tenmann, Wang (ScienceMachine), 2025. [bioRxiv:2025.09.01.673319](https://www.biorxiv.org/content/10.1101/2025.09.01.673319v2).

**What it is**: The first benchmark for end-to-end biomedical ML agent evaluation. Tasks are sourced from real benchmarking platforms (ProteinGym, OpenProblems, Kaggle, PolarisHub) across 4 domains: protein engineering, drug discovery, single-cell omics, and biomedical imaging.

**Task format**: End-to-end ML pipeline tasks. Agents must parse task descriptions, build computational pipelines, implement models, and submit predictions. Approximately 24 tasks total (6 protein engineering, 9 drug discovery, plus single-cell and imaging tasks).

**Key metrics**: Domain-specific ML metrics (AUROC, AUPRC, MAE, Pearson/Spearman correlation), leaderboard percentile ranking, and completion rate. A key finding: biomedical specialization conferred no consistent advantage over generalist agents — agents employing more diverse ML strategies scored highest.

**BioAgentEval suite**: `bioml_bench.yaml` contains 3 representative tasks covering drug response prediction, protein classification, and biomarker feature selection. Graders:
- `code_valid` — validates ML pipeline code
- `numeric_tolerance` — checks ML metric values (RMSE, F1) within tolerance
- `set_similarity` — compares predicted feature/gene sets against expected sets (Jaccard coefficient)
- `keyword_coverage` — verifies methodology terms
- `model` grader for pipeline quality

---

### SpatialBench

**Source**: Workman, Yang, Muralidharan, Le (LatchBio), 2025. [arXiv:2512.21907](https://arxiv.org/abs/2512.21907).

**What it is**: A benchmark of 146 verifiable problems testing agent ability to analyze real-world spatial transcriptomics data. Covers 5 technology platforms (Vizgen MERFISH, Takara Seeker, 10x Visium, 10x Xenium, Atlasxomics DBIT-seq) and 7 analysis categories.

**Task format**: Each problem provides an AnnData snapshot captured before an analysis decision point, a natural-language task description, and a deterministic grader. Agents must write and execute analysis code against the data.

**Analysis categories**: QC, Normalization, Dimensionality Reduction, Clustering, Cell Typing, Differential Expression, Spatial Analysis.

**Key metrics**: Deterministic pass/fail grading; accuracy (%). Base model accuracy ranges 20-38%. Platform choice affects accuracy as much as model choice (15-20 pp swings).

**BioAgentEval suite**: `spatialbench.yaml` contains 3 representative tasks covering spatial domain identification, spatially variable genes, and cell-cell interaction analysis. Graders:
- `set_similarity` — compares predicted labels/gene sets against expected sets
- `code_valid` — validates analysis code
- `keyword_coverage` — verifies analytical methodology terms
- `model` grader for biological relevance

---

### scBench

**Source**: Workman, Yang, Muralidharan, Abdulali, Le (LatchBio), 2026. [arXiv:2602.09063](https://arxiv.org/abs/2602.09063).

**What it is**: A companion benchmark to SpatialBench, focusing on single-cell RNA-seq analysis. Contains 394 verifiable problems across 6 sequencing platforms (BD Rhapsody, 10x Chromium, CSGenetics, Illumina, MissionBio Tapestri, ParseBio) and 7 analysis categories.

**Task format**: Same design as SpatialBench — AnnData snapshots with natural-language tasks and deterministic graders producing pass/fail results. Structured JSON output.

**Grader types in the original benchmark**: NumericTolerance, MultipleChoice, MarkerGenePrecisionRecall, LabelSetJaccard, DistributionComparison.

**Key metrics**: Accuracy (%). Claude Opus 4.6 leads at 52.8%, followed by Claude Opus 4.5 (49.9%), GPT-5.2 (45.2%). Platform choice dramatically affects accuracy (59.1% CSGenetics to 26.4% MissionBio — a 32.7 pp gap exceeding model variation). Normalization is easiest (84%), Differential Expression hardest (41%).

**BioAgentEval suite**: `scbench.yaml` contains 3 representative tasks covering cell type annotation, trajectory inference, and batch correction. Graders:
- `set_similarity` — compares predicted cell types/lineage labels against expected sets
- `code_valid` — validates analysis code
- `numeric_tolerance` — checks quantitative integration metrics
- `keyword_coverage` — verifies analytical terms
- `model` grader for biological correctness

---

### BioAgent Bench

**Source**: Fa, Culjak, Pandza, Cupic, 2026. [arXiv:2601.21800](https://arxiv.org/html/2601.21800v1).

**What it is**: A benchmark of 10 curated end-to-end bioinformatics pipeline tasks. Agents must complete multi-step computational workflows from raw data to final results, testing tool orchestration, error recovery, and robustness.

**Tasks**: Alzheimer's comparative pathway analysis, co-evolving gene clusters, Mendelian variant identification, RNA-seq differential expression, experimental evolution variant calling, GIAB variant calling, metagenomics community comparison, scRNA-seq analysis, transcript quantification, and viral species identification.

**Robustness testing**: Unique among the benchmarks, BioAgent Bench includes perturbation tests: corrupted inputs (agents must detect and avoid), decoy files (irrelevant sequences to exclude), and prompt bloat (non-essential background text).

**Key metrics**: Completion rate, steps completed, F1-score (categorical), Pearson correlation (numerical). Grading uses LLM-based evaluation against rubrics, prioritizing pipeline completion over numerical accuracy. Claude Opus 4.5 achieves 100% completion; open-weight models range 65-82.5%.

**BioAgentEval suite**: `bioagent_bench.yaml` contains 3 representative tasks covering RNA-seq pipeline, variant calling pipeline, and metagenomics pipeline. Graders:
- `tool_calls` — verifies correct tool orchestration (FastQC, alignment, quantification, etc.)
- `code_valid` — validates pipeline code
- `numeric_tolerance` — checks quantitative results (e.g., Ti/Tv ratio)
- `keyword_coverage` — verifies methodology coverage
- `model` grader for pipeline completeness and correctness

---

## Grader Type Reference

The following grader types are used across benchmark suites. All are implemented in `CodeGrader` and dispatched based on `expected_output.type`.

| Grader Type | Description | Common Use |
|---|---|---|
| `mcq_answer` | Checks if agent selected the correct MCQ letter (A-E) | HLE, LAB-Bench |
| `exact_match` | Normalized string comparison with case/whitespace options | FrontierScience Olympiad |
| `numeric_tolerance` | Numeric answer within absolute or relative tolerance | FrontierScience, BioML-Bench, scBench |
| `set_similarity` | Jaccard coefficient between predicted and expected sets | SpatialBench, scBench, BioML-Bench |
| `keyword_coverage` | Fraction of required keywords present in outcome | FrontierScience Research, all agentic |
| `code_valid` | Validates Python code syntax via `ast.parse()` | BixBench, BioML-Bench, all agentic |
| `tool_calls` | Verifies expected tool calls appear in transcript | Biomni-Eval1, BioAgent Bench |
| `groundedness` | Checks for citations/references in outcome | Biomni-Eval1 (literature review) |
| `model` | LLM-based rubric scoring (Anthropic or OpenAI) | All suites |
| `entities` | Checks for expected entity mentions | Biomedical core |
| `cypher_patterns` | Regex matching on Cypher queries in transcript | Biomedical core |

---

## Adding New Benchmark Tasks

To add tasks from a new benchmark or extend an existing suite:

1. Create a YAML file in `tasks/` following the established format:
   ```yaml
   name: my_benchmark
   description: Brief description of the benchmark
   eval_type: capability  # or "regression"
   default_num_trials: 3
   default_graders:
     - type: code
     - type: model
       rubric: Default rubric for all tasks
   tasks:
     - id: unique_task_id
       question: The question or task description
       expected_output:
         - type: mcq_answer  # or other grader type
           value: B
       graders:
         - type: code
         - type: model
           rubric: Task-specific rubric
       tags:
         benchmark: My Benchmark
         category: Subcategory
         subject: Domain
         answer_type: mcq
         difficulty: medium
   ```

2. Add validation tests in `tests/` to verify loading and grading.

3. Run `pytest` to confirm no regressions.

**Tag conventions**: All benchmark tasks should include at minimum: `benchmark` (source benchmark name), `category` (subcategory), `subject` (scientific domain), and `answer_type` (mcq, short_answer, free_text, code, pipeline, etc.).

---

## References

- Phan et al. (2025). "Humanity's Last Exam." [arXiv:2501.14249](https://arxiv.org/abs/2501.14249)
- OpenAI (2025). "FrontierScience." [arXiv:2601.21165](https://arxiv.org/abs/2601.21165)
- Laurent et al. (2024). "LAB-Bench." [arXiv:2407.10362](https://arxiv.org/abs/2407.10362)
- Huang et al. (2025). "Biomni." [bioRxiv:2025.05.30.656746](https://www.biorxiv.org/content/10.1101/2025.05.30.656746v1)
- Mitchener et al. (2025). "BixBench." [arXiv:2503.00096](https://arxiv.org/abs/2503.00096)
- Miller et al. (2025). "BioML-Bench." [bioRxiv:2025.09.01.673319](https://www.biorxiv.org/content/10.1101/2025.09.01.673319v2)
- Workman et al. (2025). "SpatialBench." [arXiv:2512.21907](https://arxiv.org/abs/2512.21907)
- Workman et al. (2026). "scBench." [arXiv:2602.09063](https://arxiv.org/abs/2602.09063)
- Fa et al. (2026). "BioAgent Bench." [arXiv:2601.21800](https://arxiv.org/html/2601.21800v1)
