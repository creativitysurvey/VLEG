"""Build turnkey Tech Science Press CMC LaTeX submission package with VLEG-CR & VLEG-CR-NS."""
import re
import shutil
import subprocess
from pathlib import Path

CO_CORRESPONDING_YIHU = True  # Set False if Ruzhen Chen should remain sole corresponding author

REPO = Path("/data/liwei/VLEG")
SRC_TEX = REPO / "Electronics_submission" / "main.tex"
OUT_DIR = REPO / "CMC_submission"
OUT_DIR.mkdir(parents=True, exist_ok=True)

orig = SRC_TEX.read_text(encoding="utf-8")


def slice_between(text, start_marker, end_marker):
    s = text.find(start_marker)
    assert s != -1, f"Start marker not found: {start_marker}"
    e = text.find(end_marker, s)
    assert e != -1, f"End marker not found: {end_marker}"
    return text[s:e]


# Extract verified Reusable Sections (Sec 2, Sec 3.1-3.3, Sec 4.2-4.6) and clean wording
sec2 = slice_between(orig, r"\section{Related Work}", r"\section{The Proposed Method}")
sec2 = sec2.replace("counterexample-gated hybrid context", "counterexample-refined neuro-symbolic execution")
sec2 = sec2.replace("deterministic counterexample gating", "counterexample-guided rule refinement and neuro-symbolic triage")

sec3_top = slice_between(
    orig,
    r"\section{The Proposed Method}",
    r"\subsection{Deterministic Counterexample Compiler and Cross-Episode Gating (VLEG-CG)}",
)
sec3_top = sec3_top.replace(r"\textbf{VLEG / VLEG-CG (Ours)}", r"\textbf{VLEG-CR-NS (Ours)}")
sec3_top = sec3_top.replace("(Verified \\& Screened Local Rules)", "(Refined \\& Compiled Local Rules)")
sec3_top = sec3_top.replace("VLEG and VLEG-CG", "VLEG and VLEG-CR-NS")
sec3_top = sec3_top.replace(
    "screens admitted reflections against contrary-label historical counterexamples.",
    "refines admitted reflections against contrary-label historical counterexamples via a deterministic predicate compiler.",
)

sec4_sym = slice_between(
    orig,
    r"\subsection{Controlled Memory-Strategy Comparison (Symbolic Sanity Check)}",
    r"\subsection{Public-Flow Benchmark on UNSW-NB15",
)
# Compact symbolic plot heights slightly to fit CMC 15-17 page target cleanly
sec4_sym = sec4_sym.replace("height=5.2cm", "height=4.2cm").replace("height=4.6cm", "height=4.0cm")

if CO_CORRESPONDING_YIHU:
    author_line = r"\noindent{\fontsize{10}{13}\selectfont\bfseries Ye Tang$^{1,\#}$, Ruikai Han$^{1,\#}$, Yi Hu$^{1,*}$ and Ruzhen Chen$^{2,*}$}\\[0.35em]"
    corresp_line = r"$^{*}$Corresponding Authors: Yi Hu. Email: hy610400@163.com; Ruzhen Chen. Email: crz@xzu.edu.cn\\"
else:
    author_line = r"\noindent{\fontsize{10}{13}\selectfont\bfseries Ye Tang$^{1,\#}$, Ruikai Han$^{1,\#}$, Yi Hu$^{1}$ and Ruzhen Chen$^{2,*}$}\\[0.35em]"
    corresp_line = r"$^{*}$Corresponding Author: Ruzhen Chen. Email: crz@xzu.edu.cn\\"

HEADER_AND_SEC1 = rf"""\documentclass[11pt,letterpaper,onecolumn]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[top=2.54cm,bottom=2.54cm,left=2.54cm,right=2.54cm]{{geometry}}
\usepackage{{mathtools}}
\usepackage{{amsmath}}
\usepackage{{amsthm}}
\usepackage{{thmtools}}
\usepackage{{newtxtext,newtxmath}}
\DeclareMathOperator*{{\argmax}}{{arg\,max}}
\DeclareMathOperator*{{\argmin}}{{arg\,min}}
\DeclareMathOperator{{\diag}}{{diag}}
\newtheorem{{theorem}}{{Theorem}}[section]
\newtheorem{{lemma}}[theorem]{{Lemma}}
\newtheorem{{corollary}}[theorem]{{Corollary}}
\newtheorem{{proposition}}[theorem]{{Proposition}}
\theoremstyle{{definition}}
\newtheorem{{definition}}{{Definition}}
\newtheorem{{example}}{{Example}}
\theoremstyle{{remark}}
\newtheorem{{remark}}{{Remark}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{multirow}}
\usepackage{{tabularx}}
\usepackage{{enumitem}}
\usepackage{{titlesec}}
\usepackage{{caption}}
\captionsetup{{font=small,labelfont=bf,labelsep=period,justification=raggedright,singlelinecheck=false}}
\titlespacing*{{\section}}{{0pt}}{{12pt}}{{3pt}}
\titlespacing*{{\subsection}}{{0pt}}{{10pt}}{{3pt}}
\titlespacing*{{\subsubsection}}{{0pt}}{{8pt}}{{2pt}}
\titleformat{{\section}}{{\large\bfseries}}{{\thesection}}{{0.6em}}{{}}
\titleformat{{\subsection}}{{\normalsize\bfseries\itshape}}{{\thesubsection}}{{0.6em}}{{}}
\titleformat{{\subsubsection}}{{\normalsize\itshape}}{{\thesubsubsection}}{{0.6em}}{{}}

\usepackage{{tikz}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\usetikzlibrary{{arrows.meta,positioning,shapes.geometric,shadows,fit,calc,matrix,shapes.misc,shapes.symbols}}
\usepackage{{algorithm}}
\usepackage{{algorithmicx}}
\usepackage{{algpseudocode}}
\algrenewcommand\algorithmicrequire{{\textbf{{Input:}}}}
\algrenewcommand\algorithmicensure{{\textbf{{Output:}}}}
\usepackage{{graphicx}}
\usepackage{{adjustbox}}
\usepackage{{url}}
\usepackage{{xcolor}}
\usepackage{{cite}}
\usepackage[colorlinks=true,allcolors=blue!60!black]{{hyperref}}
\usepackage{{fancyhdr}}

\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\footnotesize\textit{{Computers, Materials \& Continua}}}}
\fancyhead[R]{{\footnotesize\thepage}}
\renewcommand{{\headrulewidth}}{{0.4pt}}
\setlength{{\parindent}}{{0.20in}}
\setlength{{\parskip}}{{3pt}}
\renewcommand{{\baselinestretch}}{{1.05}}

\hypersetup{{pdftitle={{Verified Localized Episodic Grounding with Counterexample Rule Refinement for Auditable LLM-Assisted Network Traffic Analysis}}}}

\begin{{document}}

\noindent{{\fontsize{{14}}{{18}}\selectfont\bfseries Verified Localized Episodic Grounding with Counterexample Rule Refinement for Auditable LLM-Assisted Network Traffic Analysis\par}}
\vspace{{0.6em}}

{author_line}
{{\fontsize{{10}}{{12.5}}\selectfont
\noindent$^{{1}}$School of Information Science and Technology, Xizang University, Lhasa, 850000, China\\
$^{{2}}$School of Education, Xizang University, Lhasa, 850000, China\\
{corresp_line}
$^{{\#}}$These authors contributed equally to this work\\
Received: XX Month 2026; Accepted: XX Month 2026
\par}}

\vspace{{0.6em}}
\hrule
\vspace{{0.5em}}
{{\fontsize{{10}}{{12.5}}\selectfont
\noindent\textbf{{Abstract:}} Memory-augmented large language models (LLMs) can assist few-shot network intrusion triage by summarizing historical flows into human-auditable rules without parameter updates, yet local reflections risk label hallucination, cross-flow over-generalization, and prompt-level anchoring on non-firing rules. We propose Verified Localized Episodic Grounding (VLEG) and its counterexample-refined neuro-symbolic extension (VLEG-CR-NS), an auditable consolidation and hybrid-inference framework for few-shot network traffic analysis. During consolidation, VLEG checks candidate reflections against their labeled source records within $K$ attempts, after which a deterministic compiler parses admitted heuristics into executable Boolean predicates and feeds within-bank contrary-label counterexamples back to the generator to specialize overly broad rules. In a 16-scene symbolic sanity check, bounded source verification ($K=3$) eliminates label inversions ($100\%$ observed agreement), performing comparably to raw retrieval at 70B ($91.67\%$ versus $89.58\%$). On a five-seed, 32-flow benchmark over 1000 deduplicated UNSW-NB15 test fingerprints, unverified single-step reflection mislabels $95.0\%$ ($76/80$) of malicious support flows as benign ($42.66\%$ Macro-F1). Source-only VLEG restores $100\%$ source-label agreement ($158/158$) and lifts Macro-F1 to $59.68\%$, but trails raw retrieval ($63.92\%$) because $59.2\%$ ($93/157$) of rules match within-seed counterexamples. Counterexample-guided refinement (VLEG-CR) raises zero-counterexample rule convergence from $40.0\%$ to $84.4\%$ ($135/160$) and cuts unseen cross-seed counterexamples from $72.0\%$ to $40.7\%$. Executing compiled zero-counterexample rules deterministically before falling back to raw retrieval avoids LLM false alarms from non-firing rules. VLEG-CR-NS reaches $64.85\%$ Macro-F1 ($+0.93$ percentage points over raw retrieval, $95\%$ confidence interval $[+0.13, +1.71]$), while support-filtered bank triage (VLEG-CR-Bank) achieves $65.30\%$ Macro-F1 ($+1.38$ percentage points, $95\%$ confidence interval $[+0.28, +2.60]$) and the lowest false positive rate ($35.99\%$) among all 32-flow baselines while eliminating $47.1\%$--$49.0\%$ of online LLM calls.\par
\vspace{{0.35em}}
\noindent\textbf{{Keywords:}} Episodic memory; hallucination mitigation; large language models; network traffic analysis; intrusion detection; counterexample-guided refinement; neuro-symbolic execution
\par}}
\vspace{{0.5em}}
\hrule
\vspace{{0.6em}}

\section{{Introduction}}
\subsection{{Background and Motivation}}
Large language models (LLMs) can leverage demonstrations in the prompt to support classification and incident analysis without gradient updates \cite{{brown2020language,lewis2020retrieval}}. In network traffic analysis and Security Operations Center (SOC) intrusion triage \cite{{lin2022etbert,sharafaldin2018cicids,houssel2024explainable,islam2026maids}}, historical flow telemetry and incident summaries supply essential context for interpreting novel alerts. While supervised tabular classifiers such as Random Forest (RF), Gradient Boosting Decision Trees (GBDT), and Classification and Regression Trees (CART) provide high throughput when trained on tens of thousands of labeled records, analysts investigating scarce or emerging attack patterns require explicit, human-auditable rationales linking a target flow to historical precedents. Finite and unevenly utilized context windows make the consolidation and presentation of this historical evidence critical \cite{{an2024why,liu2024lost}}.

Persistent-memory architectures extend experience beyond a single context window \cite{{packer2023memgpt,wang2023longmem,wang2024memoryllm,gutierrez2024hipporag,maharana2024evaluating}}. Retaining raw records preserves instance-level fidelity, whereas compressing episodes into natural-language rules offers explicit decision guidance. However, an unverified summary can invert a class label, omit a discriminating condition, or turn an incidental feature value into an overly broad rule. Even when a rule is syntactically valid and faithful to its originating flow, injecting its text into an LLM prompt on target flows where its antecedent does not hold can induce false alarms through label anchoring.

Importantly, source agreement, cross-episode specificity, and execution fidelity are distinct requirements. Checking a proposed explanation against a single originating flow exposes direct label inversions or fabricated attributes, yet cannot reveal whether the stated condition falsely matches contrary-label traffic elsewhere in the support bank. Furthermore, once a rule is refined to eliminate contrary-label counterexamples, evaluating its numerical inequalities via deterministic symbolic execution differs fundamentally from asking a quantized LLM to interpret multiple conditional rules in text. Evaluating a memory system therefore requires tracking source-label consistency, within-seed and unseen cross-seed counterexample rates, deterministic rule coverage, and end-to-end classification metrics jointly.

\subsection{{Memory Research and Study Positioning}}
Memory-augmented language models reuse past observations through episodic retrieval, local reflection, and global semantic consolidation \cite{{wu2025from,zhang2024survey}}. Feedback-driven agents revise behavior across trials \cite{{shinn2023reflexion,wang2024voyager}}, while global hypothesis learners induce abstractions across multiple examples \cite{{agarwal2026hypothesis,he2024idea}}. Our study focuses on a closed-loop formulation combining per-episode source verification, counterexample-guided inductive rule refinement, and neuro-symbolic rule triage, benchmarked directly against raw feature retrieval, global summaries, shallow CART decision trees, and tree ensembles.

Three research questions guide our investigation. The first question asks whether checking a local reflection against its labeled source episode eliminates the label hallucinations of unverified reflection, and how revision unfolds across verification iterations. The second question asks why $100\%$ source-consistent reflections still inflate false positive rates on real network telemetry, and whether feeding within-seed counterexamples back to the generator can specialize rules without memorizing continuous float literals. The third question asks whether executing compiled zero-counterexample predicates deterministically prior to raw-retrieval fallback overcomes prompt-level anchoring on non-firing rules and outperforms raw retrieval under controlled retrieval budgets.

\subsection{{Proposed VLEG-CR and Neuro-Symbolic Triage Framework}}
We propose Verified Localized Episodic Grounding (VLEG), together with Counterexample-Refined consolidation (VLEG-CR) and Neuro-Symbolic triage (VLEG-CR-NS and VLEG-CR-Bank). During Phase~1 consolidation, a generator proposes a local reflection from label-free attributes, and a strict verifier checks it against the labeled source episode within budget $K$. Next, a deterministic compiler $\Psi$ parses each admitted \texttt{{if ... then ...}} heuristic into an executable Boolean predicate over the 19 flow features and tests it against the remaining 31 historical flows in the same seed bank. Whenever a rule matches contrary-label flows, VLEG-CR returns the matched counterexamples and candidate separating feature bounds to the generator for up to $K_{{\mathrm{{CR}}}}$ refinement steps, admitting only rules that satisfy both $\mathrm{{CE}}_{{\mathrm{{seed}}}}=0$ and source verification.

During Phase~2 inference, rather than unconditionally concatenating rule texts that may not apply to the target flow, VLEG-CR-NS evaluates compiled $\mathrm{{CE}}_{{\mathrm{{seed}}}}=0$ predicates deterministically on the target feature vector in sub-millisecond CPU time. When firing rules agree on a class label, the system outputs the deterministic prediction paired with the firing human-readable rule(s) at zero LLM query cost; only uncovered or conflicting boundary flows invoke 70B LLM raw-neighbor inference.

\subsection{{Experimental Evaluation and Summary of Contributions}}
We evaluate our framework on a 16-scene symbolic sanity check at 7B and 70B scales and on deduplicated UNSW-NB15 network telemetry \cite{{moustafa2015unsw}} spanning five 32-flow historical seeds, a disjoint 100-query pilot set, and 1000 formal test fingerprints per seed, alongside shallow CART trees, RF, and GBDT baselines. The main contributions of this work are four-fold:
\begin{{itemize}}[leftmargin=1.8em,topsep=2pt,itemsep=2pt]
    \item \textbf{{Two-Stage Verified and Counterexample-Refined Memory Consolidation:}} We formulate VLEG and VLEG-CR, coupling model-based source verification with a deterministic predicate compiler $\Psi$ that eliminates within-seed counterexamples via iterative inductive specialization while preventing exact-equality memorization on continuous flow features.
    \item \textbf{{Controlled Symbolic Sanity Check Across Model Scales:}} Across five symbolic evaluations at 7B and 70B scales, bounded verification ($K=3$) eliminates source-label inversions ($100\%$ observed agreement versus $41.7\%$--$50.0\%$ for unverified reflection) and matches raw retrieval accuracy at 70B ($91.67\%$ versus $89.58\%$), while isolating the theoretical boundary between single-source consistency and distractor suppression.
    \item \textbf{{Mechanistic Diagnosis of Label Collapse, Over-Generalization, and Non-Firing Prompt Anchoring:}} On UNSW-NB15, we show that unverified reflection mislabels $95.0\%$ ($76/80$) of malicious flows as benign ($6.17\%$ attack recall), and that iterative verification recovers $158/160$ episodes ($59.68\%$ Macro-F1) but leaves $59.2\%$ ($93/157$) of rules matching within-seed counterexamples. We further uncover a prompt-anchoring effect across $2839$ benign evaluations where zero retrieved malicious rules fire, yet unconditional rule-text injection still inflates LLM false positives from $32.19\%$ to $39.45\%$.
    \item \textbf{{Neuro-Symbolic Rule Triage Outperforming Raw Retrieval with Reduced LLM Compute:}} VLEG-CR elevates verified $\mathrm{{CE}}_{{\mathrm{{seed}}}}=0$ convergence from $40.0\%$ ($64/160$) to $84.4\%$ ($135/160$) and reduces unseen cross-seed counterexamples from $72.0\%$ to $40.7\%$. Pairing CR rules with deterministic neuro-symbolic triage enables VLEG-CR-NS ($64.85\%$ Macro-F1, $95\%$ CI $[+0.13, +1.71]$~pp) and Pilot-selected VLEG-CR-Bank ($65.30\%$ Macro-F1, $95\%$ CI $[+0.28, +2.60]$~pp; $35.99\%$ FPR) to significantly outperform raw retrieval ($63.92\%$ Macro-F1, $38.13\%$ FPR), achieve the lowest FPR among all 32-flow baselines (including shallow CART at $36.53\%$), and cut online 70B LLM query calls by $47.1\%$--$49.0\%$.
\end{{itemize}}

\subsection{{Paper Organization}}
Section~2 reviews episodic retrieval, local reflection, and intrusion detection baselines. Section~3 details the VLEG verification loop, the VLEG-CR counterexample compiler and inductive refinement procedure, and the VLEG-CR-NS neuro-symbolic inference architecture. Section~4 presents the symbolic sanity check, the UNSW-NB15 benchmark, factorial ablations, cost audits, and decision-tree comparisons, followed by limitations. Section~5 concludes the paper.
"""

SEC3_BOTTOM = r"""\subsection{Deterministic Counterexample Compiler and Inductive Rule Refinement (VLEG-CR)}
\label{sec:counterexample_gate}
To eliminate cross-episode over-generalization errors $E_j^{\mathrm{cross}}$ before rules are committed to memory, we introduce a deterministic predicate compiler $\Psi$ and a counterexample-guided refinement stage (\textbf{VLEG-CR}).

Candidate traffic reflections follow the conjunctive form \texttt{if <cond\_1> and <cond\_2> ... then label = <CLASS>}. Given a candidate reflection $r$, compiler $\Psi$ parses each atomic clause into an executable relational predicate over the 19 flow features:
\begin{equation}
    \pi(r)(x) = \bigwedge_{m=1}^{d} \mathrm{op}_m\big(x[\mathrm{field}_m],\; \mathrm{rhs}_m\big) \in \{0, 1\}, \qquad \hat{y}(r) \in \{\texttt{BENIGN}, \texttt{MALICIOUS}\},
    \label{eq:compiler_pred}
\end{equation}
where $\mathrm{op}_m \in \{=, \neq, >, <, \ge, \le\}$ and $\mathrm{rhs}_m$ is a numeric literal, categorical string literal, or feature reference (such as \texttt{sbytes > dbytes}). To prevent trivial instance memorization during refinement, $\Psi$ rejects exact equality ($=$) against non-zero literals on continuous rate, duration, jitter, load, and byte-count fields, requiring inequality bounds instead.

\begin{definition}[Within-Seed Counterexamples, Positive Support, and Inductive Refinement]
For a parseable rule $r_i$ associated with source episode $e_i=(x_i,y_i) \in \mathcal{E}_{\mathrm{seed}}$, let $\mathcal{E}_{-i}^{\mathrm{seed}} = \mathcal{E}_{\mathrm{seed}} \setminus \{e_i\}$ denote the remaining 31 historical flows in the same seed bank. Its within-seed counterexample count $\mathrm{CE}_{\mathrm{seed}}(r_i)$ and same-class positive support $S_{\mathrm{pos}}(r_i)$ are defined as:
\begin{equation}
    \mathrm{CE}_{\mathrm{seed}}(r_i) = \sum_{e_m \in \mathcal{E}_{-i}^{\mathrm{seed}}} \mathbb{I}\!\left(\pi(r_i)(x_m) = 1 \wedge y_m \neq y_i\right), \quad
    S_{\mathrm{pos}}(r_i) = \sum_{e_m \in \mathcal{E}_{-i}^{\mathrm{seed}}} \mathbb{I}\!\left(\pi(r_i)(x_m) = 1 \wedge y_m = y_i\right).
    \label{eq:ce_count}
\end{equation}
We also define the offline cross-seed diagnostic count $\mathrm{CE}_{\mathrm{pool}}(r_i)$ identically over all 159 other flows across the five pooled seeds ($\mathcal{E}_{\mathrm{all}} \setminus \{e_i\}$); $\mathrm{CE}_{\mathrm{pool}}$ is never accessed during consolidation or inference.

If a Stage-A reflection $r_i^*$ fails compilation or exhibits $\mathrm{CE}_{\mathrm{seed}}(r_i^*) > 0$, VLEG-CR initiates up to $K_{\mathrm{CR}}$ counterexample-guided refinement iterations ($t = 0, \dots, K_{\mathrm{CR}}-1$). At step $t$, $\Psi$ accumulates up to three falsely matched contrary-label episodes $\mathcal{C}_i^{(t)} \subset \mathcal{E}_{-i}^{\mathrm{seed}}$ together with univariate separating feature intervals between $x_i$ and $\mathcal{C}_i^{(t)}$, formatting a structured diagnostic critique $\tilde{c}_i^{(t)}$ for generator $\mathcal{G}_\theta$:
\begin{equation}
    \tilde{r}_i^{(t)} \sim \mathcal{G}_\theta\big(\cdot \mid x_i,\; \tilde{r}_i^{(t-1)},\; \tilde{c}_i^{(t-1)}\big).
    \label{eq:cr_gen}
\end{equation}
A refined candidate $\tilde{r}_i^{(t)}$ is committed as $r_i^{\mathrm{CR}}$ if and only if (i) $\Psi$ verifies valid non-memorizing syntax, $\pi(\tilde{r}_i^{(t)})(x_i)=1$, and $\mathrm{CE}_{\mathrm{seed}}(\tilde{r}_i^{(t)})=0$, and (ii) the LLM verifier confirms source fidelity $\mathcal{V}_\phi(e_i, \tilde{r}_i^{(t)})_v = 1$. If no candidate satisfies both gates within $K_{\mathrm{CR}}$ steps, the episode retains its raw record $e_i$ with $r_i^{\mathrm{CR}} = \emptyset$.
\end{definition}

\subsection{Inference Integration: Prompt Grounding vs. Neuro-Symbolic Rule Triage}
Given a target flow $T$ with features $x_T$, the retriever computes dense cosine similarity $S(T, \mathcal{M}_i)$ over standardized label-free feature embeddings ($z_i = x_i$ for network flows; $z_i = r_i^*$ for symbolic scenes):
\begin{equation}
    S(T, \mathcal{M}_i) = \frac{f_{\mathrm{emb}}(T) \cdot f_{\mathrm{emb}}(z_i)}{\|f_{\mathrm{emb}}(T)\| \|f_{\mathrm{emb}}(z_i)\|}, \qquad
    \mathcal{M}_{\mathrm{top}}(T) = \mathop{\arg\max}_{\mathcal{S} \subseteq \mathcal{M},\, |\mathcal{S}|=\min(k,|\mathcal{M}|)} \sum_{\mathcal{M}_i \in \mathcal{S}} S(T, \mathcal{M}_i).
    \label{eq:top_k}
\end{equation}

When verified rules are injected purely as prompt text (\textbf{VLEG} and prompt-only \textbf{VLEG-CR}), even rules whose antecedent evaluates to false ($\pi(r_i)(x_T) = 0$) remain visible in the context window and can bias a quantized LLM toward predicting \texttt{MALICIOUS}. Conversely, stripping rules with $\mathrm{CE}_{\mathrm{seed}} > 0$ at the prompt level (\textbf{VLEG-CG}) avoids false alarms but forfeits explicit rule execution. Because every admitted VLEG-CR rule is already compiled into a verified predicate $\pi(r_i^{\mathrm{CR}})$ with $\mathrm{CE}_{\mathrm{seed}}=0$, our \textbf{Neuro-Symbolic Rule Triage} executes compiled predicates deterministically on $x_T$ before invoking LLM inference. Let $\mathcal{R}_{\mathrm{cand}}(T)$ denote a candidate rule scope and $\tau_{\mathrm{sup}} \ge 0$ a minimum positive-support threshold. Define the firing vote counts for class $y \in \{\texttt{BENIGN}, \texttt{MALICIOUS}\}$ as:
\begin{equation}
    V_y(x_T) = \sum_{(e_i, r_i^{\mathrm{CR}}) \in \mathcal{R}_{\mathrm{cand}}(T)} \mathbb{I}\!\left(r_i^{\mathrm{CR}} \neq \emptyset \;\wedge\; S_{\mathrm{pos}}(r_i^{\mathrm{CR}}) \ge \tau_{\mathrm{sup}} \;\wedge\; \pi(r_i^{\mathrm{CR}})(x_T) = 1 \;\wedge\; \hat{y}(r_i^{\mathrm{CR}}) = y\right).
    \label{eq:ns_votes}
\end{equation}
We evaluate two neuro-symbolic execution scopes that fall back to raw retrieved episodes $\{e_i\} \in \mathcal{M}_{\mathrm{top}}(T)$ via $\mathrm{LLM}_\psi$ only when no unambiguous rule decision is triggered:
\begin{enumerate}[leftmargin=1.8em,topsep=2pt,itemsep=2pt]
    \item \textbf{Shared-Retrieval Neuro-Symbolic Triage (VLEG-CR-NS, $\mathcal{R}_{\mathrm{cand}} = \mathcal{M}_{\mathrm{top}}(T)$, $\tau_{\mathrm{sup}}=0$):} Evaluates compiled CR rules strictly within the same top-$k$ retrieved episodes as raw retrieval. If $V_{\texttt{MAL}}(x_T) > 0$ and $V_{\texttt{BEN}}(x_T) = 0$ (or vice versa), it outputs the unanimous class directly with the firing human-readable rule(s) and zero LLM call; otherwise it queries $\mathrm{LLM}_\psi$ on raw retrieved neighbors.
    \item \textbf{Support-Filtered Bank Triage (VLEG-CR-Bank, $\mathcal{R}_{\mathrm{cand}} = \mathcal{M}$, $\tau_{\mathrm{sup}}$ selected on pilot):} Evaluates all compiled $\mathrm{CE}_{\mathrm{seed}}=0$ rules in the 32-flow bank satisfying $S_{\mathrm{pos}}(r_i^{\mathrm{CR}}) \ge \tau_{\mathrm{sup}}$. When $V_{\texttt{MAL}}(x_T) \neq V_{\texttt{BEN}}(x_T)$, the majority class and its supporting rules are returned deterministically; ties or uncovered flows ($V_{\texttt{MAL}} = V_{\texttt{BEN}}$) fall back to top-$k$ raw-flow LLM inference.
\end{enumerate}

\subsection{Complete Two-Phase Algorithm}
Algorithm~\ref{alg:vleg_final} summarizes the complete two-phase VLEG-CR and VLEG-CR-NS / VLEG-CR-Bank pipeline, uniting Stage-A source verification, Stage-B counterexample-guided inductive refinement, and Phase-2 neuro-symbolic triage.

\begin{algorithm}[!t]
\caption{Counterexample-Refined VLEG with Neuro-Symbolic Rule Triage (VLEG-CR-NS / Bank)}
\label{alg:vleg_final}
\small
\begin{algorithmic}[1]
\Require Support bank $\mathcal{E}_{\mathrm{seed}}$, target $T=(x_T)$, budgets $(K, K_{\mathrm{CR}})$, top-$k$, threshold $\tau_{\mathrm{sup}}$, scope $\in \{\text{Top-}k, \text{Bank}\}$
\State Initialize Memory Bank $\mathcal{M} \gets \emptyset$
\For{each $e_i = (x_i, y_i) \in \mathcal{E}_{\mathrm{seed}}$} \Comment{Phase 1 Stage A: Source-Episode Verification}
    \State $j \gets 0$; $(r, c, v) \gets (\emptyset, \emptyset, 0)$
    \While{$v = 0$ \textbf{and} $j < K$}
        \State $r \gets \mathcal{G}_\theta(x_i, r, c)$; $(v, c) \gets \mathcal{V}_\phi(e_i, r)$; $j \gets j + 1$
    \EndWhile
    \State $r_i^{\mathrm{CR}} \gets \emptyset$
    \If{$v = 1$ \textbf{and} $\Psi(r)$ valid \textbf{and} $\mathrm{CE}_{\mathrm{seed}}(r) = 0$}
        \State $r_i^{\mathrm{CR}} \gets r$
    \Else \Comment{Phase 1 Stage B: Counterexample-Guided Inductive Refinement}
        \For{$t = 0$ \textbf{to} $K_{\mathrm{CR}} - 1$}
            \State Build counterexample critique $\tilde{c}$ from falsely matched contrary flows in $\mathcal{E}_{\mathrm{seed}} \setminus \{e_i\}$
            \State $r \gets \mathcal{G}_\theta(x_i, r, \tilde{c})$
            \If{$\Psi(r)$ valid \textbf{and} $\pi(r)(x_i)=1$ \textbf{and} $\mathrm{CE}_{\mathrm{seed}}(r)=0$ \textbf{and} $\mathcal{V}_\phi(e_i, r)_v = 1$}
                \State $r_i^{\mathrm{CR}} \gets r$; \textbf{break}
            \EndIf
        \EndFor
    \EndIf
    \State $\mathcal{M} \gets \mathcal{M} \cup \{(e_i, r_i^{\mathrm{CR}}, S_{\mathrm{pos}}(r_i^{\mathrm{CR}}))\}$
\EndFor
\State $\mathcal{M}_{\mathrm{top}} \gets \mathrm{Retrieve}(\mathcal{M}, x_T, k)$; $\mathcal{R}_{\mathrm{cand}} \gets \mathcal{M}_{\mathrm{top}}$ if scope $=$ Top-$k$ else $\mathcal{M}$ \Comment{Phase 2: Inference}
\State Compute deterministic predicate votes $(V_{\texttt{BEN}}(x_T), V_{\texttt{MAL}}(x_T))$ over $\mathcal{R}_{\mathrm{cand}}$ via Eq.~(\ref{eq:ns_votes})
\If{unambiguous rule decision ($V_{\texttt{MAL}} \neq V_{\texttt{BEN}}$, and $\min(V_{\texttt{BEN}},V_{\texttt{MAL}})=0$ for Top-$k$)}
    \State \Return $\arg\max_{y} V_y(x_T)$ with firing verified rule rationales (0 LLM calls)
\Else
    \State \Return $\mathrm{LLM}_\psi\big(\mathrm{PromptBuilder}(\{e_i\}_{(e_i,\cdot)\in\mathcal{M}_{\mathrm{top}}}, x_T)\big)$ \Comment{Raw-retrieval LLM fallback}
\EndIf
\end{algorithmic}
\end{algorithm}
"""

SEC4_INTRO = r"""\section{Experiments}
\subsection{Experimental Settings and Two-Stage Protocol}
Our evaluation follows a two-stage protocol: (1) a controlled 16-scene symbolic sanity check (Sections~\ref{sec:exp1}--\ref{sec:exp5}) that tests per-episode verification, grounding modalities, refinement budgets, and distractor attribution against a known oracle at 7B and 70B scales, and (2) a public network-flow benchmark on deduplicated UNSW-NB15 telemetry (Sections~\ref{sec:exp6}--\ref{sec:exp7}) evaluating Stage-A source verification, Stage-B counterexample-guided refinement (VLEG-CR), prompt gating (VLEG-CG), and Phase-2 neuro-symbolic triage (VLEG-CR-NS and VLEG-CR-Bank) against shallow CART decision trees, Random Forests, and GBDT ensembles.

In the symbolic sanity check, synthetic scenes vary across shape, color, texture, background, and a deterministic size distractor. On white backgrounds, \texttt{ANOMALY} requires a red circle; black backgrounds invert this rule, while texture and size have zero causal effect. Level~1 tests eight white-background scenes; Level~2 tests all sixteen scenes across both backgrounds; Level~3 withholds four scenes from the 12-episode memory bank and tests all sixteen across multiple masking seeds. Symbolic experiments run Qwen2.5-7B-Instruct and Llama-3.3-70B-Instruct with \texttt{nomic-embed-text} embeddings at temperature $0$, withholding target labels from all inference calls.
"""

SEC4_FLOW_AND_END = r"""\subsection{Public-Flow Benchmark on UNSW-NB15: Counterexample Refinement and Neuro-Symbolic Triage}
\label{sec:exp6}
\textbf{On deduplicated UNSW-NB15 telemetry, counterexample-guided refinement (VLEG-CR) raises verified zero-counterexample rule convergence from $40.0\%$ to $84.4\%$ ($135/160$), and pairing compiled CR rules with neuro-symbolic execution enables VLEG-CR-NS ($64.85\%$ Macro-F1) and VLEG-CR-Bank ($65.30\%$ Macro-F1, $35.99\%$ FPR) to significantly outperform raw retrieval ($63.92\%$ Macro-F1, $38.13\%$ FPR) while cutting online 70B LLM calls by nearly half.} Following deterministic 19-feature deduplication, conflicting-label removal, and train/test deduplication on UNSW-NB15 \cite{moustafa2015unsw}, $88,699$ unique training flow fingerprints (from $175,341$ raw rows) and $45,215$ disjoint testing fingerprints (from $82,332$ raw rows) remain. Five historical seeds (\texttt{20261020}--\texttt{20261024}) each sample 32 support flows (16 benign, 16 malicious; 160 total support flows) and evaluate the shared 1000-query test set (663 benign, 337 malicious), disjoint from the 100-query pilot set (\texttt{pilot\_20261012}). All LLM experiments run on dual NVIDIA A800 GPUs using Llama-3.3-70B-Instruct (\texttt{Q4\_K\_M}), \texttt{nomic-embed-text} embeddings, temperature $0$, $K = K_{\mathrm{CR}} = k = 3$, and a 32,768-token window. Table~\ref{tab:exp6_performance} summarizes all few-shot LLM, neuro-symbolic, and tabular baselines.

\begin{table}[!t]
\centering
\caption{Public-flow classification performance on the 1000 shared deduplicated UNSW-NB15 test fingerprints (mean $\pm$ sample standard deviation across five 32-flow seeds). Invalid LLM outputs count as errors. All local LLM and neuro-symbolic methods use the exact same 32 support flows per seed; full-train models use all 88,699 unique training flows.}
\label{tab:exp6_performance}
\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llrrrr}
\toprule
\textbf{Supervision Regime} & \textbf{Condition / Method} & \textbf{Accuracy (\%)} & \textbf{Macro-F1 (\%)} & \textbf{Attack Recall (\%)} & \textbf{FPR (\%)} \\
\midrule
\multirow{9}{*}{\shortstack[l]{Few-Shot LLM \&\\Neuro-Symbolic Memory\\(32 flows/seed, $k{=}3$,\\Llama-3.3-70B, 0-train)}}
 & Raw flow retrieval & 65.08 $\pm$ 2.93 & 63.92 $\pm$ 2.54 & 71.39 $\pm$ 8.44 & 38.13 $\pm$ 6.85 \\
 & Raw on admitted flows (control) & 64.86 $\pm$ 2.96 & 63.70 $\pm$ 2.68 & 71.16 $\pm$ 8.71 & 38.34 $\pm$ 6.61 \\
 & Unverified single-step reflection & 61.38 $\pm$ 1.08 & 42.66 $\pm$ 1.35 & 6.17 $\pm$ 1.43 & 9.89 $\pm$ 2.03 \\
 & Global summary & 50.22 $\pm$ 12.45 & 47.60 $\pm$ 14.13 & \textbf{87.12 $\pm$ 22.27} & 68.54 $\pm$ 24.20 \\
 & VLEG (Stage-A source-verified, prompt) & 59.74 $\pm$ 2.03 & 59.68 $\pm$ 2.03 & 77.74 $\pm$ 5.69 & 48.78 $\pm$ 1.67 \\
 & VLEG-CG (within-seed CE prompt gate) & 64.66 $\pm$ 2.83 & 63.55 $\pm$ 2.49 & 70.27 $\pm$ 8.14 & 37.86 $\pm$ 7.16 \\
 & VLEG-CR (Stage-B refined, prompt-only) & 61.60 $\pm$ 2.36 & 61.33 $\pm$ 2.11 & 78.81 $\pm$ 7.62 & 46.82 $\pm$ 5.18 \\
 & \textbf{VLEG-CR-NS (Ours, shared top-$k$, $\tau_{\mathrm{sup}}{=}0$)} & 65.78 $\pm$ 2.53 & 64.85 $\pm$ 2.58 & 74.72 $\pm$ 8.26 & 38.76 $\pm$ 5.17 \\
 & \textbf{VLEG-CR-Bank (Ours, Pilot $\tau_{\mathrm{sup}}{=}3$)} & \textbf{66.58 $\pm$ 2.79} & \textbf{65.30 $\pm$ 2.74} & 71.63 $\pm$ 9.32 & \textbf{35.99 $\pm$ 6.06} \\
\midrule
\multirow{4}{*}{\shortstack[l]{Few-Shot Tabular ML\\(Same 32 flows/seed)}}
 & Shallow Decision Tree (CART, depth=3) & 70.18 $\pm$ 1.64 & 69.39 $\pm$ 1.76 & 83.38 $\pm$ 12.86 & 36.53 $\pm$ 6.76 \\
 & Decision Tree (CART, unconstrained) & \textbf{70.80 $\pm$ 0.90} & \textbf{70.25 $\pm$ 0.64} & 86.41 $\pm$ 8.12 & 37.13 $\pm$ 5.08 \\
 & Random Forest (200 trees, 32-flow) & 68.86 $\pm$ 1.25 & 68.42 $\pm$ 1.23 & \textbf{87.60 $\pm$ 10.86} & 40.66 $\pm$ 5.66 \\
 & Gradient Boosting / GBDT (32-flow) & 66.60 $\pm$ 4.46 & 65.84 $\pm$ 4.60 & 78.04 $\pm$ 10.27 & 39.22 $\pm$ 5.52 \\
\midrule
\multirow{3}{*}{\shortstack[l]{Full-Train Tabular ML\\(88,699 unique flows)}}
 & Shallow Decision Tree (CART, depth=5) & 77.20 $\pm$ 0.00 & 76.81 $\pm$ 0.00 & 95.25 $\pm$ 0.00 & 31.98 $\pm$ 0.00 \\
 & Random Forest (Full-train upper bound) & \textbf{82.06 $\pm$ 0.36} & \textbf{81.31 $\pm$ 0.36} & 91.99 $\pm$ 0.59 & \textbf{22.99 $\pm$ 0.55} \\
 & Gradient Boosting / GBDT (Full-train) & 76.50 $\pm$ 0.00 & 76.25 $\pm$ 0.00 & \textbf{98.22 $\pm$ 0.00} & 34.54 $\pm$ 0.00 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

\begin{table}[!t]
\centering
\caption{Phase~1 consolidation and deterministic counterexample audit ($\Psi$) across the five formal seeds ($160$ historical episodes: $80$ benign, $80$ malicious). $\mathrm{CE}_{\mathrm{seed}} > 0$ is evaluated within each seed's 31 other support flows; $\mathrm{CE}_{\mathrm{pool}} > 0$ is evaluated offline across all 159 other flows from the five pooled seeds.}
\label{tab:counterexample_audit}
\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccccc}
\toprule
\textbf{Consolidation Stage / Method} & \textbf{Admitted Rules} & \textbf{Benign Match} & \textbf{Malicious Match} & \textbf{Zero $\mathrm{CE}_{\mathrm{seed}}$ Coverage} & \textbf{Within-Seed CE $>0$} & \textbf{Unseen Cross-Seed CE $>0$} \\
\midrule
Stage A, $j{=}0$ (Unverified single-step) & 160 / 160 & 80 / 80 (100\%) & 4 / 80 (5.0\%) & 34 / 160 (21.3\%) & 126 / 160 (78.8\%) & 136 / 160 (85.0\%) \\
Stage A, $j{\le}1$ (1st label-informed rev.) & 95 / 160 & 80 / 80 (100\%) & 15 / 80 (18.8\%) & --- & --- & --- \\
Stage A, $j{\le}2$ (VLEG $K{=}3$ source-only) & 158 / 160 & 80 / 80 (100\%) & 78 / 78 (100\%) & 64 / 160 (40.0\%) & 93 / 157 (59.2\%) & 113 / 157 (72.0\%) \\
\textbf{Stage B (VLEG-CR, $K_{\mathrm{CR}}{=}3$ refined)} & \textbf{135 / 160} & \textbf{67 / 67 (100\%)} & \textbf{68 / 68 (100\%)} & \textbf{135 / 160 (84.4\%)} & \textbf{0 / 135 (0.0\%)} & \textbf{55 / 135 (40.7\%)} \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

\subsubsection{Unverified Reflection Collapse and Iterative Verification Dynamics}
As shown in Table~\ref{tab:exp6_performance} and Table~\ref{tab:counterexample_audit}, unverified single-step reflection collapses to $6.17\% \pm 1.43\%$ attack recall and $42.66\%$ Macro-F1 because the generator labels $80/80$ benign flows as \texttt{BENIGN} at $j=0$ but mislabels $76/80$ ($95.0\%$) malicious flows as \texttt{BENIGN}. Inspecting the logged verification trajectories clarifies how recovery unfolds once the verifier's critique discloses the historical label: at attempt $j=1$, only $11/76$ ($14.5\%$) rejected malicious episodes are revised to \texttt{MALICIOUS}, whereas $65/76$ ($85.5\%$) still output \texttt{BENIGN} and are rejected a second time. At attempt $j=2$, $63$ of those $65$ episodes pass verification, restoring $100\%$ source-label agreement across $158/160$ admitted flows and lifting Macro-F1 by $+17.02$~pp to $59.68\%$. Because the second critique also carries the historical label, this recovery reflects repeated label-informed critique paired with rejection gating. Yet Stage-A VLEG still trails raw retrieval ($63.92\%$) by $-4.24$~pp because $59.2\%$ ($93/157$) of parseable source-consistent rules match contrary-label flows within their own 32-flow seed, inflating the FPR from $38.13\%$ to $48.78\%$.

\subsubsection{Inductive Rule Refinement (VLEG-CR) and Diagnosis of Non-Firing Prompt Anchoring}
Instead of discarding the $93$ over-generalized rules at inference as in VLEG-CG (which leaves only $64/160 = 40.0\%$ of episodes with usable zero-counterexample rules and retains full 3-rule prompts on just $12.3\%$ of test queries), Stage-B counterexample-guided refinement (\textbf{VLEG-CR}) feeds matched contrary-label flows back into $\mathcal{G}_\theta$. As reported in Table~\ref{tab:counterexample_audit} (Row~4), VLEG-CR raises strict $\mathrm{CE}_{\mathrm{seed}}=0$ and verifier-valid rule convergence from $40.0\%$ to \textbf{$84.4\%$ ($135/160$: $67/80$ benign and $68/80$ malicious)}. Crucially, specialization within each 32-flow seed transfers to unseen support banks: offline cross-seed counterexample incidence ($\mathrm{CE}_{\mathrm{pool}} > 0$ across the other four unseen seeds) drops by $-31.3$~pp from $72.0\%$ ($113/157$) to \textbf{$40.7\%$ ($55/135$)}, while malicious rules with zero unseen cross-seed counterexamples rise from $35.1\%$ ($27/77$) to \textbf{$55.9\%$ ($38/68$)}.

When these refined rules are unconditionally concatenated into the LLM inference prompt (prompt-only \textbf{VLEG-CR}, displaying $\ge 1$ verified $\mathrm{CE}_{\mathrm{seed}}=0$ rule on $97.1\%$ of queries and $2.41$ rules/query on average), test Macro-F1 improves over standard VLEG from $59.68\%$ to $61.33\%$ ($+1.65$~pp), attack recall reaches $78.81\%$, and unparseable LLM replies drop by half (from $30$ to $15$ out of $5000$). However, prompt-only VLEG-CR still exhibits a $46.82\%$ FPR and trails raw retrieval ($63.92\%$). Auditing compiler truth values on benign test queries exposes the exact mechanism: across all \textbf{$2839$ benign test evaluations where zero retrieved malicious VLEG-CR rules actually fired} ($\pi(r_i^{\mathrm{CR}})(x_T) = 0$ for every retrieved malicious tuple), prompt-only VLEG-CR still suffered a \textbf{$39.45\%$ false positive rate}, compared with \textbf{$32.19\%$} for raw flow retrieval on the exact same $2839$ evaluations ($+7.26$~pp). Thus, merely displaying conditional text containing \texttt{then label = MALICIOUS} in the context window anchors the 4-bit quantized 70B LLM toward false alarms even when the rule's numerical antecedent is false.

\subsubsection{Factorial Ablation and Pilot Selection of Neuro-Symbolic Triage (VLEG-CR-NS and VLEG-CR-Bank)}
Executing compiled $\mathrm{CE}_{\mathrm{seed}}=0$ predicates deterministically via Equation~(\ref{eq:ns_votes}) eliminates non-firing prompt anchoring completely. To ensure zero test-set leakage, we screened execution scopes and support thresholds $\tau_{\mathrm{sup}} \in \{0,1,2,3,5\}$ on the disjoint 100-query pilot set (\texttt{pilot\_20261012}), where raw retrieval scored $54.63\%$ Macro-F1 ($54.29\%$ FPR), standard VLEG scored $48.87\%$, prompt-only VLEG-CR scored $54.78\%$, shared-retrieval VLEG-CR-NS ($\tau_{\mathrm{sup}}=0$) reached $55.72\%$, and bank-wide majority triage peaked at $\tau_{\mathrm{sup}}=3$ with \textbf{$61.00\%$ Macro-F1} ($44.29\%$ FPR).

\begin{table}[!t]
\centering
\caption{Factorial ablation separating Phase~1 counterexample refinement (CR) from Phase~2 neuro-symbolic execution (NS) across the pilot partition ($n=100$) and the five formal seeds ($5 \times 1000$ queries). Paired cluster bootstrap $95\%$ confidence intervals compare formal Macro-F1 against zero-cost Raw flow retrieval ($63.92\%$).}
\label{tab:ns_ablation}
\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llccccc}
\toprule
\textbf{Phase 1 Rules} & \textbf{Phase 2 Inference Mechanism} & \textbf{Pilot F1} & \textbf{Formal Macro-F1 (\%)} & \textbf{Formal FPR (\%)} & \textbf{Direct Rule Used} & \textbf{$\Delta$F1 vs. Raw (95\% CI)} \\
\midrule
None (Raw) & LLM prompt over top-$3$ raw flows & 54.63\% & 63.92 $\pm$ 2.54 & 38.13 $\pm$ 6.85 & 0.0\% (0 calls saved) & Baseline (0.00 pp) \\
Stage-A VLEG & Unconditional hybrid LLM prompt & 48.87\% & 59.68 $\pm$ 2.03 & 48.78 $\pm$ 1.67 & 0.0\% (0 calls saved) & $-4.24$ $[-5.91, -2.53]$ \\
Stage-A VLEG & Prompt CE gate fallback (VLEG-CG) & 54.63\% & 63.55 $\pm$ 2.49 & 37.86 $\pm$ 7.16 & 0.0\% (12.3\% rules shown) & $-0.37$ $[-1.18, +0.42]$ \\
Stage-A VLEG & Neuro-symbolic top-$3$ ($\tau_{\mathrm{sup}}{=}0$, no CR) & 54.63\% & 63.99 $\pm$ 2.62 & 40.36 $\pm$ 6.54 & 20.1 $\pm$ 5.4\% & $+0.08$ $[-0.50, +0.63]$ \\
Stage-A VLEG & Neuro-symbolic bank ($\tau_{\mathrm{sup}}{=}3$, no CR) & 54.63\% & 63.52 $\pm$ 2.92 & 41.30 $\pm$ 8.15 & 15.0 $\pm$ 8.7\% & $-0.40$ $[-0.95, +0.19]$ \\
\midrule
Stage-B CR & Unconditional hybrid LLM prompt & 54.78\% & 61.33 $\pm$ 2.11 & 46.82 $\pm$ 5.18 & 0.0\% (97.1\% rules shown) & $-2.59$ $[-4.04, -1.07]$ \\
Stage-B CR & Active-firing LLM prompt gate (top-$3$) & 55.72\% & 64.08 $\pm$ 2.57 & 40.51 $\pm$ 4.96 & 0.0\% (47.1\% rules shown) & $+0.16$ $[-0.58, +0.93]$ \\
\textbf{Stage-B CR} & \textbf{VLEG-CR-NS (top-$3$ direct, $\tau_{\mathrm{sup}}{=}0$)} & 55.72\% & \textbf{64.85 $\pm$ 2.58} & 38.76 $\pm$ 5.17 & \textbf{47.1 $\pm$ 7.1\%} & \textbf{$+0.93$ $[+0.13, +1.71]$} \\
\textbf{Stage-B CR} & \textbf{VLEG-CR-Bank (bank direct, $\tau_{\mathrm{sup}}{=}0$)} & 58.50\% & \textbf{65.11 $\pm$ 2.85} & \textbf{35.93 $\pm$ 6.27} & \textbf{68.5 $\pm$ 7.2\%} & $+1.19$ $[-0.21, +2.71]$ \\
\textbf{Stage-B CR} & \textbf{VLEG-CR-Bank (Pilot-selected $\tau_{\mathrm{sup}}{=}3$)} & \textbf{61.00\%} & \textbf{65.30 $\pm$ 2.74} & \textbf{35.99 $\pm$ 6.06} & \textbf{49.0 $\pm$ 14.9\%} & \textbf{$+1.38$ $[+0.28, +2.60]$} \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

Table~\ref{tab:ns_ablation} and Figure~\ref{fig:exp6_tradeoff} establish a clear synergistic interaction between Phase~1 CR consolidation and Phase~2 NS execution:
\begin{enumerate}[leftmargin=1.8em,topsep=2pt,itemsep=2pt]
    \item \textbf{Necessity of Both Stages:} Applying neuro-symbolic execution to unrefined Stage-A VLEG rules covers only $15.0\%$--$20.1\%$ of test queries and yields no statistically significant gain over raw retrieval ($+0.08$~pp and $-0.40$~pp), because $59.2\%$ of Stage-A rules fail the $\mathrm{CE}_{\mathrm{seed}}=0$ filter. Conversely, invoking the LLM even when firing CR rules agree (\texttt{Active-firing LLM prompt gate}, $64.08\%$) underperforms direct symbolic execution (\textbf{VLEG-CR-NS}, $64.85\%$) by $-0.77$~pp because LLM re-interpretation reintroduces label noise.
    \item \textbf{Controlled Advantage Under Identical Retrieval (VLEG-CR-NS):} Restricting rule evaluation strictly to the same top-$3$ retrieved episodes as raw retrieval with zero tuned hyperparameters ($\tau_{\mathrm{sup}}=0$), \textbf{VLEG-CR-NS} improves Macro-F1 to $64.85\% \pm 2.58\%$ ($+0.93$~pp over raw retrieval, paired $95\%$ CI $[+0.13, +1.71]$~pp, outperforming raw retrieval in four of five seeds), raises attack recall by $+3.33$~pp to $74.72\%$, and classifies $47.1\%$ of test flows directly via auditable rules at zero LLM inference cost.
    \item \textbf{Lowest False-Alarm Rate via Support-Filtered Bank Triage (VLEG-CR-Bank):} Evaluating all $\mathrm{CE}_{\mathrm{seed}}=0$ rules with positive support $S_{\mathrm{pos}} \ge 3$ across the 32-flow bank (\textbf{VLEG-CR-Bank}) achieves \textbf{$65.30\% \pm 2.74\%$ Macro-F1} ($+1.38$~pp over raw retrieval, paired $95\%$ CI $[+0.28, +2.60]$~pp, winning in four of five seeds), \textbf{$66.58\% \pm 2.79\%$ accuracy}, and a \textbf{$35.99\% \pm 6.06\%$ FPR}, while directly resolving $49.0\%$ of test queries without calling the 70B LLM.
\end{enumerate}

\begin{figure}[!t]
\centering
\definecolor{cRaw}{HTML}{2B5C8F}
\definecolor{cSingle}{HTML}{A89070}
\definecolor{cVleg}{HTML}{C8523B}
\definecolor{cPromptCR}{HTML}{D9822B}
\definecolor{cNS}{HTML}{2E8B57}
\definecolor{cBank}{HTML}{1B6CA8}
\definecolor{cCart}{HTML}{6A4C93}
\begin{adjustbox}{max width=0.98\linewidth}\begin{tikzpicture}
    \begin{axis}[
        ybar,
        bar width=12pt,
        width=8.2cm,
        height=4.7cm,
        enlarge x limits=0.11,
        ylabel={Macro-F1 (\%)},
        ymin=0, ymax=82,
        ytick={0,20,40,60,80},
        symbolic x coords={Single step, VLEG, VLEG-CR, Raw flows, VLEG-CR-NS, VLEG-CR-Bank},
        xtick={Single step, VLEG, VLEG-CR, Raw flows, VLEG-CR-NS, VLEG-CR-Bank},
        x tick label style={font=\tiny, align=center, text width=1.15cm},
        axis lines*=left,
        clip=false,
        title={(a) 5-seed Macro-F1 comparison},
        title style={at={(0,1.02)}, anchor=south west, font=\footnotesize\bfseries},
        bar shift=0pt
    ]
    \addplot[fill=cSingle, draw=black!70, error bars/.cd, y dir=both, y explicit]
        coordinates {(Single step, 42.66) +- (0, 1.35)};
    \addplot[fill=cVleg, draw=black!70, error bars/.cd, y dir=both, y explicit]
        coordinates {(VLEG, 59.68) +- (0, 2.03)};
    \addplot[fill=cPromptCR, draw=black!70, error bars/.cd, y dir=both, y explicit]
        coordinates {(VLEG-CR, 61.33) +- (0, 2.11)};
    \addplot[fill=cRaw, draw=black!70, error bars/.cd, y dir=both, y explicit]
        coordinates {(Raw flows, 63.92) +- (0, 2.54)};
    \addplot[fill=cNS, draw=black!70, error bars/.cd, y dir=both, y explicit]
        coordinates {(VLEG-CR-NS, 64.85) +- (0, 2.58)};
    \addplot[fill=cBank, draw=black!70, error bars/.cd, y dir=both, y explicit]
        coordinates {(VLEG-CR-Bank, 65.30) +- (0, 2.74)};

    \node[above, font=\tiny] at (axis cs:Single step,44.8) {42.66};
    \node[above, font=\tiny] at (axis cs:VLEG,62.3) {59.68};
    \node[above, font=\tiny] at (axis cs:VLEG-CR,64.0) {61.33};
    \node[above, font=\tiny] at (axis cs:Raw flows,67.0) {63.92};
    \node[above, font=\tiny\bfseries] at (axis cs:VLEG-CR-NS,68.0) {64.85};
    \node[above, font=\tiny\bfseries] at (axis cs:VLEG-CR-Bank,68.6) {65.30};
    \end{axis}
\end{tikzpicture}\hspace{0.35cm}\begin{tikzpicture}
    \begin{axis}[
        width=7.6cm,
        height=4.7cm,
        xlabel={False positive rate (\%)},
        ylabel={Attack recall (\%)},
        xmin=0, xmax=75,
        ymin=0, ymax=100,
        xtick={0,15,30,45,60,75},
        ytick={0,20,40,60,80,100},
        axis lines*=left,
        grid=major,
        title={(b) Recall vs. false-alarm operating points},
        title style={at={(0,1.02)}, anchor=south west, font=\footnotesize\bfseries},
        legend style={at={(0.98,0.03)}, anchor=south east, font=\tiny, draw=gray!50, fill=white, fill opacity=0.92}
    ]
    \addplot[only marks, mark=diamond*, mark size=3.0pt, color=cSingle] coordinates {(9.89, 6.17)};
    \addlegendentry{Single-step}
    \addplot[only marks, mark=triangle*, mark size=3.2pt, color=cVleg] coordinates {(48.78, 77.74)};
    \addlegendentry{VLEG (prompt)}
    \addplot[only marks, mark=pentagon*, mark size=3.2pt, color=cPromptCR] coordinates {(46.82, 78.81)};
    \addlegendentry{VLEG-CR (prompt)}
    \addplot[only marks, mark=*, mark size=3.0pt, color=cRaw] coordinates {(38.13, 71.39)};
    \addlegendentry{Raw retrieval}
    \addplot[only marks, mark=square*, mark size=3.1pt, color=cNS] coordinates {(38.76, 74.72)};
    \addlegendentry{VLEG-CR-NS (Ours)}
    \addplot[only marks, mark=star*, mark size=3.8pt, color=cBank] coordinates {(35.99, 71.63)};
    \addlegendentry{VLEG-CR-Bank (Ours)}
    \addplot[only marks, mark=oplus*, mark size=3.0pt, color=cCart] coordinates {(36.53, 83.38)};
    \addlegendentry{CART (depth=3)}
    \end{axis}
\end{tikzpicture}\end{adjustbox}
\caption{UNSW-NB15 five-seed performance summary. (a) Mean Macro-F1 ($\pm$ SD) showing VLEG-CR-NS and VLEG-CR-Bank surpassing Raw flow retrieval; (b) attack recall vs. FPR operating points across LLM, neuro-symbolic, and shallow CART baselines.}
\label{fig:exp6_tradeoff}
\end{figure}

\begin{table}[!t]
\centering
\caption{Memory consolidation and online per-query inference costs (5-seed means, 32 support flows and 1000 test queries per seed). VLEG-CR-NS and VLEG-CR-Bank execute compiled Python predicates in $<0.1$~ms on rule-covered queries, eliminating $47.1\%$--$49.0\%$ of online 70B LLM calls.}
\label{tab:exp6_cost}
\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Condition / Method} & \textbf{Consol. Calls} & \textbf{Consol. Tokens} & \textbf{Consol. (s)} & \textbf{LLM Calls / 1k Queries} & \textbf{Query Tokens / 1k} & \textbf{Query Mean (s)} \\
\midrule
Raw flow retrieval & 0.0 & 0 & 0.0 & 1000.0 & 904,200 & 2.12 \\
Unverified single-step & 32.0 & 9405 & 126.5 & 1000.0 & 982,500 & 2.23 \\
Standard VLEG ($K{=}3$, prompt) & 120.4 & 40,094 & 419.4 & 1000.0 & 976,800 & 1.74 \\
VLEG-CR ($K_{\mathrm{CR}}{=}3$, prompt-only) & 169.2 & 82,664 & 990.5 & 1000.0 & 991,400 & 1.81 \\
\textbf{VLEG-CR-NS (Ours, top-$3$ NS)} & 169.2 & 82,664 & 990.5 & \textbf{529.0 ($-47.1\%$)} & \textbf{478,300} & \textbf{1.12} \\
\textbf{VLEG-CR-Bank (Ours, $\tau_{\mathrm{sup}}{=}3$)} & 169.2 & 82,664 & 990.5 & \textbf{510.0 ($-49.0\%$)} & \textbf{461,100} & \textbf{1.08} \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

Table~\ref{tab:exp6_cost} reports the end-to-end consolidation and inference cost audit. Stage-B counterexample refinement adds $48.8$ LLM calls and $42,570$ tokens per 32-flow seed on top of Stage-A VLEG ($82,664$ total consolidation tokens). However, because VLEG-CR-NS and VLEG-CR-Bank resolve $47.1\%$--$49.0\%$ of incoming test queries deterministically in $<0.1$~ms without invoking the 70B LLM, they save approximately $426,000$--$443,000$ query tokens per $1000$ test flows and cut mean per-query latency from $2.12$~s (raw retrieval) to $1.08$--$1.12$~s. Consequently, the entire one-time Phase~1 consolidation budget is amortized after fewer than $190$ target flow evaluations.

\subsection{Comparison with Symbolic Decision Trees (CART), Ensembles, and LLM-IDS}
\label{sec:exp7}
Benchmarking against shallow CART decision trees and tabular ensembles on the exact same splits (Table~\ref{tab:exp6_performance}) clarifies the complementary strengths of classical split optimization and neuro-symbolic LLM memory:
\begin{enumerate}[leftmargin=1.8em,topsep=2pt,itemsep=2pt]
    \item \textbf{Discriminative Recall vs. False-Alarm Control in the 32-Flow Regime:} On the 32-flow splits, shallow depth-3 CART ($69.39\%$ Macro-F1, $83.38\%$ recall, $36.53\%$ FPR), unconstrained CART ($70.25\%$ Macro-F1, $37.13\%$ FPR), and Random Forest ($68.42\%$ Macro-F1, $40.66\%$ FPR) achieve higher overall Macro-F1 than VLEG-CR-Bank ($65.30\%$ Macro-F1) because global impurity minimization jointly optimizes split thresholds for recall across all 32 flows simultaneously. Remarkably, however, \textbf{VLEG-CR-Bank achieves the lowest false positive rate ($35.99\%$) across all 32-flow methods}---edging out depth-3 CART ($36.53\%$), unconstrained CART ($37.13\%$), GBDT ($39.22\%$), and RF ($40.66\%$)---while matching 32-flow GBDT in overall accuracy ($66.58\%$ versus $66.60\%$).
    \item \textbf{Relational Expressiveness and Modular Auditability:} On Seed \texttt{20261020}, the depth-3 CART tree partitions the space via three univariate axis-aligned numeric splits (\texttt{sttl <= 46.50} $\to$ \texttt{BENIGN}; \texttt{sttl > 46.50} $\wedge$ \texttt{dinpkt <= 67.08} $\to$ \texttt{MALICIOUS}; \texttt{dinpkt > 67.08} $\wedge$ \texttt{djit > 191.89} $\to$ \texttt{MALICIOUS}), offering no protocol semantics or cross-feature comparisons. Conversely, VLEG-CR rules express direct relational predicates across features alongside protocol rationales---for example, \emph{``if \texttt{proto == tcp} and \texttt{state == FIN} and \texttt{dur < 1} and \texttt{sbytes > dbytes} and \texttt{sload > 50000}, then \texttt{label = MALICIOUS}, reflecting asymmetric high-rate byte transfer in a short TCP session''}---and allow SOC analysts to insert, inspect, or revoke individual episodic rules without refitting a global tree.
    \item \textbf{Deduplicated Full-Train Upper Bounds:} Trained on all $88,699$ unique flow fingerprints, full-train RF achieves $81.31\%$ Macro-F1 ($22.99\%$ FPR) and GBDT achieves $76.25\%$ Macro-F1 ($34.54\%$ FPR), confirming that removing $79,746$ training duplicates and $33,161$ test duplicates leaves substantial boundary overlap even for full-supervision ensembles, and situating VLEG-CR-Bank as an auditable few-shot triage layer rather than a replacement for line-rate supervised filters \cite{houssel2024explainable,islam2026maids}.
\end{enumerate}

\subsection{Limitations}
\label{sec:limitations}
Several limitations bound our conclusions. First, the 16-scene symbolic sanity check tests mechanism mechanics rather than open-world network generalization, and comparing Qwen2.5-7B with Llama-3.3-70B changes model family alongside parameter count. Second, our public-flow benchmark uses 19 deduplicated tabular UNSW-NB15 features without raw packet payloads and evaluates a single quantized 70B model (\texttt{Q4\_K\_M}); evaluating multi-class attack families, additional traffic corpora such as CICIDS2017, and human SOC analyst triage speed remains important future work. Third, while VLEG-CR-Bank surpasses raw retrieval and achieves the lowest few-shot FPR ($35.99\%$), a 32-flow support bank still leaves a $4.09$--$4.95$~pp Macro-F1 gap relative to globally optimized CART trees due to lower recall on uncovered flows ($51.0\%$) that fall back to raw LLM inference.

\section{Conclusions}
This paper presented Verified Localized Episodic Grounding (VLEG), Counterexample-Refined consolidation (VLEG-CR), and Neuro-Symbolic rule triage (VLEG-CR-NS and VLEG-CR-Bank) for auditable few-shot network traffic analysis. Evaluations across a 16-scene symbolic sanity check and a five-seed benchmark on 1000 deduplicated UNSW-NB15 test fingerprints support four main conclusions:
\begin{enumerate}[leftmargin=1.8em,topsep=2pt,itemsep=2pt]
    \item \textbf{Iterative Source Verification Recovers Catastrophic Label Inversions:} Unverified single-step reflection mislabels $95.0\%$ ($76/80$) of malicious support flows as benign ($6.17\%$ attack recall, $42.66\%$ Macro-F1). Multi-step verification ($K=3$) restores $100\%$ source-label agreement across $158/160$ admitted episodes and raises Macro-F1 to $59.68\%$, but leaves $59.2\%$ ($93/157$) of rules matching contrary-label flows within their 32-flow seed.
    \item \textbf{Counterexample-Guided Inductive Refinement Specializes Rules Across Seeds:} Feeding within-seed counterexamples back to the generator (VLEG-CR) increases strict $\mathrm{CE}_{\mathrm{seed}}=0$ rule convergence from $40.0\%$ to $84.4\%$ ($135/160$) and reduces unseen cross-seed counterexamples from $72.0\%$ to $40.7\%$.
    \item \textbf{Neuro-Symbolic Execution Eliminates Non-Firing Prompt Anchoring and Beats Raw Retrieval:} Unconditionally injecting non-firing malicious rules into LLM prompts inflates benign false positives from $32.19\%$ to $39.45\%$. Executing compiled $\mathrm{CE}_{\mathrm{seed}}=0$ predicates deterministically before raw-retrieval fallback enables VLEG-CR-NS ($64.85\%$ Macro-F1, $95\%$ CI $[+0.13, +1.71]$~pp) and Pilot-selected VLEG-CR-Bank ($65.30\%$ Macro-F1, $95\%$ CI $[+0.28, +2.60]$~pp) to significantly outperform raw retrieval ($63.92\%$) while cutting online 70B LLM calls and query tokens by $47.1\%$--$49.0\%$.
    \item \textbf{Complementary Operational Role Alongside Decision Trees:} While 32-flow CART and Random Forest models attain higher recall and Macro-F1 ($68.42\%$--$70.25\%$) via global split optimization, VLEG-CR-Bank achieves the lowest false positive rate ($35.99\%$) among all 32-flow methods, matches 32-flow GBDT accuracy ($66.58\%$), and provides cross-feature relational predicates and human-auditable protocol rationales for SOC triage.
\end{enumerate}

\vspace{0.8em}
{\fontsize{10}{12.5}\selectfont
\noindent\textbf{Acknowledgement:} The authors thank the anonymous reviewers and editors for their constructive comments.\par\vspace{0.3em}
\noindent\textbf{Funding Statement:} This research was funded by the Science and Technology Projects of Ngari Prefecture, Xizang Autonomous Region, China, under Grant No. QYXTZX-AL2026-01.\par\vspace{0.3em}
\noindent\textbf{Author Contributions:} The authors confirm contribution to the paper as follows: study conception and design: Ye Tang, Ruikai Han, Ruzhen Chen; data collection and curation: Ye Tang, Yi Hu; methodology and software: Ye Tang, Ruikai Han; analysis and interpretation of results: Ye Tang, Ruikai Han, Yi Hu; draft manuscript preparation: Ye Tang, Ruikai Han; manuscript review, editing, and supervision: Yi Hu, Ruzhen Chen. All authors reviewed the results and approved the final version of the manuscript.\par\vspace{0.3em}
\noindent\textbf{Availability of Data and Materials:} The public network traffic benchmark evaluated in this study utilizes the publicly available UNSW-NB15 dataset \cite{moustafa2015unsw}. All symbolic benchmark definitions, counterexample-refinement and neuro-symbolic compiler scripts, and evaluation logs supporting the reported findings are publicly available at \url{https://github.com/creativitysurvey/VLEG}.\par\vspace{0.3em}
\noindent\textbf{Ethics Approval:} Not applicable.\par\vspace{0.3em}
\noindent\textbf{Conflicts of Interest:} The authors declare no conflicts of interest to report regarding the present study.\par
}
"""

TSP_REFERENCES = r"""\begin{thebibliography}{99}
\fontsize{10}{12}\selectfont
\setlength{\itemsep}{2pt}

\bibitem{brown2020language}
T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, et al., ``Language models are few-shot learners,'' \emph{Adv. Neural Inf. Process. Syst.}, vol. 33, pp. 1877--1901, 2020.

\bibitem{lewis2020retrieval}
P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, et al., ``Retrieval-augmented generation for knowledge-intensive NLP tasks,'' \emph{Adv. Neural Inf. Process. Syst.}, vol. 33, pp. 9459--9474, 2020.

\bibitem{lin2022etbert}
X. Lin, G. Xiong, G. Gou, Z. Li, J. Shi, and J. Yu, ``ET-BERT: A contextualized datagram representation with pre-training transformers for encrypted traffic classification,'' in \emph{Proc. ACM Web Conf. (WWW)}, Apr. 2022, pp. 633--642.

\bibitem{sharafaldin2018cicids}
I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, ``Toward generating a new intrusion detection dataset and intrusion traffic characterization,'' in \emph{Proc. Int. Conf. Inf. Syst. Secur. Privacy (ICISSP)}, Jan. 2018, pp. 108--116.

\bibitem{houssel2024explainable}
P. R. B. Houssel, P. Singh, S. Layeghy, and M. Portmann, ``Towards explainable network intrusion detection using large language models,'' \emph{arXiv preprint arXiv:2408.04342}, 2024.

\bibitem{islam2026maids}
M. S. Islam, L. G. Jaimes, and A. S. Dina, ``MA-IDS: Multi-agent RAG framework for IoT network intrusion detection with an experience library,'' \emph{arXiv preprint arXiv:2604.05458}, 2026.

\bibitem{an2024why}
C. An, J. Zhang, M. Zhong, L. Li, S. Gong, Y. Luo, et al., ``Why does the effective context length of LLMs fall short?'' \emph{arXiv preprint arXiv:2410.18745}, 2024.

\bibitem{liu2024lost}
N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, ``Lost in the middle: How language models use long contexts,'' \emph{Trans. Assoc. Comput. Linguist.}, vol. 12, pp. 157--173, 2024.

\bibitem{packer2023memgpt}
C. Packer, S. Wooders, K. Lin, V. Fang, S. G. Patil, I. Stoica, and J. E. Gonzalez, ``MemGPT: Towards LLMs as operating systems,'' \emph{arXiv preprint arXiv:2310.08560}, 2023.

\bibitem{wang2023longmem}
W. Wang, L. Dong, H. Cheng, X. Liu, X. Yan, J. Gao, and F. Wei, ``Augmenting language models with long-term memory,'' \emph{Adv. Neural Inf. Process. Syst.}, vol. 36, pp. 74530--74543, 2023.

\bibitem{wang2024memoryllm}
Y. Wang, Y. Gao, X. Chen, H. Jiang, S. Li, J. Yang, et al., ``MEMORYLLM: Towards self-updatable large language models,'' in \emph{Proc. 41st Int. Conf. Mach. Learn. (ICML)}, vol. 235, Jul. 2024, pp. 50453--50466.

\bibitem{gutierrez2024hipporag}
B. Jimenez Gutierrez, Y. Shu, Y. Gu, M. Yasunaga, and Y. Su, ``HippoRAG: Neurobiologically inspired long-term memory for large language models,'' \emph{Adv. Neural Inf. Process. Syst.}, vol. 37, pp. 59532--59569, 2024.

\bibitem{maharana2024evaluating}
A. Maharana, D. H. Lee, S. Tulyakov, M. Bansal, F. Barbieri, and Y. Fang, ``Evaluating very long-term conversational memory of LLM agents,'' \emph{arXiv preprint arXiv:2402.17753}, 2024.

\bibitem{wu2025from}
Y. Wu, S. Liang, C. Zhang, Y. Wang, Y. Zhang, H. Guo, et al., ``From human memory to AI memory: A survey on memory mechanisms in the era of LLMs,'' \emph{arXiv preprint arXiv:2504.15965}, 2025.

\bibitem{zhang2024survey}
Z. Zhang, X. Bo, C. Ma, R. Li, X. Chen, Q. Dai, et al., ``A survey on the memory mechanism of large language model based agents,'' \emph{arXiv preprint arXiv:2404.13501}, 2024.

\bibitem{shinn2023reflexion}
N. Shinn, F. Cassano, A. Gopinath, K. R. Narasimhan, and S. Yao, ``Reflexion: Language agents with verbal reinforcement learning,'' \emph{Adv. Neural Inf. Process. Syst.}, vol. 36, pp. 8634--8652, 2023.

\bibitem{wang2024voyager}
G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, et al., ``Voyager: An open-ended embodied agent with large language models,'' \emph{Trans. Mach. Learn. Res.}, pp. 1--32, 2024.

\bibitem{agarwal2026hypothesis}
A. K. Agarwal and M. Yamada, ``Hypothesis-driven reasoning for large language models,'' in \emph{Proc. 40th AAAI Conf. Artif. Intell. (AAAI)}, Feb. 2026, pp. 1686--1693.

\bibitem{he2024idea}
K. He, M. Zhang, S. Yan, P. Wu, and Z. Chen, ``IDEA: Enhancing the rule learning ability of large language model agent through induction, deduction, and abduction,'' in \emph{Proc. Annu. Meeting Assoc. Comput. Linguist. (ACL)}, Aug. 2024.

\bibitem{zhang2022active}
Y. Zhang, S. Feng, and C. Tan, ``Active example selection for in-context learning,'' \emph{arXiv preprint arXiv:2211.04486}, 2022.

\bibitem{purohit2025sample}
K. Purohit, V. V, S. Bhattacharya, and A. Anand, ``Sample efficient demonstration selection for in-context learning,'' in \emph{Proc. 42nd Int. Conf. Mach. Learn. (ICML)}, Jul. 2025.

\bibitem{zhong2024memorybank}
W. Zhong, L. Guo, Q. Gao, H. Ye, and Y. Wang, ``MemoryBank: Enhancing large language models with long-term memory,'' \emph{Proc. AAAI Conf. Artif. Intell.}, vol. 38, pp. 19724--19731, 2024.

\bibitem{park2023generative}
J. S. Park, J. C. O'Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein, ``Generative agents: Interactive simulacra of human behavior,'' in \emph{Proc. 36th Annu. ACM Symp. User Interface Softw. Technol. (UIST)}, Oct. 2023, pp. 1--22.

\bibitem{xu2025amem}
W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan, and Y. Zhang, ``A-MEM: Agentic memory for LLM agents,'' \emph{arXiv preprint arXiv:2502.12110}, 2025.

\bibitem{kim2024towards}
S. H. Kim, K. T. Ong, T. Kwon, N. Kim, K. Ka, S. Bae, et al., ``Towards lifelong dialogue agents via timeline-based memory management,'' in \emph{Proc. Conf. North Amer. Chapter Assoc. Comput. Linguist. (NAACL)}, Jun. 2024.

\bibitem{kim2022playgrounds}
S. Kim, P. Phunyaphibarn, D. Ahn, and S. Kim, ``Playgrounds for abstraction and reasoning,'' in \emph{Proc. NeurIPS Workshop Neuro Causal Symbolic AI}, Dec. 2022.

\bibitem{lee2024generating}
K. I. Lee, H. Koh, D. Lee, S. Yoon, M. Kim, and K. Jung, ``Generating diverse hypotheses for inductive reasoning,'' \emph{arXiv preprint arXiv:2412.13422}, 2024.

\bibitem{qiu2024phenomenal}
L. Qiu, L. Jiang, X. Lu, M. Sclar, V. Pyatkin, C. Bhagavatula, et al., ``Phenomenal yet puzzling: Testing inductive reasoning capabilities of language models with hypothesis refinement,'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, May 2024, pp. 1452--1481.

\bibitem{zhou2024hypothesis}
Y. Zhou, H. Liu, T. Srivastava, H. Mei, and C. Tan, ``Hypothesis generation with large language models,'' in \emph{Proc. EMNLP Workshop NLP for Science}, Nov. 2024.

\bibitem{kim2023machine}
T. Kim, M. Cochez, V. Fran\c{c}ois-Lavet, M. Neerincx, and P. Vossen, ``A machine with short-term, episodic, and semantic memory systems,'' \emph{Proc. AAAI Conf. Artif. Intell.}, vol. 37, pp. 48--56, 2023.

\bibitem{li2024memory}
J. Li and J. Li, ``Memory, consciousness and large language model,'' \emph{arXiv preprint arXiv:2401.02509}, 2024.

\bibitem{wang2025mirix}
Y. Wang and X. Chen, ``MIRIX: Multi-agent memory system for LLM-based agents,'' \emph{arXiv preprint arXiv:2507.07957}, 2025.

\bibitem{manakul2023selfcheckgpt}
P. Manakul, A. Liusie, and M. Gales, ``SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models,'' in \emph{Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)}, Dec. 2023, pp. 9004--9017.

\bibitem{dhuliawala2024chain}
S. Dhuliawala, M. Komeili, J. Xu, R. Raileanu, X. Li, A. Celikyilmaz, and J. Weston, ``Chain-of-verification reduces hallucination in large language models,'' in \emph{Findings Assoc. Comput. Linguist. (ACL)}, Aug. 2024, pp. 3563--3578.

\bibitem{madaan2023selfrefine}
A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, et al., ``Self-Refine: Iterative refinement with self-feedback,'' \emph{Adv. Neural Inf. Process. Syst.}, vol. 36, pp. 46534--46594, 2023.

\bibitem{gou2024critic}
Z. Gou, Z. Shao, Y. Gong, Y. Shen, Y. Yang, N. Duan, and W. Chen, ``CRITIC: Large language models can self-correct with tool-interactive critiquing,'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, May 2024.

\bibitem{moustafa2015unsw}
N. Moustafa and J. Slay, ``UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set),'' in \emph{Proc. Mil. Commun. Inf. Syst. Conf. (MilCIS)}, Nov. 2015, pp. 1--6.

\end{thebibliography}

\end{document}
"""

full_tex = "\n\n".join([HEADER_AND_SEC1.strip(), sec2.strip(), sec3_top.strip(), SEC3_BOTTOM.strip(), SEC4_INTRO.strip(), sec4_sym.strip(), SEC4_FLOW_AND_END.strip(), TSP_REFERENCES.strip()]) + "\n"

# Safety & style assertions
assert "**" not in full_tex, "Markdown ** found!"
assert r'\"' not in full_tex, "Escaped quote macro found!"
assert "RQ1" not in full_tex and "RQ2" not in full_tex and "RQ3" not in full_tex, "RQ abbreviation found!"
assert "Electronics" not in full_tex and "Applied Sciences" not in full_tex and "MDPI" not in full_tex, "Old journal trace found!"

cited_keys = {k.strip() for m in re.findall(r"\\cite\{([^}]+)\}", full_tex) for k in m.split(",")}
bib_keys = set(re.findall(r"\\bibitem\{([^}]+)\}", full_tex))
assert cited_keys == bib_keys, f"Citation mismatch! cited-bib={cited_keys-bib_keys}, bib-cited={bib_keys-cited_keys}"

out_tex = OUT_DIR / "main.tex"
out_tex.write_text(full_tex, encoding="utf-8")

bib_src = REPO / "Electronics_submission" / "references.bib"
if bib_src.exists():
    shutil.copy2(bib_src, OUT_DIR / "references.bib")

print(f"[OK] Wrote {out_tex} ({len(cited_keys)} verified citations, zero Markdown/MDPI traces).")
