# Letter to the Editor - correction notice

[Date]

To the Editor-in-Chief
[Journal name]

**Re: Manuscript [Manuscript ID] - "Conditional modal branch selection in
physics-informed active control of lid-driven cavity flow"**

Dear Editor,

I am writing to disclose, proactively, a correction to the classification scheme
used in the above manuscript, which is currently under review. In assembling a
clean and fully reproducible package of the analysis, I identified an error in the
definition of the modal branches. The main findings are unaffected, but a subset
of the discussion requires correction. I would like to submit a revised version
together with this notice.

**Nature of the correction.** The manuscript classifies control realizations into
three branches: `B_1` (second-harmonic dominated), `B_2` (time-dependent
dominated), and `B_3` ("mixed"). In reality, the runs obtained with **zero
variance regularization (`lambda_var = 0`)** and the corresponding **"No L_var"
ablation** are not mixed at all; they are dominated by the **stationary first
harmonic** (`f_1 = Efrac_0_0 > 0.9`, i.e. a `sin(pi x)` control). These form a
physically distinct fourth class, `B_first`, which the original three-class scheme
folded into `B_3`. Two further points are corrected: the main text quoted
thresholds (`f_2 > 0.70`, `f_t > 0.70`) that did not match the implemented
dominance rule (`f_2 > 2 f_t and f_2 > 0.5`, resp. `f_t > 2 f_2 and f_t > 0.5`;
identical labels on the reference grid), and the normalized Shannon entropy was
scaled by `ln 3` instead of `ln 4`.

**Scope and impact.** The 185-run reference campaign was performed at
`lambda_var = 1` and contains no `B_first` realization (`f_1 <= 0.13`
throughout). Its reported statistics (`B_1` 42.2 %, `B_2` 43.8 %, `B_3` 14.1 %)
and all main conclusions are therefore **unchanged**; an independent multi-seed
audit still shows 0/185 label discrepancies. The correction affects only the
`lambda_var` sweep and the loss ablation, where the interpretation changes from
"mixed" to a **deterministic selection of the stationary first harmonic** at
`lambda_var = 0`. This actually reinforces the central message: the variance
penalty excludes the lowest-dissipation fundamental mode.

**Materials provided.** A corrected manuscript and supplementary material (LaTeX
and PDF), with recomputed tables and regenerated figures (in particular the
`lambda_var` branch-probability heatmap, the entropy curve, and the loss-ablation
figure), together with a detailed correction report that lists every changed
number and a data-reuse registry. The entire correction is reproducible from the
frozen per-run data.

I believe prompt self-disclosure is the appropriate course and I apologize for the
oversight. I am happy to provide any additional information or to follow whatever
revision procedure you prefer. Could you please advise on how best to submit the
corrected files (revised manuscript, supplementary, and correction report)?

Thank you for your consideration.

Yours sincerely,

[Name]
[Affiliation]
[Email]
