# Evaluation Metrics for the Hierarchy Emergence ART Paper

This document specifies the metrics this project should report to defend its three core claims:

1. **Category formation** — the model recruits one hub per ground-truth concept.
2. **Robustness to corruption** — template decay causes orphan hubs (recruited by garbage samples) to wither while real-concept hubs stay healthy.
3. **Hierarchy emergence** — lower vigilance produces superordinate hubs (e.g. sunfish + salmon → "fish") in a principled, taxonomy-aligned way.

For each metric we record: **what it measures**, **who introduced it**, **why it belongs in this paper**, and **how it maps to a claim**. Protocol conventions (seeds, paired tests, corruption sweep) follow the metric list.

---

## 1. Adjusted Rand Index (ARI) + Adjusted Mutual Information (AMI)

**What it measures.** Agreement between the model's hub assignment and the ground-truth concept labels, corrected for chance. ARI compares pairs of samples (are they in the same cluster in both partitions, or different in both?); AMI compares the mutual information between the two partitions normalized by their entropies. Both peak at 1.0 for perfect agreement and sit near 0.0 for chance-level agreement, with negative values possible.

**Provenance.**
- **Adjusted Rand Index**: Hubert & Arabie (1985), "Comparing partitions," *Journal of Classification* 2(1).
- **Adjusted Mutual Information**: Vinh, Epps & Bailey (2010), "Information theoretic measures for clusterings comparison: variants, properties, normalization and correction for chance," *JMLR* 11.

**Why include it.** ARI and AMI are the *de facto* clustering-quality metrics in modern ML. A paper proposing an unsupervised clustering method that does not report at least one of them is unusual and will draw reviewer questions. AMI in particular fixes a known bias in raw NMI where the score inflates with cluster count — important here because our model can be tuned to recruit many small hubs.

**Maps to claim.** Claim 1 (category formation). Report on each dataset (sensory_dropout 8-concept, synthetic_12, synthetic_12_clean 14-concept) and on the corrupted dataset for completeness.

---

## 2. Homogeneity and Completeness (reported separately)

**What it measures.** Two entropy-based metrics that together decompose cluster quality:
- **Homogeneity**: each cluster contains samples from a single ground-truth class. Penalizes mixed hubs.
- **Completeness**: each ground-truth class is contained in a single cluster. Penalizes over-segmentation (one concept split across multiple hubs).

Both are in [0, 1]. V-measure is their harmonic mean and may also be reported as a summary.

**Provenance.** Rosenberg & Hirschberg (2007), "V-Measure: A conditional entropy-based external cluster evaluation measure," *EMNLP-CoNLL*.

**Why include it.** The paper's central category-formation claim is *"exactly one hub per concept"* — which is precisely the conjunction `homogeneity == 1.0 AND completeness == 1.0`. ARI/AMI summarize both effects into a single scalar; reporting H and C separately lets us state the claim in its native shape and lets reviewers see *which* failure mode appears when a config falls short (mixed hubs vs. over-segmentation). Purity alone is insufficient because it does not penalize over-segmentation: a model with one hub per sample scores purity = 1.0.

**Maps to claim.** Claim 1. Also informative for the decay ablation in Claim 2, since corrupted samples without decay typically hurt completeness (orphan hubs split off) more than homogeneity.

---

## 3. Number of Hubs vs. Vigilance Curve

**What it measures.** For each vigilance value on a sweep (e.g. 0.80 → 0.999 in 0.01 increments), the number of distinct hubs the model recruits after training, holding other hyperparameters fixed. Overlaid with annotations marking the values at which known superordinate merges occur (sunfish/salmon → fish, robin/canary → bird, oak/pine → tree, etc.).

**Provenance.** Established as standard ART reporting practice across the foundational and follow-up literature:
- Carpenter & Grossberg (1987), "A massively parallel architecture for a self-organizing neural pattern recognition machine," *CVGIP* 37.
- Carpenter, Grossberg & Rosen (1991), "Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system," *Neural Networks* 4.
- Williamson (1996), "Gaussian ARTMAP: A neural network for fast incremental learning of noisy multidimensional maps," *Neural Networks* 9.
- Anagnostopoulos & Georgiopoulos (2002), "Category regions as new geometrical concepts in Fuzzy-ART and Fuzzy-ARTMAP," *Neural Networks* 15.

**Why include it.** This is the ART community's signature plot. Omitting it from an ART paper is conspicuous. Beyond convention, it does load-bearing work for two claims at once: it shows the vigilance regime where category count matches concept count (claim 1) and the regime where adjacent species collapse into their superordinate (claim 3) — in one figure.

**Maps to claims.** 1 (where the curve plateaus at #concepts) and 3 (where it steps down to superordinate counts).

---

## 4. Orphan-Hub Precision / Recall / F1

**What it measures.** A binary classification of each recruited hub as "real concept hub" vs. "orphan." A hub is labeled *real* if at least *k* of its members come from a single ground-truth concept (we use k=15 of 20, i.e. 75%); otherwise *orphan*. Ground-truth orphans are hubs whose members are predominantly corrupted (`_BAD`) samples. Precision = (correctly identified orphans) / (all hubs flagged orphan); recall = (correctly identified orphans) / (all true-orphan hubs); F1 is their harmonic mean.

A weaker model produces many orphans and few real hubs; template decay should drive orphan recall toward 1.0 (decayed → pruned → flagged orphan) without losing real-hub precision.

**Provenance.** The precision/recall/F1 framing for "is this cluster real or noise?" follows the novelty-detection / one-class classification convention:
- Pimentel, Clifton, Clifton & Tarassenko (2014), "A review of novelty detection," *Signal Processing* 99.
- The decay mechanism itself is mathematically adjacent to trimmed-mean robust clustering: García-Escudero, Gordaliza, Matrán & Mayo-Iscar (2010), "A review of robust clustering methods," *Advances in Data Analysis and Classification* 4.

**Why include it.** Claim 2 is the paper's most novel contribution. ARI/AMI on the corrupted dataset show *that* decay helps, but they collapse "rejected the garbage" and "kept the real concepts" into one number. Precision/recall on the orphan-detection task decomposes them — the paper can argue "decay achieves orphan recall ≈ 1.0 with no drop in real-hub precision," which is the actual mechanistic claim, not just an aggregate score.

**Maps to claim.** Claim 2 (robustness).

---

## 5. Cophenetic Correlation Between Model Dendrogram and Ground-Truth Taxonomy

**What it measures.** Sweep vigilance from high (many small hubs) to low (few large hubs) and record the merge order: at what vigilance does each pair of hubs collapse into one? This produces a dendrogram. Compute the **cophenetic distance** between every pair of leaf concepts in that dendrogram (the vigilance level at which they first share a hub) and correlate it (Pearson) against the same pairwise distances in the ground-truth concept taxonomy (e.g. sunfish-salmon distance < sunfish-robin distance < sunfish-oak distance). Values near 1.0 mean the model's merge order recapitulates the taxonomy.

**Provenance.**
- **Cophenetic correlation**: Sokal & Rohlf (1962), "The comparison of dendrograms by objective methods," *Taxon* 11.
- **Application to connectionist semantic memory**: Rogers & McClelland (2004), *Semantic Cognition: A Parallel Distributed Processing Approach*, MIT Press — uses hierarchical clustering of hidden-layer activations across training epochs as the canonical evaluation of emergent taxonomic structure.
- **Modern analytic treatment**: Saxe, McClelland & Ganguli (2019), "A mathematical theory of semantic development in deep neural networks," *PNAS* 116(23) — derives the singular-value trajectories that produce the differentiation pattern Rogers & McClelland observed.

**Why include it.** Claim 3 needs a metric that says "the hierarchy the model finds matches the hierarchy that exists in the data." Cophenetic correlation is the standard metric for that comparison and the one Rogers & McClelland's tradition uses, so a cognitive-science reviewer will look for it (or for a representational dissimilarity matrix figure that conveys the same information, which we should also produce as a companion).

**Maps to claim.** Claim 3 (hierarchy emergence).

---

## Protocol conventions

These apply uniformly across the metrics above.

### Seeds and replication

- **Number of seeds**: 20-30 per condition, consistent with the ART and connectionist-semantics literatures.
- **Two independent seeds per run**: (a) *dataset regeneration seed* — controls which random vectors the corrupted samples receive and the Gaussian noise on clean samples; (b) *training-order seed* — controls sample presentation order. Report which is varied; ideally vary both and cross them, or fix one and sweep the other with explicit justification.
- **Reporting**: mean ± SD or 95% CI (bootstrap is fine given the small sample size) on every aggregate number in the paper. No bare point estimates.

### Decay ablation

- **Paired Wilcoxon signed-rank** across seeds, comparing decay-on vs. decay-off on the same dataset regenerations and presentation orders. Non-parametric, robust to small-n distribution shape, and the convention used in Saxe et al. (2019) and the broader clustering-robustness literature.
- Report effect size alongside p-value (e.g. rank-biserial correlation or median difference with CI). With 20-30 seeds, statistical significance is cheap; effect size is what matters.

### Corruption sensitivity sweep

A single corruption level (2 BAD samples per concept) is one data point. For the paper, sweep the BAD-per-concept ratio over `{0, 1, 2, 5, 10}` and plot ARI and orphan-F1 curves for decay-on vs. decay-off. This converts "decay helped at our chosen corruption level" into "decay extends the model's tolerable corruption range from X to Y," a stronger and more falsifiable claim.

### Baselines

ART-with-decay should be compared on every metric against:
- **Standard ART** (decay rate = 0) on the same datasets — the natural ablation.
- **k-means** (with k = true concept count, and with k chosen by silhouette) — the canonical clustering baseline.
- **Gaussian Mixture Model** — handles the noisy-prototype generative story directly.
- **Agglomerative clustering** with average linkage — for the hierarchy claim specifically; cophenetic correlation is its native evaluation.

Without baselines, the metrics report internal performance but not relative merit. A reviewer's first question on any clustering paper is "compared to what?"

---

## What we are deliberately *not* reporting, and why

- **Raw purity** — superseded by homogeneity (chance-corrected analog) and by ARI (pair-counting analog). Reporting purity in addition would be redundant and slightly misleading because of its lack of an over-segmentation penalty.
- **Classification accuracy** — appropriate for ARTMAP (supervised ART), not for the unsupervised model under evaluation. Inviting a supervised head onto our model just to report accuracy would obscure rather than support claim 1.
- **Silhouette score** — useful for choosing k in k-means baselines, but as a quality metric for the proposed model it has no ground-truth anchor and is dominated by ARI/AMI for our purposes.

---

## Summary table

| Metric | Claim | Standard in | One-line role in the paper |
|---|---|---|---|
| ARI + AMI | 1 | Modern ML clustering | Chance-corrected agreement with ground-truth labels |
| Homogeneity & Completeness | 1, 2 | NLP / clustering eval | States "one hub per concept" in its native form |
| #hubs vs. vigilance curve | 1, 3 | ART literature | Mandatory ART plot; shows category and hierarchy regimes |
| Orphan precision / recall / F1 | 2 | Novelty detection | Decomposes "decay helps" into its mechanistic parts |
| Cophenetic correlation | 3 | Connectionist semantics | Standard hierarchy-vs-taxonomy comparison |

---

## References (full)

- Anagnostopoulos, G. C., & Georgiopoulos, M. (2002). Category regions as new geometrical concepts in Fuzzy-ART and Fuzzy-ARTMAP. *Neural Networks*, 15(10).
- Carpenter, G. A., & Grossberg, S. (1987). A massively parallel architecture for a self-organizing neural pattern recognition machine. *Computer Vision, Graphics, and Image Processing*, 37(1).
- Carpenter, G. A., Grossberg, S., & Rosen, D. B. (1991). Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system. *Neural Networks*, 4(6).
- García-Escudero, L. A., Gordaliza, A., Matrán, C., & Mayo-Iscar, A. (2010). A review of robust clustering methods. *Advances in Data Analysis and Classification*, 4(2-3).
- Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1).
- Pimentel, M. A. F., Clifton, D. A., Clifton, L., & Tarassenko, L. (2014). A review of novelty detection. *Signal Processing*, 99.
- Rogers, T. T., & McClelland, J. L. (2004). *Semantic Cognition: A Parallel Distributed Processing Approach*. MIT Press.
- Rosenberg, A., & Hirschberg, J. (2007). V-Measure: A conditional entropy-based external cluster evaluation measure. *EMNLP-CoNLL*.
- Saxe, A. M., McClelland, J. L., & Ganguli, S. (2019). A mathematical theory of semantic development in deep neural networks. *PNAS*, 116(23).
- Sokal, R. R., & Rohlf, F. J. (1962). The comparison of dendrograms by objective methods. *Taxon*, 11(2).
- Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic measures for clusterings comparison: Variants, properties, normalization and correction for chance. *JMLR*, 11.
- Williamson, J. R. (1996). Gaussian ARTMAP: A neural network for fast incremental learning of noisy multidimensional maps. *Neural Networks*, 9(5).
