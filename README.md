# Mech-Interp Work


## SVD Interpretability

This very **preliminary** work was done as part of a few day "research sprint" for phase 1 of SERIMATS under Neel Nanda. In large part, its a replication of work by Beren and Sid Black of Conjecture: [Singular Value Decompositions of Transformer Weight Matrices are Highly Interpretable](https://www.lesswrong.com/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight)

**Note:** This was auto converted from the [following google doc](https://docs.google.com/document/d/1uuqB14GeROqit7V0G39BWI1BiAkCatiDO8uY67ugDV0/edit?usp=sharing), so some formatting maybe weird. 


**Research Sprint - SVD Interpretability**

-   Initially a replication, and exploration of this LW post: [**[The
    > Singular Value Decompositions of Transformer Weight Matrices are
    > Highly
    > Interpretable]{.underline}**](https://www.lesswrong.com/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight)

-   This project is mostly a "bottom-up"/developing new interp tools
    > project, as opposed to the more usual "top-down"/use existing
    > tools to find/explain already observed interesting behaviors in
    > models.

```{=html}
<!-- -->
```
-   For the most part, I was able to replicate their key results on the
    > **OV circuit**

-   I did **not** yet have time to investigate the MLP layers

-   All findings were replicated on **gpt2-medium** as in the original.

    -   **gp2-small** had varying but small levels of success. In the
        > interest of time, did not thoroughly explore and switched to
        > -medium.

    -   2layer-attn-only was almost completely unsuccessful (did not
        > spend much time on this either)

-   In general, looking at a bag-of-words and deciding whether this is
    > interpretable or not, is not scalable, reproducible, or well
    > grounded.

    -   To first approximation, looking at SV words can be weighed by
        > how common a word is. In general more common words = more
        > interpretable.

    -   This is only a first order term... then theres how common
        > words/tokens are with each other.

        -   Even rare words/tokens together can be semantically
            > meaningful.

**Key Findings**

A.  (Abnormally) Large Spectrum changes in later layers?

    i.  Observed "normal" spectrum growth across layers (linked to
        > reported norm and std dev increases across depth), but
        > gpt2-small-layer11 had a several orders of magnitude jump

> ![](media/image2.png){width="4.284827209098863in"
> height="3.1437587489063867in"}

ii. Did not replicate/investigate in gpt2-medium (out of time)

```{=html}
<!-- -->
```
B.  A head's Singular Vectors (SVs) can be individually interpretable -
    > **replicated**

    i.  TransformerLens weight transformations seem to affect the
        > baseline "interpretability" level

        -   Did not have time to thoroughly dig into this, but I spent a
            > long time not being able to fully replicate results, or
            > only doing so partially.

        -   Eventually set \`fold_ln\`, \`center_writing_weights\` and
            > \`center_unembed\` to False and was able to fully
            > replicate results.

        -   More importantly, t**he baseline level of interpretability
            > seemed to noticeably increase**

        -   **Questions:**

            -   Why would these transformations affect the
                > interpretability of our SVs?

            -   What theory do we have to tell us we should have
                > expected or not expected this?

            -   This needs to be more carefully empirically explored

    ii. When comparing SVs, there is a difference between taking the
        > low-rank SVD, and taking a full SVD and truncating it
        > yourself.

        -   The valence (pos/neg) seems to be shuffled and makes
            > comparison between each other hard

        -   **Note: This was a slippery bug to track down**

        -   **Note**: TransformerLens uses \`torch.svd\` which returns V
            > (not V_T), but in the source code is called Vh for a
            > conjugate/Hermitian transpose. But for real numbers, this
            > is just the actual transpose. Thats why we have to do
            > Vh.mT when using Transformer Lens, but not
            > pytorch.linalg.svd, which directly returns Vth/V_T.

            -   I will suggest a pull request to change the naming in
                > the source.

C.  SV can be interpretable in either "valence" (positive/negative) -
    > sometimes in both - **replicated**

    i.  The SVD can be composed of either u/v or -u/-v, and information
        > can be represented in either valence.

D.  Left and Right SVs can **both** be interpretable - **mine**

    i.  Left SV can be interpretable, **depending** on the unembedding
        > used

        -   Experimented with both W_U, and torch.linalg.pinv(W_E) →
            > only got meaningful results from W_U

            -   **Questions**

                a.  What theory do we have to tell us we should have
                    > expected or not expected this?

                b.  This needs to be more carefully empirically explored

    ii. Left SVs are "less interpretable" than the right

        -   **Purely empirical, not carefully studied** - do not have a
            > good idea why this should or should not be...

    iii. **Idea/Question:** need to more carefully frame the idea of
         > left SV reading from the residual stream and right SV writing
         > back to the residual stream

         -   How does the unembedding matrix we use to "visualize" these
             > SVs relate to this framing?

         -   Using dyadic notation (sum of outer products), we can
             > really see the pairing of u_i, v_i, where one reads and
             > one writes.

             -   Will a geometric view of the outer product help us
                 > here?

             -   Does using this help us further decompose the already
                 > almost entirely linear transformer (with frozen
                 > attention and ignoring layer norm) like the original
                 > "A Mathematical Framework..." paper?

E.  Sometimes, the top-10 SV of a head **jointly** form a "semantic
    > unit" - **replicated**

    i.  **Idea**: For me, this completely reframes head ablation. If a
        > head is represented through its SVDs as a
        > "literature/media/writing" head, ablating it in a specific
        > tasks can have very many subtle effects.

        -   Does this create an avenue to measure the amount/degree of
            > superposition?

            -   For ex: we ablate the largest "weather" head, how does
                > performance degrade on weather tasks? The second head,
                > etc.

            -   Maybe related to some measure of entanglement...

        -   If a head does contain a semantic grouping of SVs (as
            > opposed to just a bunch of seemingly unrelated SVs) - does
            > that tell us anything about superposition?

            -   Or rather, if there are NO heads that do this, only
                > individually meaningful SVs\' is that an indication of
                > (or degree of) superposition?

            -   The \# of heads with semantically grouped SVs (semantic
                > heads) vs other measures of superposition?

        -   Either way, ablation needs to be more carefully considered
            > in light of this.

F.  Concept Tracing (SV tracing in the post) can be (somewhat) used to
    > "search" the SVD space - **replicated**

    i.  Cosine_sim and thresholding can be used to search for meaningful
        > SVs

G.  Rank-1 Updates are a potential avenue for "concept ablation" as
    > opposed to head/neuron ablation - **replicated/expanded**

    i.  Removing semantic SVs has meaningful effect on the prob of
        > outputting correct token

        -   Using dyadic notation, its clear we can target individual
            > SVs. Having identified them using concept tracing, we can
            > delete (ablate) them by setting their SVs to 0

        -   Ran several experiments on several meaningful SVs

    ii. Can even be used as another measure of discovery

        -   IE: removing a SV had a large effect on the prop of correct
            > token - they are related.

    iii. **Idea**: Instead of full removal, can try a more continuous
         > **dampening** or **amplifying of concepts**

         -   Note, we can write it like this:

> ![](media/image1.png){width="3.28125in" height="1.2375in"}

-   Typically, \\eps_i is set to either 0, or -1.

-   What about in between? (dampening)

-   What about between 0-1 (amplifying)?

-   **VERY PRELIM IDEA:** A possible way to deal with the removal of
    > distributed representations, is to dampen each related concept SV
    > in proportion to how related it is.

    -   This will obviously have either huge or no effect...

    -   Need to think about this more carefully...

    -   There is also always the optimization approach to setting eps...

iv. **Slippery bug:**

    -   When attempting to set the weights in TL, model properties are
        > lru_cached! Meaning we need to change and check like:
        > model.blocks\[layer\].attn.W_V\[head\].

        -   Possible avenue to add **setters** in TL that cause these
            > properties to regenerate?

**Future Work**


A.  Clearly explain a "theory" of left/right singular vector pos/neg
    > valence, a-la AMF paper

B.  Expand Concept Tracing beyond cosine sim and iterative thresholding

C.  Explore any relation this may have to measuring
    > superposition/distributed representations and/or disentanglement.

D.  Explore expanding Rank-1 editing into continuous
    > dampening/amplification setting, including adaptively setting eps
    > values

E.  Use this to explore **selectively** effecting induction. That is,
    > can we break induction like this: ABAB works for all tokens except
    > B=fire...

F.  Use these tools to explore and elucidate more complicated circuit
    > behavior (IOI, etc).

