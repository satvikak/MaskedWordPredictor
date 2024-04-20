# Analysis

## Layer TODO, Head TODO

My first experiment was to see if any attention heads learned the relationship between verbs and prepositions. I noticed that Layer 5, Head 8 was the best between the three sentences in identifying an actual relationship. Though this attention head also placed emphasis on [SEP], it overall had the best understanding of a connection between verbs and preposition amongst both sentences.

Example Sentences:
- The tourists arrived at the [MASK].
- The [MASK] consisted of fifteen chapters.

## Layer TODO, Head TODO

My second experiment was to test the relationship between verbs and determiners. More specifically, I wanted to identify attention heads where a relationship between "a" and a verb was learned. I noticed that Layer 8 Head 5 seemed to have identified this the most. In fact, in both cases, it appeared that a verb/delimeter relationship was the most prominent relationship identified compared to the other words.

Example Sentences:
- She rides a [MASK].
- Traffic moves a [MASK].