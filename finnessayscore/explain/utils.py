# XXX: This is quite bad. We should use some better form of alignment than hard-coding contractions
CONTRACTION_TOKEN_OVERRIDES = {
    "ettei": 2,
}


ALL_NER_TAGS = [
    "WORK_OF_ART", "TIME", "QUANTITY", "PRODUCT", "PERSON", "PERCENT", "ORG",
    "NORP", "MONEY", "LOC", "LAW", "LANGUAGE", "GPE", "FAC", "EVENT", "DATE"
]


class ControlTokenFilter:
    def __init__(self, tokenizer):
        # A token used as a separator between question and text and it is also
        # added to the end of the text.
        sep_token_id = tokenizer.sep_token_id
        # A token used for prepending to the concatenated question-text word
        # sequence
        cls_token_id = tokenizer.cls_token_id

        self.control_tokens = [sep_token_id, cls_token_id]

    def update_essay(self, essay_i, batch):
        pass

    def __call__(self, token_idx, token):
        return token in self.control_tokens


def precompute_token_filter_from_surfs(
    input_ids,
    tokens,
    surfs_nested,
    tags_nested,
    should_filter,
    control_tokens,
    overrides
):
    surfs = (surf for sent in surfs_nested for surf in sent)
    tags = (tag for sent in tags_nested for tag in sent)
    surf_tokens = zip(surfs, tags)
    bert_tokens = zip(input_ids, tokens)
    filter_precomputed = []
    char_pos = 0
    surf = None
    while 1:
        try:
            token_id, bert_token = next(bert_tokens)
            if token_id in control_tokens:
                filter_precomputed.append(False)
                continue
        except StopIteration:
            break
        if bert_token.startswith("##"):
            bert_token_strip = bert_token[2:]
        else:
            bert_token_strip = bert_token
            if surf is None or char_pos == len(surf):
                char_pos = 0
                try:
                    surf, tag = next(surf_tokens)
                except StopIteration:
                    break
        if not surf[char_pos:].startswith(bert_token_strip):
            if bert_token_strip in overrides:
                filter_precomputed.append(should_filter(tag))
                for _ in range(overrides[bert_token_strip]):
                    surf, tag = next(surf_tokens)
                char_pos = 0
                continue
            else:
                raise ValueError(
                    "Error zipping surfs and bert tokens: "
                    f"expected {surf} to have {bert_token_strip} "
                    f"starting at {char_pos}"
                    f"\nsurfs: {surfs_nested!r}"
                    f"\ntokens: {tokens!r}"
                )
        char_pos += len(bert_token_strip)
        filter_precomputed.append(should_filter(tag))
    while 1:
        try:
            token_id, bert_token = next(bert_tokens)
        except StopIteration:
            break
        if token_id not in control_tokens:
            raise ValueError(
                f"Got non control token {bert_token} after surfs have ended"
                f"\nsurfs: {surfs_nested!r}"
                f"\ntokens: {tokens!r}"
            )
        filter_precomputed.append(False)
    return filter_precomputed


class UposFilter:
    def __init__(self, tokenizer, filtered_upos):
        self.tokenizer = tokenizer
        self.filtered_upos = set(filtered_upos)
        self.control_tokens = [
            tokenizer.sep_token_id,
            tokenizer.cls_token_id,
            tokenizer.pad_token_id,
        ]

    def update_essay(self, essay_i, batch):
        input_ids = batch["input_ids"][essay_i]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        self.filter_precomputed = precompute_token_filter_from_surfs(
            input_ids,
            tokens,
            batch["surfs"][essay_i],
            batch["upos"][essay_i],
            lambda upos: upos in self.filtered_upos,
            self.control_tokens,
            CONTRACTION_TOKEN_OVERRIDES
        )

    def __call__(self, token_idx, token):
        return self.filter_precomputed[token_idx]


class NerFilter:
    def __init__(self, tokenizer, filtered_ner):
        self.tokenizer = tokenizer
        self.filtered_ner = set(filtered_ner)
        if "ALL" in self.filtered_ner:
            self.filtered_ner.pop("ALL")
            self.filtered_ner.update(ALL_NER_TAGS)
        self.control_tokens = [
            tokenizer.sep_token_id,
            tokenizer.cls_token_id,
            tokenizer.pad_token_id,
        ]

    def update_essay(self, essay_i, batch):
        input_ids = batch["input_ids"][essay_i]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        self.filter_precomputed = precompute_token_filter_from_surfs(
            input_ids,
            tokens,
            batch["surfs"][essay_i],
            batch["ner"][essay_i],
            lambda ner: ner[2:] in self.filtered_ner,
            self.control_tokens,
            {}
        )

    def __call__(self, token_idx, token):
        return self.filter_precomputed[token_idx]


class UnionFilter:
    def __init__(self, filter_a, filter_b):
        self.filter_a = filter_a
        self.filter_b = filter_b

    def update_essay(self, essay_i, batch):
        self.filter_a.update_essay(essay_i, batch)
        self.filter_b.update_essay(essay_i, batch)

    def __call__(self, token_idx, token):
        return (
            self.filter_a(token_idx, token) or
            self.filter_b(token_idx, token)
        )
