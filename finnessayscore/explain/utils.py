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


class UposFilter:
    def __init__(self, tokenizer, filtered_upos):
        self.tokenizer = tokenizer
        self.filtered_upos = filtered_upos
        self.control_tokens = [
            tokenizer.sep_token_id,
            tokenizer.cls_token_id,
            tokenizer.pad_token_id,
        ]

    def update_essay(self, essay_i, batch):
        input_ids = batch["input_ids"][essay_i]
        surfs = (surf for sent in batch["surfs"][essay_i] for surf in sent)
        uposes = (upos for sent in batch["upos"][essay_i] for upos in sent)
        #print(batch["upos"][essay_i])
        #print()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        surf_tokens = zip(surfs, uposes)
        bert_tokens = zip(input_ids, tokens)
        self.filter_precomputed = []
        char_pos = 0
        surf = None
        while 1:
            try:
                token_id, bert_token = next(bert_tokens)
                if token_id in self.control_tokens:
                    self.filter_precomputed.append(False)
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
                        surf, upos = next(surf_tokens)
                    except StopIteration:
                        break
            if not surf[char_pos:].startswith(bert_token_strip):
                # XXX: This is quite bad. We should use some better form of alignment than hard-coding contractions
                if bert_token_strip in ("ettei",):
                    self.filter_precomputed.append(upos in self.filtered_upos)
                    surf, upos = next(surf_tokens)
                    surf, upos = next(surf_tokens)
                    char_pos = 0
                    continue
                else:
                    raise ValueError(
                        "Error zipping surfs and bert tokens: "
                        f"expected {surf} to have {bert_token_strip} "
                        f"starting at {char_pos}"
                        f"\nsurfs: {batch['surfs'][essay_i]!r}"
                        f"\ntokens: {tokens!r}"
                    )
            char_pos += len(bert_token_strip)
            self.filter_precomputed.append(upos in self.filtered_upos)
        while 1:
            try:
                token_id, bert_token = next(bert_tokens)
            except StopIteration:
                break
            if token_id not in self.control_tokens:
                raise ValueError(
                    f"Got non control token {bert_token} after surfs have ended"
                    f"\nsurfs: {batch['surfs'][essay_i]!r}"
                    f"\ntokens: {tokens!r}"
                )
            self.filter_precomputed.append(False)

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
