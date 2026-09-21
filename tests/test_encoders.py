"""Pooling, on which a paper claim rests.

§VI-D reports that Indic MLM checkpoints underperform sentence-trained encoders
at retrieval, and defends the result by saying both mean and CLS pooling were
run so it is not a pooling artefact. That defence is worth something only if the
pooling is correct, and the arithmetic sat inline in the encode loop where no
test could reach it.

The failure that matters is averaging across padding. Batches are padded to
their longest member, so a short text batched with a long one picks up padding
states; averaging those in shrinks its vector toward zero by an amount that
depends on what else was in the batch. The encoder's output would then vary with
batch composition, degrading precisely the mean-pooled MLM arm the claim is
about -- and nothing would look wrong.
"""

from __future__ import annotations


def _t():
    import torch

    return torch


def test_padding_does_not_enter_the_mean():
    """The padded position holds 9.9s. If they reached the average, the second
    row would not equal the first row's single real token."""
    torch = _t()
    from indicrag.index.dense import pool

    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [9.9, 9.9]]])
    mask = torch.tensor([[1, 1], [1, 0]])

    pooled = pool(hidden, mask, "mean")
    assert pooled[0].tolist() == [2.0, 3.0]   # both tokens count
    assert pooled[1].tolist() == [1.0, 2.0]   # padding ignored


def test_a_sequence_encodes_the_same_however_it_is_batched():
    """The property that makes an embedding cache valid across runs: batching is
    an implementation detail and must not change the vector."""
    torch = _t()
    from indicrag.index.dense import pool

    alone = pool(
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        torch.tensor([[1, 1]]),
        "mean",
    )
    batched = pool(
        torch.tensor([[[1.0, 2.0], [3.0, 4.0], [7.0, 7.0]]]),
        torch.tensor([[1, 1, 0]]),
        "mean",
    )
    assert alone[0].tolist() == batched[0].tolist()


def test_cls_pooling_takes_the_first_position():
    torch = _t()
    from indicrag.index.dense import pool

    hidden = torch.tensor([[[5.0, 6.0], [1.0, 1.0]]])
    mask = torch.tensor([[1, 1]])
    assert pool(hidden, mask, "cls")[0].tolist() == [5.0, 6.0]


def test_the_two_poolings_differ_so_the_ablation_is_real():
    """If these agreed, running both would prove nothing and §VI-D's defence
    would be empty."""
    torch = _t()
    from indicrag.index.dense import pool

    hidden = torch.tensor([[[5.0, 6.0], [1.0, 0.0]]])
    mask = torch.tensor([[1, 1]])
    assert pool(hidden, mask, "mean")[0].tolist() != pool(hidden, mask, "cls")[0].tolist()


def test_an_all_padding_row_does_not_divide_by_zero():
    """Cannot arise from the tokenizer, but the clamp is what guarantees it and
    a NaN here would propagate silently through the whole index."""
    torch = _t()
    from indicrag.index.dense import pool

    pooled = pool(
        torch.tensor([[[1.0, 2.0]]]),
        torch.tensor([[0]]),
        "mean",
    )
    assert not torch.isnan(pooled).any()
