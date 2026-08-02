"""
flexattn_blocksize.py — lower FlexAttention's BLOCK_SIZE inside JanusDNA.

THE PROBLEM
`JanusDNAAttention.forward` (modeling_janusdna.py, around line 603) calls
create_block_mask() without BLOCK_SIZE, so PyTorch uses the default of 128. The
sequence handed to flex_attention is the CONCATENATION of the two reading
directions, so q_len = 2*L. The effective constraint is therefore 2*L >= 128:

    L >= 64      <-- floor confirmed empirically by bisection

Below that threshold the Triton kernel raises a RuntimeError or a device-side
assert. This is what blocks the short end of the window-length sweep.

THE FIX
create_block_mask is replaced in the module namespace with a version that
injects BLOCK_SIZE. Lowering it does NOT change the semantics of the model: the
block mask is only the tiling used to represent the mask, and flex_attention
still evaluates mask_mod position by position inside partial blocks. It changes
the cost, not the result.

    BLOCK_SIZE=16  ->  floor L >= 8
    BLOCK_SIZE=32  ->  floor L >= 16

USAGE — active only when the environment variable is set. Without it the
behaviour is bit-identical to the runs already produced, which is deliberate:
the patch must never silently alter an existing result.

    import flexattn_blocksize
    flexattn_blocksize.apply()      # reads JANUS_FLEX_BLOCK

to be placed AFTER importing janusdna.modeling_janusdna and BEFORE instantiating
JanusDNAModel.

    JANUS_FLEX_BLOCK=16 python src/extract_janusdna.py ...

MANDATORY GATE
Before trusting this at short lengths, re-extract a length that already works
(say L = 1000) both with and without the patch and confirm the embeddings agree
to numerical tolerance. If they diverge non-trivially, the patch is NOT neutral
in this environment and must be abandoned rather than worked around.
"""
import os

_applied = False


def apply(block_size=None, verbose=True):
    """Replace create_block_mask in the modeling_janusdna namespace.

    block_size: integer; if None, read JANUS_FLEX_BLOCK from the environment.
                If neither is set, this is a no-op.
    """
    global _applied
    if _applied:
        return

    if block_size is None:
        env = os.environ.get("JANUS_FLEX_BLOCK", "").strip()
        if not env:
            return
        block_size = int(env)

    if block_size & (block_size - 1) != 0:
        raise ValueError(f"BLOCK_SIZE must be a power of 2, got {block_size}")

    import janusdna.modeling_janusdna as m

    original = m.create_block_mask

    def create_block_mask_fixed(*args, **kwargs):
        kwargs.setdefault("BLOCK_SIZE", block_size)
        return original(*args, **kwargs)

    create_block_mask_fixed.__wrapped__ = original
    m.create_block_mask = create_block_mask_fixed
    _applied = True

    if verbose:
        print(f"[flexattn patch] BLOCK_SIZE={block_size} "
              f"-> floor L >= {max(1, block_size // 2)}")
