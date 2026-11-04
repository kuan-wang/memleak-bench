"""Naive baseline: text-delete only (current industry practice).

Deletes only the text memory entry. Leaves image files, image embeddings,
graph edges, and everything else intact. This is what most deployed
systems actually do today.

Expected: highest CMLR of any system with deletion capability,
because it only addresses the addressable text node in the IPG.
"""

from __future__ import annotations

from memleak.harness.oracle import OracleExplicit, OracleMemoryStore
from memleak.harness.base import ForgetResult
from memleak.schema.types import Profile


class NaiveBaseline(OracleExplicit):
    """Text-delete only. Leaves images and embeddings intact.

    Subclasses OracleExplicit but overrides forget() to only
    delete text entries — not images, embeddings, or graph edges.
    This simulates what Mem0.delete(), Graphiti.invalidate(), and
    most production systems actually do.
    """

    async def forget(self, instruction: str) -> ForgetResult:
        if not self.profile or not self.profile.retraction_point:
            return ForgetResult(success=False, method="naive_text_only")

        deleted = []
        for fact_id in self.profile.retraction_point.target_facts:
            # Delete ONLY the text entry — leave images intact
            if fact_id in self.store.text_entries:
                deleted.append(f"text:{fact_id}")
                del self.store.text_entries[fact_id]
            # Deliberately do NOT delete images or mark as deleted
            # self.store.images is left intact
            # self.store.deleted_facts is NOT updated (so get_active_context
            # won't filter, but the text entry is gone)

        # Mark as deleted for context filtering
        for fact_id in self.profile.retraction_point.target_facts:
            self.store.deleted_facts.add(fact_id)

        return ForgetResult(
            success=True,
            deleted_items=deleted,
            method="naive_text_only",
            metadata={
                "deleted_text_entries": len(deleted),
                "images_retained": sum(
                    len(imgs) for fid, imgs in self.store.images.items()
                    if fid in self.profile.retraction_point.target_facts
                ),
                "note": "Images and embeddings deliberately NOT deleted",
            },
        )
