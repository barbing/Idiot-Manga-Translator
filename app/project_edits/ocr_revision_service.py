# -*- coding: utf-8 -*-
"""Atomic application service for one explicit parent-scoped OCR revision."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import os
from typing import Any, Mapping

from PIL import Image

from app.io.project_edit_store import (
    ProjectEditReadSnapshot,
    ProjectEditStore,
    StalePageEditHeadError,
    StaleProjectEditHeadError,
)
from app.pipeline.hierarchy_revision_contracts import (
    ParentOrigin,
    RevisionRequiredAction,
    RevisionStage,
    RevisionStageState,
)
from app.pipeline.ocr_revision_contracts import (
    CancellationProbe,
    ExplicitOcrRevisionReceipt,
    ExplicitOcrRevisionRequest,
    OCR_SOURCE_SELECTION_EDIT_ID_PREFIX,
    OcrRecognitionRequest,
    OcrRevisionError,
    OcrRevisionErrorCode,
    OcrRevisionRecognitionPort,
    OcrSourceRevisionArtifact,
    OriginalPageAssetBinding,
)

from .contracts import EditDomain, EditTarget, EditTargetKind, create_project_edit
from .fingerprints import canonical_sha256, project_id_for
from .invalidation import (
    Dependency,
    InvalidationAction,
    InvalidationScope,
    invalidation_for_edit,
)
from .projection import (
    EffectivePageSnapshot,
    EffectiveParentSnapshot,
    project_effective_page,
)


def _project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise OcrRevisionError(
            OcrRevisionErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = tuple(
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    )
    if len(matches) != 1:
        raise OcrRevisionError(
            OcrRevisionErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is not exact: {page_id}",
        )
    return matches[0]


def _original_asset_reference(page: Mapping[str, Any]) -> str:
    cleaned = page.get("cleaned_page_base")
    if not isinstance(cleaned, Mapping):
        cleaned = {}
    nested = cleaned.get("cleaned_page_base")
    if not isinstance(nested, Mapping):
        nested = {}
    value = (
        page.get("image_path")
        or page.get("source_image_path")
        or cleaned.get("source_image_path")
        or nested.get("source_image_path")
    )
    reference = str(value or "").strip()
    if not reference:
        raise OcrRevisionError(
            OcrRevisionErrorCode.ORIGINAL_ASSET_UNAVAILABLE,
            "The committed original-page asset is unavailable.",
        )
    return reference


def _load_original_page(
    project: Mapping[str, Any],
    *,
    page_id: str,
    project_path: str,
) -> tuple[OriginalPageAssetBinding, Image.Image]:
    page = _project_page(project, page_id)
    reference = _original_asset_reference(page)
    candidate = os.path.expandvars(os.path.expanduser(reference))
    if not os.path.isabs(candidate):
        candidate = os.path.join(os.path.dirname(project_path), candidate)
    candidate = os.path.abspath(candidate)
    try:
        with open(candidate, "rb") as stream:
            payload = stream.read()
    except OSError as exc:
        raise OcrRevisionError(
            OcrRevisionErrorCode.ORIGINAL_ASSET_UNAVAILABLE,
            "The committed original-page asset cannot be read.",
        ) from exc
    content_sha256 = sha256(payload).hexdigest()
    try:
        with Image.open(BytesIO(payload)) as opened:
            opened.load()
            image = opened.copy()
    except Exception as exc:
        raise OcrRevisionError(
            OcrRevisionErrorCode.ORIGINAL_ASSET_UNAVAILABLE,
            "The committed original-page pixels cannot be decoded.",
        ) from exc
    width, height = int(image.width), int(image.height)
    asset_id = "original-page-v1-" + canonical_sha256(
        {"page_id": page_id, "asset_reference": reference}
    )
    try:
        binding = OriginalPageAssetBinding(
            asset_id=asset_id,
            asset_reference=reference,
            content_sha256=content_sha256,
            width=width,
            height=height,
        )
    except (TypeError, ValueError) as exc:
        image.close()
        raise OcrRevisionError(
            OcrRevisionErrorCode.ORIGINAL_ASSET_UNAVAILABLE,
            "The committed original-page canvas is invalid.",
        ) from exc
    return binding, image


def resolve_original_page_asset_binding(
    project: Mapping[str, Any],
    *,
    page_id: str,
    project_path: str,
) -> OriginalPageAssetBinding:
    """Resolve the exact committed original asset used to prepare a request."""

    binding, image = _load_original_page(
        project,
        page_id=page_id,
        project_path=os.path.abspath(str(project_path or "")),
    )
    image.close()
    return binding


def _effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    matches = tuple(
        parent for parent in snapshot.parents if parent.parent_id == parent_id
    )
    if len(matches) != 1:
        raise OcrRevisionError(
            OcrRevisionErrorCode.PARENT_NOT_FOUND,
            f"Effective parent identity is not exact: {parent_id}",
        )
    return matches[0]


def _require_request_state(
    snapshot: ProjectEditReadSnapshot,
    request: ExplicitOcrRevisionRequest,
) -> tuple[EffectivePageSnapshot, EffectiveParentSnapshot]:
    if project_id_for(snapshot.project) != request.project_id:
        raise OcrRevisionError(
            OcrRevisionErrorCode.PROJECT_IDENTITY_MISMATCH,
            "Project identity does not match the OCR revision request.",
        )
    if snapshot.page_head_sha256 != request.expected_page_head_sha256:
        raise OcrRevisionError(
            OcrRevisionErrorCode.STALE_PAGE_HEAD,
            "Page edits changed after the OCR revision was prepared.",
        )
    if snapshot.global_head_sha256 != request.expected_global_head_sha256:
        raise OcrRevisionError(
            OcrRevisionErrorCode.STALE_GLOBAL_HEAD,
            "Project edits changed after the OCR revision was prepared.",
        )
    try:
        effective_page = project_effective_page(
            snapshot.project,
            snapshot.ledger,
            page_id=request.page_id,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise OcrRevisionError(
            OcrRevisionErrorCode.PROJECTION_REJECTED,
            "The effective page cannot be projected for OCR.",
        ) from exc
    if (
        effective_page.hierarchy.revision_id
        != request.expected_hierarchy_revision_id
        or effective_page.hierarchy.fingerprint
        != request.expected_hierarchy_fingerprint
    ):
        raise OcrRevisionError(
            OcrRevisionErrorCode.STALE_HIERARCHY,
            "The effective hierarchy changed after the OCR revision was prepared.",
        )
    if (
        effective_page.effective_fingerprint
        != request.expected_effective_page_fingerprint
    ):
        raise OcrRevisionError(
            OcrRevisionErrorCode.STALE_EFFECTIVE_PAGE,
            "The effective page changed after the OCR revision was prepared.",
        )
    parent = _effective_parent(effective_page, request.parent_id)
    lineage = parent.lineage
    if (
        parent.origin is not ParentOrigin.USER
        or lineage is None
        or parent.root_id != request.root_id
        or lineage.root_id != request.root_id
        or lineage.authored_edit_id != request.parent_authored_edit_id
        or tuple(lineage.workflow_area_bbox) != request.sampling_bbox
        or tuple(lineage.canvas_size) != request.original_page.canvas_size
    ):
        raise OcrRevisionError(
            OcrRevisionErrorCode.PARENT_LINEAGE_MISMATCH,
            "The selected parent no longer matches its user-authored lineage.",
        )
    source_requirements = tuple(
        requirement
        for requirement in parent.stage_requirements
        if requirement.stage is RevisionStage.SOURCE
    )
    if len(source_requirements) != 1:
        raise OcrRevisionError(
            OcrRevisionErrorCode.SOURCE_NOT_RUNNABLE,
            "The selected parent has no exact SOURCE-stage requirement.",
        )
    source_requirement = source_requirements[0]
    if not (
        source_requirement.state is RevisionStageState.MISSING
        and source_requirement.required_action
        is RevisionRequiredAction.EXPLICIT_RUN
    ):
        raise OcrRevisionError(
            OcrRevisionErrorCode.SOURCE_NOT_RUNNABLE,
            "The selected parent does not require an explicit OCR run.",
        )
    return effective_page, parent


def _active_source_slot_head(
    snapshot: ProjectEditReadSnapshot,
    *,
    page_id: str,
    parent_id: str,
) -> str | None:
    candidates = tuple(
        edit
        for edit in snapshot.ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.SOURCE_TEXT
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(
        edit for edit in candidates if edit.edit_id not in superseded_ids
    )
    if len(heads) != 1:
        raise OcrRevisionError(
            OcrRevisionErrorCode.SOURCE_NOT_RUNNABLE,
            "Source revisions have competing active selection edits.",
        )
    return heads[0].edit_id


def _crop_sha256(crop: Image.Image) -> str:
    digest = sha256()
    digest.update(str(crop.mode).encode("ascii", errors="strict"))
    digest.update(int(crop.width).to_bytes(8, "big", signed=False))
    digest.update(int(crop.height).to_bytes(8, "big", signed=False))
    digest.update(crop.tobytes())
    return digest.hexdigest()


def _candidate_project_with_source_revision(
    project: Mapping[str, Any],
    artifact: OcrSourceRevisionArtifact,
) -> dict[str, Any]:
    candidate = dict(project)
    raw_catalogs = project.get("artifact_revisions")
    if not isinstance(raw_catalogs, Mapping):
        raise OcrRevisionError(
            OcrRevisionErrorCode.PROJECTION_REJECTED,
            "The artifact revision catalog is unavailable.",
        )
    catalogs = {str(key): list(value) for key, value in raw_catalogs.items()}
    catalogs.setdefault("source_revisions", []).append(artifact.to_record())
    candidate["artifact_revisions"] = catalogs
    return candidate


def _require_exact_invalidation(result: Any, parent_id: str) -> None:
    expected = {
        (
            Dependency.SOURCE,
            InvalidationAction.NEW_REVISION,
            InvalidationScope.PARENT,
            (parent_id,),
        ),
        (
            Dependency.TRANSLATION,
            InvalidationAction.RERUN,
            InvalidationScope.PARENT,
            (parent_id,),
        ),
    }
    observed = {
        (effect.dependency, effect.action, effect.scope, effect.target_ids)
        for effect in result.effects
    }
    if result.unresolved_facts or observed != expected:
        raise OcrRevisionError(
            OcrRevisionErrorCode.PROJECTION_REJECTED,
            "OCR revision invalidation differs from the source-only contract.",
        )


class ExplicitOcrRevisionService:
    """Publish one immutable OCR source artifact and its selection edit."""

    def __init__(
        self,
        *,
        project: Mapping[str, Any],
        edit_store: ProjectEditStore,
        recognition_port: OcrRevisionRecognitionPort,
        cancellation_probe: CancellationProbe | None = None,
    ) -> None:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        if not isinstance(recognition_port, OcrRevisionRecognitionPort):
            raise TypeError("recognition_port must implement OcrRevisionRecognitionPort")
        self._project = project
        self._edit_store = edit_store
        self._recognition_port = recognition_port
        self._cancellation_probe = cancellation_probe or (lambda: False)

    def _check_cancelled(self, message: str) -> None:
        if self._cancellation_probe():
            raise OcrRevisionError(OcrRevisionErrorCode.CANCELLED, message)

    def run_explicit_ocr_revision(
        self,
        request: ExplicitOcrRevisionRequest,
    ) -> ExplicitOcrRevisionReceipt:
        if not isinstance(request, ExplicitOcrRevisionRequest):
            raise TypeError("request must be an ExplicitOcrRevisionRequest")
        if self._edit_store.project_id != request.project_id:
            raise OcrRevisionError(
                OcrRevisionErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project edit store identity does not match the request.",
            )
        self._check_cancelled(
            "OCR revision was cancelled before engine initialization."
        )

        snapshot = self._edit_store.materialize_project_snapshot(
            self._project,
            page_id=request.page_id,
        )
        before_page, before_parent = _require_request_state(snapshot, request)
        actual_binding, original = _load_original_page(
            snapshot.project,
            page_id=request.page_id,
            project_path=self._edit_store.project_path,
        )
        if actual_binding != request.original_page:
            original.close()
            raise OcrRevisionError(
                OcrRevisionErrorCode.ORIGINAL_ASSET_MISMATCH,
                "The committed original-page asset changed before OCR.",
            )
        x, y, width, height = request.sampling_bbox
        crop = original.crop((x, y, x + width, y + height))
        original.close()
        crop_hash = _crop_sha256(crop)

        self._check_cancelled(
            "OCR revision was cancelled before engine initialization."
        )
        try:
            recognition = self._recognition_port.recognize(
                OcrRecognitionRequest(
                    request=request,
                    crop=crop,
                    crop_sha256=crop_hash,
                ),
                cancellation_probe=self._cancellation_probe,
            )
        finally:
            crop.close()
        if recognition.selected_ocr_engine != request.selected_ocr_engine:
            raise OcrRevisionError(
                OcrRevisionErrorCode.SETTINGS_MISMATCH,
                "OCR result engine differs from the selected run snapshot.",
            )
        if recognition.crop_sha256 != crop_hash:
            raise OcrRevisionError(
                OcrRevisionErrorCode.RECOGNITION_FAILED,
                "OCR result does not bind the exact sampled crop.",
            )
        if not recognition.authoritative:
            raise OcrRevisionError(
                OcrRevisionErrorCode.NON_AUTHORITATIVE_RESULT,
                "The selected OCR engine rejected its response as non-authoritative.",
            )
        if not recognition.text.strip():
            raise OcrRevisionError(
                OcrRevisionErrorCode.EMPTY_RESULT,
                "The selected OCR engine returned no source text.",
            )

        self._check_cancelled(
            "OCR revision was cancelled after inference; no result was published."
        )
        latest = self._edit_store.materialize_project_snapshot(
            self._project,
            page_id=request.page_id,
        )
        latest_page, latest_parent = _require_request_state(latest, request)
        latest_binding = resolve_original_page_asset_binding(
            latest.project,
            page_id=request.page_id,
            project_path=self._edit_store.project_path,
        )
        if latest_binding != request.original_page:
            raise OcrRevisionError(
                OcrRevisionErrorCode.ORIGINAL_ASSET_MISMATCH,
                "The committed original-page asset changed before persistence.",
            )
        if latest_parent != before_parent or latest_page != before_page:
            raise OcrRevisionError(
                OcrRevisionErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective parent state changed before persistence.",
            )

        selection_edit_id = OCR_SOURCE_SELECTION_EDIT_ID_PREFIX + canonical_sha256(
            {
                "command_id": request.command_id,
                "project_id": request.project_id,
                "page_id": request.page_id,
                "parent_id": request.parent_id,
            }
        )
        if latest.ledger.get(selection_edit_id) is not None:
            raise OcrRevisionError(
                OcrRevisionErrorCode.PERSISTENCE_REJECTED,
                "The OCR revision command was already recorded.",
            )
        artifact = OcrSourceRevisionArtifact(
            command_id=request.command_id,
            page_id=request.page_id,
            parent_id=request.parent_id,
            root_id=request.root_id,
            parent_authored_edit_id=request.parent_authored_edit_id,
            selection_edit_id=selection_edit_id,
            source_text=recognition.text,
            confidence=recognition.confidence,
            original_page=request.original_page,
            sampling_bbox=request.sampling_bbox,
            crop_sha256=crop_hash,
            run_settings_fingerprint=request.run_settings_fingerprint,
            selected_ocr_engine=request.selected_ocr_engine,
            hierarchy_revision_id=latest_page.hierarchy.revision_id,
            hierarchy_fingerprint=latest_page.hierarchy.fingerprint,
            input_effective_page_fingerprint=latest_page.effective_fingerprint,
            backend_name=recognition.backend_name,
            backend_metadata=recognition.backend_metadata,
            recognition_metadata=recognition.recognition_metadata,
        )
        slot_head = _active_source_slot_head(
            latest,
            page_id=request.page_id,
            parent_id=request.parent_id,
        )
        selection_edit = create_project_edit(
            project_id=request.project_id,
            page_id=request.page_id,
            target=EditTarget(
                EditTargetKind.PARENT,
                parent_id=request.parent_id,
            ),
            domain=EditDomain.SOURCE_TEXT,
            operation="select_revision",
            payload={"revision_id": artifact.revision_id},
            base_revision_id=latest_page.hierarchy.revision_id,
            base_fingerprint=latest_page.effective_fingerprint,
            supersedes_edit_id=slot_head,
            edit_id=selection_edit_id,
        )
        candidate_project = _candidate_project_with_source_revision(
            latest.project,
            artifact,
        )
        candidate_ledger = latest.ledger.append(selection_edit)
        try:
            candidate_page = project_effective_page(
                candidate_project,
                candidate_ledger,
                page_id=request.page_id,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OcrRevisionError(
                OcrRevisionErrorCode.PROJECTION_REJECTED,
                "The OCR source revision was rejected by the effective projector.",
            ) from exc
        candidate_parent = _effective_parent(candidate_page, request.parent_id)
        if (
            selection_edit.edit_id not in candidate_page.applied_edit_ids
            or candidate_parent.source_text != recognition.text
            or candidate_parent.source_authority != "ocr_revision"
            or candidate_parent.source_revision_id != artifact.revision_id
            or candidate_page.hierarchy.revision_id
            != latest_page.hierarchy.revision_id
            or candidate_page.hierarchy.fingerprint
            != latest_page.hierarchy.fingerprint
        ):
            raise OcrRevisionError(
                OcrRevisionErrorCode.PROJECTION_REJECTED,
                "The effective projector did not select the exact OCR revision.",
            )
        before_requirements = {
            requirement.stage: requirement
            for requirement in latest_parent.stage_requirements
        }
        after_requirements = {
            requirement.stage: requirement
            for requirement in candidate_parent.stage_requirements
        }
        if (
            after_requirements[RevisionStage.SOURCE].state
            is not RevisionStageState.CURRENT
            or after_requirements[RevisionStage.SOURCE].required_action
            is not RevisionRequiredAction.NONE
            or after_requirements[RevisionStage.TRANSLATION].state
            is not RevisionStageState.MISSING
            or after_requirements[RevisionStage.TRANSLATION].required_action
            is not RevisionRequiredAction.EXPLICIT_RUN
            or any(
                before_requirements[stage] != after_requirements[stage]
                for stage in before_requirements
                if stage not in {RevisionStage.SOURCE, RevisionStage.TRANSLATION}
            )
        ):
            raise OcrRevisionError(
                OcrRevisionErrorCode.PROJECTION_REJECTED,
                "OCR revision stage requirements differ from the source-only contract.",
            )
        invalidation = invalidation_for_edit(selection_edit)
        _require_exact_invalidation(invalidation, request.parent_id)

        self._check_cancelled(
            "OCR revision was cancelled before persistence; no result was published."
        )
        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (selection_edit,),
                automatic_page_sha256=latest_page.automatic_fingerprint,
                expected_page_head_sha256=latest.page_head_sha256,
                expected_global_head_sha256=latest.global_head_sha256,
                artifact_revisions=(artifact.to_record(include_catalog=True),),
                transaction_id=request.command_id,
            )
        except StalePageEditHeadError as exc:
            raise OcrRevisionError(
                OcrRevisionErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before the OCR revision committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise OcrRevisionError(
                OcrRevisionErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before the OCR revision committed.",
            ) from exc
        except Exception as exc:
            raise OcrRevisionError(
                OcrRevisionErrorCode.PERSISTENCE_REJECTED,
                "The OCR artifact-plus-selection transaction was rejected.",
            ) from exc

        # The commit receipt is the durable transaction boundary.  Do not
        # reopen mutable sidecar state here: another valid writer may commit
        # after this transaction and must neither redefine this receipt nor
        # turn the already-durable transaction into a reported failure.  The
        # exact candidate state was projected and validated before the CAS.
        return ExplicitOcrRevisionReceipt(
            command_id=request.command_id,
            project_id=request.project_id,
            page_id=request.page_id,
            parent_id=request.parent_id,
            root_id=request.root_id,
            parent_authored_edit_id=request.parent_authored_edit_id,
            source_revision_id=artifact.revision_id,
            selection_edit_id=selection_edit.edit_id,
            source_text=recognition.text,
            confidence=recognition.confidence,
            selected_ocr_engine=request.selected_ocr_engine,
            backend_name=recognition.backend_name,
            backend_metadata=recognition.backend_metadata,
            recognition_metadata=recognition.recognition_metadata,
            original_page=request.original_page,
            sampling_bbox=request.sampling_bbox,
            crop_sha256=crop_hash,
            run_settings_fingerprint=request.run_settings_fingerprint,
            hierarchy_revision_id=candidate_page.hierarchy.revision_id,
            hierarchy_fingerprint=candidate_page.hierarchy.fingerprint,
            before_effective_page_fingerprint=latest_page.effective_fingerprint,
            after_effective_page_fingerprint=candidate_page.effective_fingerprint,
            invalidation=invalidation.to_dict(),
            stage_requirements=tuple(
                requirement.to_dict()
                for requirement in candidate_parent.stage_requirements
            ),
            commit_receipt=commit_receipt.to_dict(),
        )


__all__ = [
    "ExplicitOcrRevisionService",
    "resolve_original_page_asset_binding",
]
