# -*- coding: utf-8 -*-
"""Atomic application service for one explicit target translation revision."""
from __future__ import annotations

import os
from typing import Any, Mapping

from app.config.run_settings_compiler import materialize_pipeline_settings_snapshot
from app.config.settings_contracts import RunSettingsSnapshot
from app.io.project import read_project_settings
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
from app.pipeline.ocr_revision_contracts import OcrSourceRevisionArtifact
from app.pipeline.translation_revision_contracts import (
    CancellationProbe,
    ExplicitTranslationRevisionReceipt,
    ExplicitTranslationRevisionRequest,
    TRANSLATION_SELECTION_EDIT_ID_PREFIX,
    TranslationExecutionReceipt,
    TranslationExecutionRequest,
    TranslationPolicySnapshots,
    TranslationProviderSelection,
    TranslationRevisionArtifact,
    TranslationRevisionError,
    TranslationRevisionErrorCode,
    TranslationRevisionExecutionPort,
    translation_context_fingerprint,
    translation_glossary_fingerprint,
    translation_policy_region_type,
)

from .contracts import EditDomain, EditTarget, EditTargetKind, create_project_edit
from .fingerprints import canonical_sha256, project_id_for
from .invalidation import (
    Dependency,
    InvalidationAction,
    InvalidationScope,
    invalidation_for_edit,
)
from .glossary_commands import project_glossary_snapshot
from .ledger import ProjectEditLedger
from .projection import (
    EffectivePageSnapshot,
    EffectiveParentSnapshot,
    effective_source_fingerprint,
    project_effective_page,
)


def _effective_translation_glossary(
    project: Mapping[str, Any],
    style_guide: Mapping[str, Any],
) -> dict[str, Any]:
    """Overlay typed project glossary edits onto the existing policy shape.

    The translation owner continues to receive its established style-guide
    mapping.  Aliases are represented as ordinary source entries; no matcher,
    prompt, priority, or provider behavior is changed here.
    """

    embedded = project.get("edit_ledger")
    if not isinstance(embedded, Mapping):
        raise ValueError("embedded project edit ledger is unavailable")
    ledger = ProjectEditLedger.from_dict(embedded)
    snapshot = project_glossary_snapshot(project, ledger)
    if not snapshot.entries:
        return dict(style_guide)
    current_items = style_guide.get("glossary", ())
    if (
        not isinstance(current_items, (list, tuple))
        or any(not isinstance(item, Mapping) for item in current_items)
    ):
        raise ValueError("existing style-guide glossary must be a list")
    replacement_sources = {
        value.strip()
        for entry in snapshot.entries
        for value in (entry.source, *entry.aliases)
    }
    merged = [
        dict(item)
        for item in current_items
        if str(item.get("source") or "").strip() not in replacement_sources
    ]
    for entry in snapshot.entries:
        for source in (entry.source, *entry.aliases):
            merged.append(
                {
                    "source": source,
                    "target": entry.target,
                    "priority": entry.priority,
                }
            )
    result = dict(style_guide)
    result["glossary"] = merged
    return result


def _project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = tuple(
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    )
    if len(matches) != 1:
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is not exact: {page_id}",
        )
    return matches[0]


def compile_translation_revision_policy_snapshots(
    project: Mapping[str, Any],
    *,
    page_id: str,
    run_settings_snapshot: RunSettingsSnapshot,
) -> TranslationPolicySnapshots:
    """Compile exact existing glossary and committed prior-page context.

    This boundary is read-only.  It deliberately does not create a glossary,
    run automatic discovery, ping a provider, or inspect the current/future
    page for context.
    """

    if not isinstance(project, Mapping):
        raise TypeError("project must be a mapping")
    if not isinstance(run_settings_snapshot, RunSettingsSnapshot):
        raise TypeError("run_settings_snapshot must be a RunSettingsSnapshot")
    _project_page(project, page_id)
    try:
        settings = materialize_pipeline_settings_snapshot(run_settings_snapshot)
    except (TypeError, ValueError) as exc:
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.SETTINGS_MISMATCH,
            "The immutable run settings cannot be materialized.",
        ) from exc

    from app.pipeline import controller as translation_owner

    style_guide_path = str(getattr(settings, "style_guide_path", "") or "")
    if bool(getattr(settings, "auto_glossary", False)) and not style_guide_path:
        style_guide_path = os.path.join(
            str(getattr(settings, "export_dir", "") or ""),
            "style_guide.json",
        )
    glossary = translation_owner._load_style_guide(
        style_guide_path,
        str(getattr(settings, "target_lang", "") or ""),
    )
    if not isinstance(glossary, Mapping):
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.GLOSSARY_MISMATCH,
            "The existing translation glossary is invalid.",
        )
    try:
        glossary_snapshot = _effective_translation_glossary(project, glossary)
    except (KeyError, TypeError, ValueError) as exc:
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.GLOSSARY_MISMATCH,
            "The effective project glossary is invalid.",
        ) from exc

    context_enabled = bool(
        getattr(settings, "translator_backend", "") == "GGUF"
        and getattr(settings, "target_lang", "") == "Simplified Chinese"
        and bool(getattr(settings, "gguf_cross_page_context", False))
    )
    context_lines: list[str] = []
    pages = project.get("pages") or ()
    target_seen = False
    if context_enabled:
        for page in pages:
            if not isinstance(page, Mapping):
                raise TranslationRevisionError(
                    TranslationRevisionErrorCode.CONTEXT_MISMATCH,
                    "Committed project pages are invalid.",
                )
            candidate_page_id = str(page.get("page_id") or "").strip()
            if candidate_page_id == page_id:
                target_seen = True
                break
            page_class = str(page.get("page_class") or "normal")
            regions = page.get("regions") or page.get("blocks") or ()
            if not isinstance(regions, (list, tuple)):
                raise TranslationRevisionError(
                    TranslationRevisionErrorCode.CONTEXT_MISMATCH,
                    "Committed prior-page regions are invalid.",
                )
            for region in regions:
                if not isinstance(region, Mapping):
                    raise TranslationRevisionError(
                        TranslationRevisionErrorCode.CONTEXT_MISMATCH,
                        "Committed prior-page region records are invalid.",
                    )
                record = dict(region)
                if not translation_owner._region_can_feed_context(
                    record,
                    page_class,
                ):
                    continue
                translation = str(record.get("translation") or "").strip()
                if translation:
                    context_lines.append(translation)
            context_lines = context_lines[-4:]
        if not target_seen:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.PAGE_NOT_FOUND,
                "The selected page is not in the committed page order.",
            )
    context = tuple(context_lines[-4:])
    return TranslationPolicySnapshots(
        glossary_snapshot=glossary_snapshot,
        glossary_fingerprint=translation_glossary_fingerprint(
            glossary_snapshot
        ),
        prior_page_context=context,
        context_fingerprint=translation_context_fingerprint(context),
    )


def _effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    matches = tuple(
        parent for parent in snapshot.parents if parent.parent_id == parent_id
    )
    if len(matches) != 1:
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.PARENT_NOT_FOUND,
            f"Effective parent identity is not exact: {parent_id}",
        )
    return matches[0]


def _require_stage(
    parent: EffectiveParentSnapshot,
    *,
    stage: RevisionStage,
    state: RevisionStageState,
    action: RevisionRequiredAction,
) -> None:
    matches = tuple(
        requirement
        for requirement in parent.stage_requirements
        if requirement.stage is stage
    )
    if len(matches) != 1 or not (
        matches[0].state is state and matches[0].required_action is action
    ):
        code = (
            TranslationRevisionErrorCode.SOURCE_NOT_CURRENT
            if stage is RevisionStage.SOURCE
            else TranslationRevisionErrorCode.PROJECTION_REJECTED
        )
        raise TranslationRevisionError(
            code,
            f"The selected parent has no exact {stage.value} stage state.",
        )


def _source_artifact(parent: EffectiveParentSnapshot) -> OcrSourceRevisionArtifact:
    try:
        artifact = OcrSourceRevisionArtifact.from_record(
            dict(parent.source_revision_metadata)
        )
    except (TypeError, ValueError) as exc:
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.SOURCE_MISMATCH,
            "The selected OCR source artifact is unavailable.",
        ) from exc
    return artifact


def _require_request_state(
    snapshot: ProjectEditReadSnapshot,
    request: ExplicitTranslationRevisionRequest,
) -> tuple[EffectivePageSnapshot, EffectiveParentSnapshot]:
    if project_id_for(snapshot.project) != request.project_id:
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.PROJECT_IDENTITY_MISMATCH,
            "Project identity does not match the translation request.",
        )
    if snapshot.page_head_sha256 != request.expected_page_head_sha256:
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.STALE_PAGE_HEAD,
            "Page edits changed after the translation was prepared.",
        )
    if snapshot.global_head_sha256 != request.expected_global_head_sha256:
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.STALE_GLOBAL_HEAD,
            "Project edits changed after the translation was prepared.",
        )
    try:
        effective_page = project_effective_page(
            snapshot.project,
            snapshot.ledger,
            page_id=request.page_id,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.PROJECTION_REJECTED,
            "The effective page cannot be projected for translation.",
        ) from exc
    if (
        effective_page.hierarchy.revision_id
        != request.expected_hierarchy_revision_id
        or effective_page.hierarchy.fingerprint
        != request.expected_hierarchy_fingerprint
    ):
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.STALE_HIERARCHY,
            "The effective hierarchy changed after preparation.",
        )
    if (
        effective_page.effective_fingerprint
        != request.expected_effective_page_fingerprint
    ):
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.STALE_EFFECTIVE_PAGE,
            "The effective page changed after preparation.",
        )
    parent = _effective_parent(effective_page, request.parent_id)
    lineage = parent.lineage
    if (
        parent.origin is not ParentOrigin.USER
        or lineage is None
        or parent.root_id != request.root_id
        or lineage.root_id != request.root_id
        or lineage.authored_edit_id != request.parent_authored_edit_id
        or parent.role != request.parent_role
        or translation_policy_region_type(parent.role)
        != request.policy_region_type
        or request.bubble_local_nested_speech
    ):
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.PARENT_LINEAGE_MISMATCH,
            "The selected parent no longer matches its typed user lineage.",
        )
    source_fingerprint = effective_source_fingerprint(
        parent.parent_id,
        parent.source_text,
    )
    source_artifact = _source_artifact(parent)
    if (
        parent.source_text != request.effective_source_text
        or parent.source_authority != request.effective_source_authority
        or source_fingerprint != request.effective_source_fingerprint
        or parent.source_revision_id != request.source_revision_id
        or source_artifact.revision_id != request.source_revision_id
        or source_artifact.selection_edit_id
        != request.source_selection_edit_id
        or source_artifact.source_text != request.effective_source_text
        or source_artifact.page_id != request.page_id
        or source_artifact.parent_id != request.parent_id
        or source_artifact.root_id != request.root_id
        or source_artifact.parent_authored_edit_id
        != request.parent_authored_edit_id
        or request.source_selection_edit_id not in parent.applied_edit_ids
    ):
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.SOURCE_MISMATCH,
            "The selected OCR source binding changed after preparation.",
        )
    _require_stage(
        parent,
        stage=RevisionStage.SOURCE,
        state=RevisionStageState.CURRENT,
        action=RevisionRequiredAction.NONE,
    )
    _require_stage(
        parent,
        stage=RevisionStage.TRANSLATION,
        state=RevisionStageState.MISSING,
        action=RevisionRequiredAction.EXPLICIT_RUN,
    )
    try:
        settings_state = read_project_settings(snapshot.project)
    except (KeyError, TypeError, ValueError) as exc:
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.SETTINGS_MISMATCH,
            "Current project run settings are unavailable.",
        ) from exc
    if settings_state.last_run_snapshot != request.run_settings_snapshot:
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.SETTINGS_MISMATCH,
            "Current project run settings differ from the request.",
        )
    expected_provider = TranslationProviderSelection.from_run_settings_snapshot(
        request.run_settings_snapshot
    )
    if expected_provider != request.provider:
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.SETTINGS_MISMATCH,
            "Current provider selection differs from the request.",
        )
    compiled = compile_translation_revision_policy_snapshots(
        snapshot.project,
        page_id=request.page_id,
        run_settings_snapshot=request.run_settings_snapshot,
    )
    if (
        compiled.glossary_snapshot != request.glossary_snapshot
        or compiled.glossary_fingerprint != request.glossary_fingerprint
    ):
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.GLOSSARY_MISMATCH,
            "Existing style-guide content changed after preparation.",
        )
    if (
        compiled.prior_page_context != request.prior_page_context
        or compiled.context_fingerprint != request.context_fingerprint
    ):
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.CONTEXT_MISMATCH,
            "Committed prior-page context changed after preparation.",
        )
    return effective_page, parent


def _active_target_slot_head(
    snapshot: ProjectEditReadSnapshot,
    *,
    page_id: str,
    parent_id: str,
) -> str | None:
    candidates = tuple(
        edit
        for edit in snapshot.ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.TARGET_TEXT
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
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.PROJECTION_REJECTED,
            "Target revisions have competing active selection edits.",
        )
    return heads[0].edit_id


def _candidate_project_with_translation_revision(
    project: Mapping[str, Any],
    artifact: TranslationRevisionArtifact,
) -> dict[str, Any]:
    candidate = dict(project)
    raw_catalogs = project.get("artifact_revisions")
    if not isinstance(raw_catalogs, Mapping):
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.PROJECTION_REJECTED,
            "The artifact revision catalog is unavailable.",
        )
    catalogs = {str(key): list(value) for key, value in raw_catalogs.items()}
    catalogs.setdefault("translation_revisions", []).append(
        artifact.to_record()
    )
    candidate["artifact_revisions"] = catalogs
    return candidate


def _require_exact_invalidation(result: Any, parent_id: str) -> None:
    expected = {
        (
            Dependency.TRANSLATION,
            InvalidationAction.NEW_REVISION,
            InvalidationScope.PARENT,
            (parent_id,),
        ),
        (
            Dependency.LAYOUT_RENDER,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PARENT,
            (parent_id,),
        ),
        (
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            (parent_id,),
        ),
    }
    observed = {
        (effect.dependency, effect.action, effect.scope, effect.target_ids)
        for effect in result.effects
    }
    if result.unresolved_facts or observed != expected:
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.PROJECTION_REJECTED,
            "Translation revision invalidation differs from the target-only contract.",
        )


class ExplicitTranslationRevisionService:
    """Publish one immutable translation artifact and its selection edit."""

    def __init__(
        self,
        *,
        project: Mapping[str, Any],
        edit_store: ProjectEditStore,
        translation_port: TranslationRevisionExecutionPort,
        cancellation_probe: CancellationProbe | None = None,
    ) -> None:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        if not isinstance(translation_port, TranslationRevisionExecutionPort):
            raise TypeError(
                "translation_port must implement TranslationRevisionExecutionPort"
            )
        self._project = project
        self._edit_store = edit_store
        self._translation_port = translation_port
        self._cancellation_probe = cancellation_probe or (lambda: False)

    def _check_cancelled(self, message: str) -> None:
        if self._cancellation_probe():
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.CANCELLED,
                message,
            )

    def run_explicit_translation_revision(
        self,
        request: ExplicitTranslationRevisionRequest,
    ) -> ExplicitTranslationRevisionReceipt:
        if not isinstance(request, ExplicitTranslationRevisionRequest):
            raise TypeError(
                "request must be an ExplicitTranslationRevisionRequest"
            )
        if self._edit_store.project_id != request.project_id:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project edit store identity does not match the request.",
            )
        self._check_cancelled(
            "Translation was cancelled before provider initialization."
        )
        snapshot = self._edit_store.materialize_project_snapshot(
            self._project,
            page_id=request.page_id,
        )
        before_page, before_parent = _require_request_state(snapshot, request)
        try:
            execution = self._translation_port.translate(
                TranslationExecutionRequest(request=request),
                cancellation_probe=self._cancellation_probe,
            )
        except TranslationRevisionError:
            raise
        except Exception as exc:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.TRANSLATION_FAILED,
                "The selected translation provider failed.",
            ) from exc
        if not isinstance(execution, TranslationExecutionReceipt):
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.TRANSLATION_FAILED,
                "The translation provider returned an invalid receipt.",
            )
        if (
            execution.source_fingerprint
            != request.effective_source_fingerprint
            or execution.run_settings_fingerprint
            != request.run_settings_fingerprint
            or execution.provider != request.provider
            or execution.glossary_fingerprint
            != request.glossary_fingerprint
            or execution.context_fingerprint
            != request.context_fingerprint
        ):
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.TRANSLATION_FAILED,
                "The translation result does not bind the exact request.",
            )
        if not execution.target_text.strip():
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.EMPTY_RESULT,
                "The selected translation provider returned no target text.",
            )
        self._check_cancelled(
            "Translation completed after cancellation; no result was published."
        )

        latest = self._edit_store.materialize_project_snapshot(
            self._project,
            page_id=request.page_id,
        )
        latest_page, latest_parent = _require_request_state(latest, request)
        if latest_parent != before_parent or latest_page != before_page:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective parent state changed before persistence.",
            )

        selection_edit_id = (
            TRANSLATION_SELECTION_EDIT_ID_PREFIX
            + canonical_sha256(
                {
                    "command_id": request.command_id,
                    "project_id": request.project_id,
                    "page_id": request.page_id,
                    "parent_id": request.parent_id,
                    "source_fingerprint": request.effective_source_fingerprint,
                }
            )
        )
        if latest.ledger.get(selection_edit_id) is not None:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.PERSISTENCE_REJECTED,
                "The translation command was already recorded.",
            )
        artifact = TranslationRevisionArtifact(
            command_id=request.command_id,
            project_id=request.project_id,
            page_id=request.page_id,
            parent_id=request.parent_id,
            root_id=request.root_id,
            parent_authored_edit_id=request.parent_authored_edit_id,
            parent_role=request.parent_role,
            policy_region_type=request.policy_region_type,
            bubble_local_nested_speech=request.bubble_local_nested_speech,
            selection_edit_id=selection_edit_id,
            target_text=execution.target_text,
            source_text=request.effective_source_text,
            source_authority=request.effective_source_authority,
            source_fingerprint=request.effective_source_fingerprint,
            source_revision_id=request.source_revision_id,
            source_selection_edit_id=request.source_selection_edit_id,
            run_settings_snapshot=request.run_settings_snapshot,
            run_settings_fingerprint=request.run_settings_fingerprint,
            provider=request.provider,
            glossary_snapshot=request.glossary_snapshot,
            glossary_fingerprint=request.glossary_fingerprint,
            prior_page_context=request.prior_page_context,
            context_fingerprint=request.context_fingerprint,
            hierarchy_revision_id=latest_page.hierarchy.revision_id,
            hierarchy_fingerprint=latest_page.hierarchy.fingerprint,
            input_effective_page_fingerprint=latest_page.effective_fingerprint,
            policy_metadata=execution.policy_metadata,
            quality_warnings=execution.quality_warnings,
        )
        selection_edit = create_project_edit(
            project_id=request.project_id,
            page_id=request.page_id,
            target=EditTarget(
                EditTargetKind.PARENT,
                parent_id=request.parent_id,
            ),
            domain=EditDomain.TARGET_TEXT,
            operation="select_revision",
            payload={
                "revision_id": artifact.revision_id,
                "source_fingerprint": request.effective_source_fingerprint,
            },
            base_revision_id=latest_page.hierarchy.revision_id,
            base_fingerprint=latest_page.effective_fingerprint,
            supersedes_edit_id=_active_target_slot_head(
                latest,
                page_id=request.page_id,
                parent_id=request.parent_id,
            ),
            edit_id=selection_edit_id,
        )
        candidate_project = _candidate_project_with_translation_revision(
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
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.PROJECTION_REJECTED,
                "The translation revision was rejected by the projector.",
            ) from exc
        candidate_parent = _effective_parent(candidate_page, request.parent_id)
        if (
            selection_edit.edit_id not in candidate_page.applied_edit_ids
            or candidate_parent.target_text != execution.target_text
            or candidate_parent.target_authority != "translation_revision"
            or candidate_parent.target_revision_id != artifact.revision_id
            or candidate_parent.target_freshness.value != "current"
            or candidate_parent.source_text != latest_parent.source_text
            or candidate_parent.source_revision_id
            != latest_parent.source_revision_id
            or candidate_page.hierarchy != latest_page.hierarchy
        ):
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.PROJECTION_REJECTED,
                "The projector did not select the exact translation revision.",
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
            is not RevisionStageState.CURRENT
            or after_requirements[RevisionStage.TRANSLATION].required_action
            is not RevisionRequiredAction.NONE
            or any(
                before_requirements[stage] != after_requirements[stage]
                for stage in before_requirements
                if stage is not RevisionStage.TRANSLATION
            )
        ):
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.PROJECTION_REJECTED,
                "Translation stage requirements differ from the target-only contract.",
            )
        invalidation = invalidation_for_edit(selection_edit)
        _require_exact_invalidation(invalidation, request.parent_id)
        self._check_cancelled(
            "Translation was cancelled before persistence; no result was published."
        )
        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (selection_edit,),
                automatic_page_sha256=latest_page.automatic_fingerprint,
                expected_page_head_sha256=latest.page_head_sha256,
                expected_global_head_sha256=latest.global_head_sha256,
                artifact_revisions=(
                    artifact.to_record(include_catalog=True),
                ),
                transaction_id=request.command_id,
            )
        except StalePageEditHeadError as exc:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before the translation committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before the translation committed.",
            ) from exc
        except Exception as exc:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.PERSISTENCE_REJECTED,
                "The translation artifact-plus-selection transaction was rejected.",
            ) from exc

        return ExplicitTranslationRevisionReceipt(
            command_id=request.command_id,
            project_id=request.project_id,
            page_id=request.page_id,
            parent_id=request.parent_id,
            root_id=request.root_id,
            parent_authored_edit_id=request.parent_authored_edit_id,
            parent_role=request.parent_role,
            policy_region_type=request.policy_region_type,
            bubble_local_nested_speech=request.bubble_local_nested_speech,
            translation_revision_id=artifact.revision_id,
            selection_edit_id=selection_edit.edit_id,
            target_text=execution.target_text,
            source_text=request.effective_source_text,
            source_authority=request.effective_source_authority,
            source_fingerprint=request.effective_source_fingerprint,
            source_revision_id=request.source_revision_id,
            source_selection_edit_id=request.source_selection_edit_id,
            run_settings_snapshot=request.run_settings_snapshot,
            run_settings_fingerprint=request.run_settings_fingerprint,
            provider=request.provider,
            glossary_snapshot=request.glossary_snapshot,
            glossary_fingerprint=request.glossary_fingerprint,
            prior_page_context=request.prior_page_context,
            context_fingerprint=request.context_fingerprint,
            hierarchy_revision_id=candidate_page.hierarchy.revision_id,
            hierarchy_fingerprint=candidate_page.hierarchy.fingerprint,
            before_effective_page_fingerprint=latest_page.effective_fingerprint,
            after_effective_page_fingerprint=candidate_page.effective_fingerprint,
            policy_metadata=execution.policy_metadata,
            quality_warnings=execution.quality_warnings,
            invalidation=invalidation.to_dict(),
            stage_requirements=tuple(
                requirement.to_dict()
                for requirement in candidate_parent.stage_requirements
            ),
            commit_receipt=commit_receipt.to_dict(),
        )


__all__ = [
    "ExplicitTranslationRevisionService",
    "compile_translation_revision_policy_snapshots",
]
