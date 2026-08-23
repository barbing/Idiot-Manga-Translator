# -*- coding: utf-8 -*-
"""Native outer-box geometry contracts for the Hybrid Pro Qt shell.

The portable token table mirrors the accepted web prototype key-for-key. Qt
Style Sheets, however, apply width minima to a widget's content box and then add
padding and borders. Native composite controls therefore need a separate,
explicit outer-box contract so host arithmetic and interactive rectangles never
depend on QSS box-model side effects.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProductNavigationGeometry:
    """Outer geometry for the centered four-route application navigation."""

    header_height: int = 54
    route_count: int = 4
    route_gap: int = 8
    nominal_route_width: int = 126
    enlarged_route_width: int = 144
    compact_route_width: int = 72
    icon_size: int = 17
    side_column_minimum: int = 300
    nominal_minimum_shell_width: int = 1160
    enlarged_minimum_shell_width: int = 1240

    def __post_init__(self) -> None:
        values = (
            self.header_height,
            self.route_count,
            self.nominal_route_width,
            self.enlarged_route_width,
            self.compact_route_width,
            self.icon_size,
            self.side_column_minimum,
            self.nominal_minimum_shell_width,
            self.enlarged_minimum_shell_width,
        )
        if any(value <= 0 for value in values) or self.route_gap < 0:
            raise ValueError("product-navigation geometry must be positive")

    def route_width(self, *, compact: bool, enlarged: bool = False) -> int:
        if compact:
            return self.compact_route_width
        return self.enlarged_route_width if enlarged else self.nominal_route_width

    def outer_width(self, *, compact: bool, enlarged: bool = False) -> int:
        return (
            self.route_count
            * self.route_width(compact=compact, enlarged=enlarged)
            + (self.route_count - 1) * self.route_gap
        )

    def minimum_shell_width(self, *, enlarged: bool) -> int:
        return (
            self.enlarged_minimum_shell_width
            if enlarged
            else self.nominal_minimum_shell_width
        )


@dataclass(frozen=True, slots=True)
class ModulePolicyGeometry:
    """Shared columns for every module-policy card on the Settings surface."""

    index_width: int = 24
    expanded_identity_width: int = 224
    compact_identity_width: int = 184
    row_gap: int = 12
    form_horizontal_spacing: int = 12
    form_vertical_spacing: int = 6
    minimum_label_width: int = 72
    minimum_status_width: int = 104

    def __post_init__(self) -> None:
        values = (
            self.index_width,
            self.expanded_identity_width,
            self.compact_identity_width,
            self.row_gap,
            self.form_horizontal_spacing,
            self.form_vertical_spacing,
            self.minimum_label_width,
            self.minimum_status_width,
        )
        if any(value <= 0 for value in values):
            raise ValueError("module-policy geometry must be positive")

    def identity_width(self, *, compact: bool) -> int:
        return self.compact_identity_width if compact else self.expanded_identity_width


PRODUCT_NAVIGATION_GEOMETRY = ProductNavigationGeometry()
MODULE_POLICY_GEOMETRY = ModulePolicyGeometry()


__all__ = [
    "MODULE_POLICY_GEOMETRY",
    "PRODUCT_NAVIGATION_GEOMETRY",
    "ModulePolicyGeometry",
    "ProductNavigationGeometry",
]
