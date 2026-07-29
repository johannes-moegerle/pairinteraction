# SPDX-FileCopyrightText: 2025 PairInteraction Developers
# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from PySide6.QtCore import QLocale, Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QLabel, QLineEdit

from pairinteraction_gui.qobjects import WidgetH

if TYPE_CHECKING:
    from pairinteraction_gui.plotwidget.plotwidget import PlotWidget

logger = logging.getLogger(__name__)

KEYS = ("xmin", "xmax", "ymin", "ymax")


class AxisRangeWidget(WidgetH):
    """Input fields to display and modify the x and y range of a plot."""

    spacing = 4

    # Emitted (possibly from a worker thread) to update the fields inside the gui thread.
    signal_limits_changed = Signal(float, float, float, float)

    def __init__(self, plotwidget: PlotWidget) -> None:
        """Initialize the range fields for the given plotwidget."""
        self.plotwidget = plotwidget
        super().__init__(plotwidget, name="PlotRangeWidget")

    def setupWidget(self) -> None:
        self._is_updating = False
        self.fields: dict[str, QLineEdit] = {}

        validator = QDoubleValidator(self)
        validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        validator.setLocale(QLocale.c())  # always use "." as decimal separator

        for axis in ("x", "y"):
            if axis == "y":
                self.layout().addSpacing(10)
            label = QLabel(f"{axis} limits:", self)
            label.setObjectName("PlotRangeLabel")
            self.layout().addWidget(label)
            for bound in ("min", "max"):
                field = QLineEdit(self)
                field.setObjectName(f"PlotRange_{axis}{bound}")
                field.setValidator(validator)
                field.setAlignment(Qt.AlignmentFlag.AlignCenter)
                field.setFixedWidth(62)
                field.setToolTip(f"{bound.capitalize()}imum of the plotted {axis} range")
                field.editingFinished.connect(self.apply_to_axes)
                self.layout().addWidget(field)
                self.fields[axis + bound] = field

    def postSetupWidget(self) -> None:
        self.signal_limits_changed.connect(self._on_limits_changed)
        self.update_fields()

    def sync_from_axes(self) -> None:
        """Request an update of the fields from the current axes limits (may be called from any thread)."""
        x_min, x_max = self.plotwidget.canvas.ax.get_xlim()
        y_min, y_max = self.plotwidget.canvas.ax.get_ylim()
        self.signal_limits_changed.emit(x_min, x_max, y_min, y_max)

    def _on_limits_changed(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        if self._is_updating:
            return
        values = dict(zip(KEYS, (x_min, x_max, y_min, y_max), strict=True))
        for key, field in self.fields.items():
            if field.hasFocus() and field.isModified():
                continue  # dont overwrite a value the user is currently typing
            field.setText(format_value(values[key]))

    def update_fields(self) -> None:
        """Set all fields to the current axes limits (must be called from the gui thread)."""
        x_min, x_max = self.plotwidget.canvas.ax.get_xlim()
        y_min, y_max = self.plotwidget.canvas.ax.get_ylim()
        for key, value in zip(KEYS, (x_min, x_max, y_min, y_max), strict=True):
            self.fields[key].setText(format_value(value))

    def apply_to_axes(self) -> None:
        """Zoom the plot to the ranges given in the fields."""
        if self._is_updating:
            return

        for field in self.fields.values():
            field.setModified(False)  # the editing is done, the fields may be updated from the axes again

        try:
            values = {key: float(self.fields[key].text()) for key in KEYS}
        except ValueError:
            self.update_fields()  # revert invalid input
            return

        if not all(math.isfinite(v) for v in values.values()):
            self.update_fields()
            return
        if values["xmin"] >= values["xmax"] or values["ymin"] >= values["ymax"]:
            logger.debug("Ignoring plot range with min >= max.")
            self.update_fields()
            return

        ax = self.plotwidget.canvas.ax
        new_xlim = (values["xmin"], values["xmax"])
        new_ylim = (values["ymin"], values["ymax"])
        if new_xlim == ax.get_xlim() and new_ylim == ax.get_ylim():
            return

        self._is_updating = True
        try:
            # set_xlim/set_ylim also disable autoscaling, so the range is kept on the next draw
            ax.set_xlim(*new_xlim)
            ax.set_ylim(*new_ylim)
        finally:
            self._is_updating = False

        self.plotwidget.canvas.draw_idle()
        # make the new view part of the zoom history, so the back/forward/home buttons still work as expected
        self.plotwidget.navigation_toolbar.push_current()
        self.update_fields()


def format_value(value: float) -> str:
    """Format an axis limit for display in a range field."""
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"
