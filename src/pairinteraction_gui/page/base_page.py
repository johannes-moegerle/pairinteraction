# SPDX-FileCopyrightText: 2025 PairInteraction Developers
# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import nbformat
from nbconvert import PythonExporter
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QStyle,
    QToolBox,
    QWidget,
)

import pairinteraction
from pairinteraction_gui.config import BaseConfig
from pairinteraction_gui.plotwidget.plotwidget import PlotEnergies
from pairinteraction_gui.qobjects import NamedStackedWidget, WidgetV, show_status_tip
from pairinteraction_gui.worker import MultiThreadWorker

if TYPE_CHECKING:
    from collections.abc import Callable

    from PySide6.QtGui import QHideEvent, QShowEvent

    from pairinteraction_gui.calculate.calculate_base import Parameters, Results
    from pairinteraction_gui.config.calculation_config import CalculationConfig
    from pairinteraction_gui.config.ket_config import KetConfig
    from pairinteraction_gui.config.system_config import SystemConfig
    from pairinteraction_gui.plotwidget.plotwidget import PlotWidget

logger = logging.getLogger(__name__)


class BasePage(WidgetV):
    """Base class for all pages in this application."""

    margin = (20, 20, 20, 20)
    spacing = 15

    title: str
    tooltip: str
    icon_path: Path | None = None

    def showEvent(self, event: QShowEvent) -> None:
        """Show event."""
        super().showEvent(event)
        self.window().setWindowTitle(
            f"PairInteraction v{pairinteraction.__version__} - " + self.title.replace("\n", " ")
        )


class SimulationPage(BasePage):
    """Base class for all simulation pages in this application."""

    ket_config: KetConfig

    plotwidget: PlotWidget

    def setupWidget(self) -> None:
        self.toolbox = QToolBox()

        # Create a dummy icon to allow adjusting the height of the toolbox tabs,
        # see https://stackoverflow.com/questions/48503645/customizing-qtoolbox-tab-height
        px = QPixmap(1, 1)
        px.fill(Qt.GlobalColor.transparent)
        self._toolbox_dummy_icon = QIcon(px)

    def postSetupWidget(self) -> None:
        for attr in self.__dict__.values():
            if isinstance(attr, BaseConfig):
                self.toolbox.addItem(attr, self._toolbox_dummy_icon, attr.title)

        for i, species_combo in enumerate(self.ket_config.species_combo_list):
            self.ket_config.signal_species_changed.emit(i, species_combo.currentText())

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.window().dockwidget.setWidget(self.toolbox)
        self.window().dockwidget.setVisible(True)
        self.toolbox.show()

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        self.window().dockwidget.setVisible(False)


class CalculationPage(SimulationPage):
    """Base class for all pages with a calculation button."""

    plotwidget: PlotEnergies
    # the following configs only exist on pages which support calculating in the selected plot limits
    system_config: SystemConfig
    calculation_config: CalculationConfig
    supports_calculate_in_limits = False
    _calculation_finished = False
    _plot_finished = False

    def setupWidget(self) -> None:
        super().setupWidget()

        # Plot Panel
        self.plotwidget = self._create_plot_widget()
        self.layout().addWidget(self.plotwidget)

        # Control panel below the plot
        bottom_layout = QHBoxLayout()
        bottom_layout.setObjectName("bottomLayout")

        # Calculate/Abort stacked buttons
        self.calculate_and_abort = NamedStackedWidget[QWidget](self)

        margin = (0, 1, 0, 1) if self.supports_calculate_in_limits else (0, 5, 0, 5)
        calculate_widget = WidgetV(self, name="CalculateButtons", margin=margin, spacing=2)
        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.calculate_button.setToolTip("Calculate with the current settings")
        self.calculate_button.clicked.connect(self.calculate_clicked)
        calculate_widget.layout().addWidget(self.calculate_button)

        if self.supports_calculate_in_limits:
            self.calculate_in_limits_button = QPushButton("Calculate in selected limits")
            self.calculate_in_limits_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
            self.calculate_in_limits_button.setToolTip(
                "Calculate again, but only inside the limits currently shown in the plot:\n"
                "the eigenenergies are calculated in the displayed energy range and all quantities which\n"
                "change along the x axis are restricted to the displayed x range, using the same number of steps."
            )
            self.calculate_in_limits_button.clicked.connect(self.calculate_in_limits_clicked)
            calculate_widget.layout().addWidget(self.calculate_in_limits_button)

        self.calculate_and_abort.addNamedWidget(calculate_widget, "Calculate")

        abort_button = QPushButton("Abort")
        abort_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserStop))
        abort_button.clicked.connect(self.abort_clicked)
        self.calculate_and_abort.addNamedWidget(abort_button, "Abort")

        self.calculate_and_abort.setFixedHeight(60 if self.supports_calculate_in_limits else 50)
        bottom_layout.addWidget(self.calculate_and_abort, stretch=2)

        # Create export button with menu
        export_button = QPushButton("Export")
        export_button.setObjectName("Export")
        export_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        export_menu = QMenu(self)
        for label, handler in self._get_export_actions():
            export_menu.addAction(label, handler)
        export_button.setMenu(export_menu)
        export_button.setFixedHeight(50)
        bottom_layout.addWidget(export_button, stretch=1)

        self.layout().addLayout(bottom_layout)

    def calculate_clicked(self) -> None:
        self._calculation_finished = False
        self._plot_finished = False
        self.before_calculate()

        def update_plot(
            parameters_and_results: tuple[Parameters[Any], Results],
        ) -> None:
            worker_plot = MultiThreadWorker(self.update_plot, *parameters_and_results)
            worker_plot.signals.progress.connect(lambda message: show_status_tip(self, message))
            worker_plot.signals.finished.connect(lambda _: setattr(self, "_plot_finished", True))
            worker_plot.start()

        worker = MultiThreadWorker(self.calculate)
        if hasattr(self, "calculation_config"):
            calculation_config: CalculationConfig = self.calculation_config
            number_of_steps = calculation_config.steps.value()
            worker.enable_busy_indicator(self.plotwidget, add_progress_label=True, number_of_steps=number_of_steps)
        else:
            worker.enable_busy_indicator(self.plotwidget)
        worker.signals.progress.connect(lambda message: show_status_tip(self, message))
        worker.signals.result.connect(update_plot)
        worker.signals.finished.connect(self.after_calculate)
        worker.signals.finished.connect(lambda _: setattr(self, "_calculation_finished", True))
        worker.start()

    def calculate_in_limits_clicked(self) -> None:
        """Restrict the settings to the limits currently shown in the plot and calculate again."""
        if not self.apply_plot_limits_to_config():
            return
        self.calculate_clicked()

    def apply_plot_limits_to_config(self) -> bool:
        """Restrict the calculation settings to the limits currently shown in the plot.

        The displayed energy range is used as diagonalization energy range and all quantities which change along
        the x axis are restricted to the displayed x range (the number of steps is left unchanged, i.e. all steps
        are now inside the selected range).

        Returns whether the limits could be applied.
        """
        parameters = self.plotwidget.parameters
        if parameters is None:
            show_status_tip(self, "Please calculate first, before calculating in the selected limits.", logger=logger)
            return False

        x_min, x_max = self.plotwidget.canvas.ax.get_xlim()
        y_min, y_max = self.plotwidget.canvas.ax.get_ylim()

        # the plotted energies are relative to the energy of interest, just like the diagonalization energy range
        self.calculation_config.energy_range.setValues(y_min, y_max)

        x_values = parameters.get_x_values()
        x_start, x_stop = x_values[0], x_values[-1]
        if x_start == x_stop:
            return True  # nothing changes along the x axis, so there are no ranges to restrict

        # all quantities which change along the x axis are restricted to the same relative part of their range
        range_items = self.system_config.get_range_items_dict()
        for key, values in parameters.ranges.items():
            if values[0] == values[-1]:
                continue
            new_values = [
                values[0] + (x - x_start) / (x_stop - x_start) * (values[-1] - values[0]) for x in (x_min, x_max)
            ]
            range_items[key].setValues(*new_values)

        return True

    def before_calculate(self) -> None:
        show_status_tip(self, "Calculating... Please wait.", logger=logger)
        self.calculate_and_abort.setCurrentNamedWidget("Abort")
        self.plotwidget.clear()

        self._start_time = time.perf_counter()

    def after_calculate(self, status: str) -> None:
        time_needed = time.perf_counter() - self._start_time
        show_status_tip(self, f"{status} after {time_needed:.2f} seconds.", logger=logger)
        self.calculate_and_abort.setCurrentNamedWidget("Calculate")

    def calculate(self) -> tuple[Parameters[Any], Results]:
        raise NotImplementedError("Subclasses must implement this method")

    def update_plot(self, parameters: Parameters[Any], results: Results) -> None:
        self.plotwidget.canvas.draw()  # draw once before, to avoid displaying artifacts during plotting
        self.plotwidget.plot(parameters, results)
        self.plotwidget.setup_annotations(parameters, results)
        self.plotwidget.canvas.draw()
        self.plotwidget.navigation_toolbar.reset_home_view()
        self.plotwidget.sync_range_widget()
        show_status_tip(self, "Finished updating plot. Tip: Click on the plot to see state information.", logger=logger)

    def export_png(self) -> None:
        """Export the current plot as a PNG file."""
        logger.debug("Exporting results as PNG...")

        filename, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG Files (*.png)")

        if filename:
            filename = filename.removesuffix(".png") + ".png"
            self.plotwidget.canvas.fig.savefig(filename, dpi=300, facecolor="white", edgecolor="none")
            logger.info("Plot saved as %s", filename)

    def _create_python_code(self) -> str:
        template_path = Path(__file__).parent.parent / "export_templates" / self._get_export_notebook_template_name()
        with Path(template_path).open() as f:
            notebook = nbformat.read(f, as_version=4)

        exporter = PythonExporter(exclude_output_prompt=True, exclude_input_prompt=True)
        content, _ = exporter.from_notebook_node(notebook)

        replacements = self._get_export_replacements()
        for key, value in replacements.items():
            content = content.replace(key, str(value))

        return content

    def export_python(self) -> None:
        """Export the current calculation as a Python script."""
        logger.debug("Exporting results as Python script...")
        filename, _ = QFileDialog.getSaveFileName(self, "Save Python Script", "", "Python Files (*.py)")
        if filename:
            filename = filename.removesuffix(".py") + ".py"
            content = self._create_python_code()
            with Path(filename).open("w") as f:
                f.write(content)
            logger.info("Python script saved as %s", filename)

    def export_notebook(self) -> None:
        """Export the current calculation as a Jupyter notebook."""
        logger.debug("Exporting results as Jupyter notebook...")

        filename, _ = QFileDialog.getSaveFileName(self, "Save Jupyter Notebook", "", "Jupyter Notebooks (*.ipynb)")

        if filename:
            filename = filename.removesuffix(".ipynb") + ".ipynb"

            template_path = (
                Path(__file__).parent.parent / "export_templates" / self._get_export_notebook_template_name()
            )
            with Path(template_path).open() as f:
                notebook = nbformat.read(f, as_version=4)

            replacements = self._get_export_replacements()
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    source = cell.source
                    for key, value in replacements.items():
                        source = source.replace(key, str(value))
                    cell.source = source

            nbformat.write(notebook, filename)

            logger.info("Jupyter notebook saved as %s", filename)

    def _get_export_notebook_template_name(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    def _get_export_replacements(self) -> dict[str, str]:
        # Override this method in subclasses to provide specific replacements for the export
        return {}

    def abort_clicked(self) -> None:
        """Handle abort button click."""
        logger.debug("Aborting calculation.")
        MultiThreadWorker.terminate_all()
        self.after_calculate("Calculation aborted.")

    def _create_plot_widget(self) -> PlotEnergies:
        return PlotEnergies(self)

    def _get_export_actions(self) -> list[tuple[str, Callable[[], None]]]:
        return [
            ("Export as PNG", self.export_png),
            ("Export as Python script", self.export_python),
            ("Export as Jupyter notebook", self.export_notebook),
        ]
