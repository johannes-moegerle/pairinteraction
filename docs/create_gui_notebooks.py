# SPDX-FileCopyrightText: 2025 PairInteraction Developers
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate GUI example notebooks for the documentation.

Run via: make gui-notebooks  (from docs/ directory)
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
from pathlib import Path

import nbformat

from pairinteraction.cli import download_databases
from pairinteraction_gui.app import Application
from pairinteraction_gui.main_window import MainWindow

if TYPE_CHECKING:
    from pairinteraction_gui.page.base_page import CalculationPage
    from pairinteraction_gui.page.one_atom_page import OneAtomPage
    from pairinteraction_gui.page.two_atoms_page import TwoAtomsPage


TEMPLATE_DIR = Path(__file__).parent.parent / "src" / "pairinteraction_gui" / "export_templates"
OUTPUT_DIR = Path(__file__).parent / "tutorials" / "examples_gui_notebooks"


def _export_notebook(page: CalculationPage, output_path: Path) -> None:
    template_name = page._get_export_notebook_template_name()
    with (TEMPLATE_DIR / template_name).open() as f:
        notebook = nbformat.read(f, as_version=4)
    replacements = page._get_export_replacements()
    for cell in notebook.cells:
        if cell.cell_type == "code":
            for key, value in replacements.items():
                cell.source = cell.source.replace(key, str(value))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)
    print(f"Created {output_path}")


def _setup_stark_map(window: MainWindow) -> OneAtomPage:
    page: OneAtomPage = window.stacked_pages.getNamedWidget("OneAtomPage")  # type: ignore [assignment]
    page.ket_config.species_combo_list[0].setCurrentText("Rb")
    ket_qn = page.ket_config.stacked_qn_list[0].currentWidget()
    ket_qn.items["n"].setValue(60)
    ket_qn.items["l"].setValue(0)
    ket_qn.items["m"].setValue(0.5)
    basis_qn = page.basis_config.stacked_basis_list[0].currentWidget()
    basis_qn.items["n"].setValue(2)
    basis_qn.items["l"].setValue(2)
    basis_qn.items["m"].setChecked(False)
    page.calculation_config.steps.setValue(50)
    page.system_config.Ez.spinboxes[1].setValue(10)
    return page


def _setup_pair_potential(window: MainWindow) -> TwoAtomsPage:
    page: TwoAtomsPage = window.stacked_pages.getNamedWidget("TwoAtomsPage")  # type: ignore [assignment]
    page.ket_config.species_combo_list[0].setCurrentText("Rb")
    for stacked in page.ket_config.stacked_qn_list:
        qn = stacked.currentWidget()
        qn.items["n"].setValue(60)
        qn.items["l"].setValue(0)
        qn.items["m"].setValue(0.5)
    for stacked in page.basis_config.stacked_basis_list:
        qn = stacked.currentWidget()
        qn.items["n"].setValue(2)
        qn.items["l"].setValue(2)
        qn.items["m"].setChecked(False)
    page.basis_config.pair_delta_energy.setValue(3)
    page.basis_config.pair_m_range.setValues(1, 1)
    page.calculation_config.steps.setValue(50)
    page.system_config.distance.setValues(1, 10)
    return page


def main() -> None:
    download_databases(["Rb"])

    app = Application.instance() or Application([])
    with tempfile.TemporaryDirectory() as tmp:
        window = MainWindow(cache_dir=Path(tmp))
        window.show()

        _export_notebook(_setup_stark_map(window), OUTPUT_DIR / "stark_map.ipynb")
        _export_notebook(_setup_pair_potential(window), OUTPUT_DIR / "pair_potential.ipynb")

    app.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
