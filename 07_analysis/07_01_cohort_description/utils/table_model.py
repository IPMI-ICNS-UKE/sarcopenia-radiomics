from dataclasses import dataclass, field
from typing import List

# Row kinds drive the visual style applied by the exporter.
KIND_DATA = "data"  # ordinary value row
KIND_SECTION = "section"  # bold shaded section header (e.g. "Functional tests")
KIND_SUB = "sub"  # indented sub-row (e.g. "Male", "Female")
KIND_COHORT = "cohort"  # the wrapped cohort-description row
KIND_CONT = "cont"  # continuation row (the "(min - max)" line)


@dataclass
class Row:
    cells: List[str]
    kind: str = KIND_DATA


@dataclass
class TableModel:
    title: str
    header: List[str]  # the styled column-header row
    rows: List[Row] = field(default_factory=list)
    sheet_name: str = "Sheet1"

    @property
    def ncols(self) -> int:
        return len(self.header)

    def add(self, cells, kind: str = KIND_DATA) -> None:
        cells = list(cells)
        # pad / trim to the header width so the grid is always rectangular
        if len(cells) < self.ncols:
            cells = cells + [""] * (self.ncols - len(cells))
        elif len(cells) > self.ncols:
            cells = cells[: self.ncols]
        self.rows.append(Row(cells, kind))
