"""TFoam's diagnostics use its cell estimates without drawing any extra random numbers."""

from __future__ import annotations

from typing import Any

from .. import cout, printing

#: The box and padding used by ROOT's TFoam 1.02M diagnostics.
BORDER = "F" * 80
SPACE = "F" + " " * 78 + "F"


def _text(text: str) -> str:
    return f"F                   {text:>40}                   F\n"


def _integer(name: str, value: int, description: str) -> str:
    return f"F {name:>10} = {value:>10} = {description:>50} F\n"


def _real(name: str, value: Any, description: str) -> str:
    shown = value if isinstance(value, str) else f"{value:.8g}"
    return f"F {name:>10} = {shown:>15}   =    {description:>40} F\n"


def _open() -> str:
    return BORDER + "\n" + SPACE + "\n"


def _close() -> str:
    return SPACE + "\n" + BORDER + "\n"


def header(foam: Any) -> None:
    printing.PRECISION[0] = 8
    text = _open()
    for title in (
        "****************************************",
        "******      TFoam::Initialize    ******",
        "****************************************",
        "TFOAM",
    ):
        text += _text(title)
    text += _real("  Version", "1.02M", "  Release date:  2005.04.10")
    for name, value, description in (
        ("     kDim", foam.dim, " Dimension of the hyper-cubical space             "),
        ("   nCells", foam.n_cells, " Requested number of Cells (half of them active)  "),
        ("   nSampl", foam.n_sampl, " No of MC events in exploration of a cell         "),
        ("     nBin", foam.n_bin, " No of bins in histograms, MC exploration of cell "),
        (" EvPerBin", foam.ev_per_bin, " Maximum No effective_events/bin, MC exploration  "),
        (" OptDrive", foam.opt_drive, " Type of Driver   =1,2 for Sigma,WtMax            "),
        ("   OptRej", foam.opt_rej, " MC rejection on/off for OptRej=0,1               "),
    ):
        text += _integer(name, value, description)
    text += _real(" MaxWtRej", foam.max_wt_rej, " Maximum wt in rejection for wt=1 evts")
    cout.write(text + _close())


def progress(foam: Any) -> None:
    last = len(foam.cells) - 1
    every = 100 if last >= 10000 else 10
    if last and last % every == 0:
        cout.write(str(foam.dim) if foam.dim < 10 else ".")
        if last % (100 * every) == 0:
            cout.line(f"|{last}")


def summary(foam: Any) -> None:
    text = _open() + _text("***  TFoam::Initialize FINISHED!!!  ***")
    text += _integer("    nCalls", foam.n_calls, "Total number of function calls         ")
    text += _real("    XPrime", foam.prime, "Primary total integral                 ")
    text += _real("    XDiver", foam.cells[0].driv, "Driver  total integral                 ")
    text += _real("  mcResult", foam.cells[0].intg, "Estimate of the true MC Integral       ")
    cout.write(text + _close())


def _serial(cell: Any) -> int:
    return cell.serial if cell is not None else -1


def _vector(values: Any) -> str:
    return "(" + ",".join(f"{value:12.7g}" for value in values) + ")"


def cells(foam: Any) -> None:
    for cell in foam.cells:
        text = f"Cell[{cell.serial}]={{ \n"
        text += f" Status= {int(cell.active)}, Volume= {cell.volume:.8g},"
        text += f" TrueInteg= {cell.intg:.8g}, DriveInteg= {cell.driv:.8g},"
        text += f" PrimInteg= {cell.prim:.8g},\n"
        text += f" Xdiv= {cell.xdiv:.8g}, Best= {cell.best},"
        text += f" Parent=  {{{_serial(cell.parent)}}} "
        text += f" Daught0= {{{_serial(cell.dau0)}}}  Daught1= {{{_serial(cell.dau1)}}} \n"
        text += f"   Posi= {_vector(cell.posi)},\n   Size= {_vector(cell.size)},\n}}\n"
        cout.write(text)
