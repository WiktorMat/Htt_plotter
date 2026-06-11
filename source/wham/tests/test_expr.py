from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from wham.expr import Expr, ExprError, parse


@pytest.fixture
def table() -> pa.Table:
    rng = np.random.default_rng(7)
    n = 5000
    return pa.table(
        {
            "pt_1": rng.uniform(0, 100, n),
            "eta_1": rng.uniform(-3, 3, n),
            "iso_1": rng.uniform(0, 0.5, n),
            "id_2": rng.integers(0, 8, n).astype(np.int32),
            "os": rng.integers(0, 2, n).astype(np.int32),
            "weight": rng.uniform(0.5, 1.5, n),
        }
    )


CUTS = [
    "pt_1 > 26",
    "abs(eta_1) < 2.4",
    "iso_1 < 0.15 & pt_1 > 30",
    "(id_2 >= 5) | (iso_1 < 0.1)",
    "~(os == 1)",
    "1 < id_2 < 5",
    "abs(eta_1) < 2.4 & (1 < id_2 < 5) & pt_1 > 26",
    "pt_1 / 2 > 20",
    "pt_1 + iso_1 * 10 > 30",
    "-eta_1 > 0",
    "id_2 != 3",
    "pt_1 > 26 and iso_1 < 0.15",
    "id_2 >= 5 or iso_1 < 0.1",
    "not (os == 1)",
    "pt_1 > 26 & iso_1 < 0.15 | id_2 == 7",
]


@pytest.mark.parametrize("source", CUTS)
def test_arrow_matches_numpy(table: pa.Table, source: str) -> None:
    parsed = parse(source)

    cols = {name: table.column(name).to_numpy() for name in parsed.columns}
    np_mask = parsed.evaluate(cols)
    assert np_mask.dtype == bool

    dataset = ds.dataset(table)
    arrow_rows = dataset.to_table(filter=parsed.arrow()).num_rows
    assert arrow_rows == int(np_mask.sum()), source


def test_columns_extracted() -> None:
    parsed = parse("abs(eta_1) < 2.4 & pt_1 > 26 & (1 < id_2 < 5)")
    assert parsed.columns == {"eta_1", "pt_1", "id_2"}


def test_weight_expression(table: pa.Table) -> None:
    parsed = parse("weight * 2 + 1")
    cols = {"weight": table.column("weight").to_numpy()}
    out = parsed.evaluate(cols)
    assert np.allclose(out, cols["weight"] * 2 + 1)


def test_amp_precedence_matches_intent(table: pa.Table) -> None:
    # Without preprocessing, python would parse this as iso_1 < (0.15 & pt_1) > 30.
    a = parse("iso_1 < 0.15 & pt_1 > 30")
    b = parse("(iso_1 < 0.15) & (pt_1 > 30)")
    cols = {n: table.column(n).to_numpy() for n in ("iso_1", "pt_1")}
    assert np.array_equal(a.evaluate(cols), b.evaluate(cols))
    assert a.evaluate(cols).sum() > 0


@pytest.mark.parametrize(
    "source, match",
    [
        ("sin(eta_1) < 1", "only abs"),
        ("pt_1.size > 3", "unsupported syntax"),
        ("__import__('os')", "only abs"),
        ("pt_1 > 'x'", "numeric literals"),
        ("[1,2]", "unsupported syntax"),
        ("", "empty"),
        ("pt_1 >", "syntax error"),
    ],
)
def test_rejections(source: str, match: str) -> None:
    with pytest.raises(ExprError, match=match):
        parse(source)


def test_missing_column_error() -> None:
    parsed = parse("nope > 1")
    with pytest.raises(ExprError, match="missing columns"):
        parsed.evaluate({"pt_1": np.ones(3)})


def test_parse_cached() -> None:
    assert parse("pt_1 > 26") is parse("pt_1 > 26")


def test_expr_is_picklable_via_source() -> None:
    # Workers re-parse from source strings; Expr itself need not pickle.
    parsed = parse("pt_1 > 26")
    assert isinstance(parsed, Expr)
    assert parse(parsed.source).columns == parsed.columns
