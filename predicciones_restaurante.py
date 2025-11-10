"""Script para generar pronósticos de tickets y ventas diarias para el Restaurante Kuche.

Este módulo lee una base de datos en Excel con los campos:
- Fecha (dd-mm-aaaa)
- Día (1=lunes, 7=domingo)
- Tickets
- Venta diaria

Genera modelos predictivos basados en la fecha y el día de la semana, evalúa la precisión
utilizando un conjunto de validación, produce gráficos con las series históricas y las
proyecciones y finalmente exporta un archivo Excel con los resultados.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResults:
    """Resultados del ajuste de un modelo."""

    pipeline: Pipeline
    mae: float
    rmse: float
    r2: float


DATE_FREQ = 365.25
DAY_FREQ = 7


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera pronósticos de tickets y ventas diarias para el Restaurante Kuche."
    )
    parser.add_argument(
        "excel_path",
        type=Path,
        help="Ruta al archivo 'BD Tickets para predicciones.xlsx'.",
    )
    parser.add_argument(
        "--forecast_days",
        type=int,
        default=15,
        help="Número de días a pronosticar a partir del último registro (default: 15).",
    )
    return parser.parse_args()


def load_data(excel_path: Path) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {excel_path}")

    df = pd.read_excel(excel_path)

    column_mapping: Dict[str, str] = {
        "Ticekts": "Tickets",
        "tickets": "Tickets",
        "venta diaria": "Venta diaria",
        "Venta Diaria": "Venta diaria",
        "venta_diaria": "Venta diaria",
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    expected_columns = {"Fecha", "Día", "Tickets", "Venta diaria"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(
            "El archivo no contiene las columnas esperadas: "
            + ", ".join(sorted(expected_columns))
        )

    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    if df["Fecha"].isna().any():
        raise ValueError("Existen fechas no válidas en el archivo de entrada.")

    df["Día"] = df["Día"].astype(int)
    df.sort_values("Fecha", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    ordinal = df["Fecha"].map(datetime.toordinal)
    features["fecha_ordinal"] = ordinal
    features["fecha_sin"] = np.sin(ordinal * 2 * np.pi / DATE_FREQ)
    features["fecha_cos"] = np.cos(ordinal * 2 * np.pi / DATE_FREQ)

    day = df["Día"].astype(float)
    features["dia"] = day
    features["dia_sin"] = np.sin(day * 2 * np.pi / DAY_FREQ)
    features["dia_cos"] = np.cos(day * 2 * np.pi / DAY_FREQ)
    features["es_lunes"] = (df["Día"] == 1).astype(int)
    return features


def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1")

    split_index = int(len(df) * (1 - test_size))
    split_index = max(split_index, 1)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> ModelResults:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingRegressor(random_state=42),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    return ModelResults(pipeline=pipeline, mae=mae, rmse=rmse, r2=r2)


def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y_true: pd.Series) -> pd.Series:
    predictions = pipeline.predict(X)
    return pd.Series(predictions, index=y_true.index)


def make_future_dataframe(last_date: pd.Timestamp, days: int) -> pd.DataFrame:
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_df = pd.DataFrame({"Fecha": future_dates})
    future_df["Día"] = [d.isoweekday() for d in future_dates]
    return future_df


def generate_plots(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    output_dir: Path,
    timestamp: str,
) -> Tuple[Path, Path]:
    plt.style.use("seaborn-v0_8")

    tickets_path = output_dir / f"pronostico_tickets_{timestamp}.png"
    ventas_path = output_dir / f"pronostico_ventas_{timestamp}.png"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(historical_df["Fecha"], historical_df["Tickets"], label="Tickets históricos", color="C0")
    ax.plot(
        forecast_df["Fecha"],
        forecast_df["Tickets pronosticados"],
        label="Tickets pronosticados",
        color="C1",
        marker="o",
    )
    ax.set_title("Evolución diaria de tickets")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Tickets")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(tickets_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(historical_df["Fecha"], historical_df["Venta diaria"], label="Ventas históricas", color="C0")
    ax.plot(
        forecast_df["Fecha"],
        forecast_df["Venta diaria pronosticada"],
        label="Ventas pronosticadas",
        color="C2",
        marker="o",
    )
    ax.set_title("Evolución diaria de ventas")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Venta diaria")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ventas_path, dpi=150)
    plt.close(fig)

    return tickets_path, ventas_path


def export_results(
    base_dir: Path,
    timestamp: str,
    historical_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> Path:
    output_name = f"Predicciones Restaurante Kuche {timestamp}.xlsx"
    output_path = base_dir / output_name

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        historical_df.to_excel(writer, sheet_name="Historico", index=False)
        validation_df.to_excel(writer, sheet_name="Validacion", index=False)
        forecast_df.to_excel(writer, sheet_name="Pronosticos", index=False)
        metrics_df.to_excel(writer, sheet_name="Metricas", index=False)

    return output_path


def main() -> None:
    args = parse_arguments()
    excel_path = args.excel_path.expanduser()
    df = load_data(excel_path)

    train_df, test_df = temporal_train_test_split(df, test_size=0.2)

    X_train = build_features(train_df)
    X_test = build_features(test_df)

    tickets_results = train_model(X_train, train_df["Tickets"], X_test, test_df["Tickets"])
    ventas_results = train_model(X_train, train_df["Venta diaria"], X_test, test_df["Venta diaria"])

    test_df = test_df.copy()
    test_df["Tickets pronosticados"] = evaluate_model(
        tickets_results.pipeline, X_test, test_df["Tickets"]
    )
    test_df["Venta diaria pronosticada"] = evaluate_model(
        ventas_results.pipeline, X_test, test_df["Venta diaria"]
    )

    last_date = df["Fecha"].max()
    future_df = make_future_dataframe(last_date, args.forecast_days)
    future_features = build_features(future_df)
    future_df["Tickets pronosticados"] = tickets_results.pipeline.predict(future_features)
    future_df["Venta diaria pronosticada"] = ventas_results.pipeline.predict(future_features)

    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    base_dir = excel_path.parent if excel_path.parent != Path("") else Path.cwd()

    metrics_df = pd.DataFrame(
        [
            {
                "Variable": "Tickets",
                "MAE": tickets_results.mae,
                "RMSE": tickets_results.rmse,
                "R^2": tickets_results.r2,
            },
            {
                "Variable": "Venta diaria",
                "MAE": ventas_results.mae,
                "RMSE": ventas_results.rmse,
                "R^2": ventas_results.r2,
            },
        ]
    )

    validation_columns = [
        "Fecha",
        "Día",
        "Tickets",
        "Tickets pronosticados",
        "Venta diaria",
        "Venta diaria pronosticada",
    ]
    validation_df = test_df[validation_columns]

    tickets_plot, ventas_plot = generate_plots(df, future_df, base_dir, timestamp)

    historical_columns = ["Fecha", "Día", "Tickets", "Venta diaria"]
    historical_df = df[historical_columns]

    forecast_columns = [
        "Fecha",
        "Día",
        "Tickets pronosticados",
        "Venta diaria pronosticada",
    ]
    forecast_df = future_df[forecast_columns]

    output_excel = export_results(
        base_dir=base_dir,
        timestamp=timestamp,
        historical_df=historical_df,
        validation_df=validation_df,
        forecast_df=forecast_df,
        metrics_df=metrics_df,
    )

    print("Archivo de resultados generado:", output_excel)
    print("Gráfico de tickets guardado en:", tickets_plot)
    print("Gráfico de ventas guardado en:", ventas_plot)
    print("Métricas de validación:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
