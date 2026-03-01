"""Panel dashboard for interactive time series analysis."""

from pathlib import Path

import hvplot.pandas  # type: ignore
import pandas as pd
import panel as pn

from pygui.analysis.statistics import (
    basic_stats,
    correlation_matrix,
)
from pygui.data.loader import load_parquet
from pygui.data.transformer import (
    detect_outliers,
    filter_by_time_range,
    impute_missing,
    resample,
    rolling_window,
)

pn.extension("tabulator")


class TimeSeriesDashboard:
    """Interactive dashboard for time series analysis."""

    def __init__(self):
        """Initialize the dashboard."""
        self.df: pd.DataFrame | None = None
        self.original_df: pd.DataFrame | None = None
        self.filepath: str | None = None

        self._setup_widgets()
        self._setup_layout()

    def _setup_widgets(self):
        """Setup all UI widgets."""
        self.file_input = pn.widgets.FileInput(accept=".parquet")
        self.file_input.param.watch(self._load_data, "value")

        self.start_time = pn.widgets.DatetimeInput(name="Start Time")
        self.end_time = pn.widgets.DatetimeInput(name="End Time")
        self.apply_filter_btn = pn.widgets.Button(name="Apply Time Filter", button_type="primary")
        self.apply_filter_btn.on_click(self._apply_filter)

        self.resample_freq = pn.widgets.Select(
            name="Resample Frequency", options=["1S", "1T", "5T", "15T", "1H", "1D"], value="1H"
        )
        self.resample_agg = pn.widgets.Select(
            name="Aggregation", options=["mean", "sum", "std", "min", "max"], value="mean"
        )
        self.apply_resample_btn = pn.widgets.Button(name="Apply Resample", button_type="primary")
        self.apply_resample_btn.on_click(self._apply_resample)

        self.rolling_window_size = pn.widgets.IntInput(name="Window Size", value=10, start=1)
        self.rolling_agg = pn.widgets.Select(
            name="Aggregation", options=["mean", "sum", "std", "min", "max"], value="mean"
        )
        self.apply_rolling_btn = pn.widgets.Button(name="Apply Rolling", button_type="primary")
        self.apply_rolling_btn.on_click(self._apply_rolling)

        self.outlier_method = pn.widgets.Select(
            name="Method", options=["zscore", "iqr"], value="zscore"
        )
        self.outlier_threshold = pn.widgets.FloatInput(name="Threshold", value=3.0, start=0.1)
        self.apply_outlier_btn = pn.widgets.Button(name="Detect Outliers", button_type="warning")
        self.apply_outlier_btn.on_click(self._apply_outlier_detection)

        self.impute_method = pn.widgets.Select(
            name="Method", options=["ffill", "bfill", "interpolate"], value="ffill"
        )
        self.apply_impute_btn = pn.widgets.Button(name="Impute Missing", button_type="success")
        self.apply_impute_btn.on_click(self._apply_impute)

        self.plot_columns = pn.widgets.MultiSelect(name="Columns to Plot", options=[], value=[])

        self.reset_btn = pn.widgets.Button(name="Reset to Original", button_type="danger")
        self.reset_btn.on_click(self._reset_data)

        self.export_plot_btn = pn.widgets.Button(name="Export Plot", button_type="primary")
        self.export_plot_btn.on_click(self._export_plot)

        self.export_data_btn = pn.widgets.Button(name="Export Data", button_type="primary")
        self.export_data_btn.on_click(self._export_data)

    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.time_series_plot = pn.pane.HoloViews(sizing_mode="stretch_width", height=400)
        self.histogram_plot = pn.pane.HoloViews(sizing_mode="stretch_width", height=300)
        self.stats_display = pn.pane.DataFrame(sizing_mode="stretch_width", height=400)
        self.correlation_display = pn.pane.DataFrame(sizing_mode="stretch_width", height=300)
        self.outlier_display = pn.pane.Markdown()

        self.control_panel = pn.Column(
            pn.pane.Markdown("### Data Loading"),
            self.file_input,
            pn.pane.Markdown("---"),
            pn.pane.Markdown("### Time Filter"),
            pn.Row(self.start_time, self.end_time),
            self.apply_filter_btn,
            pn.pane.Markdown("---"),
            pn.pane.Markdown("### Resampling"),
            pn.Row(self.resample_freq, self.resample_agg),
            self.apply_resample_btn,
            pn.pane.Markdown("---"),
            pn.pane.Markdown("### Rolling Window"),
            pn.Row(self.rolling_window_size, self.rolling_agg),
            self.apply_rolling_btn,
            pn.pane.Markdown("---"),
            pn.pane.Markdown("### Outlier Detection"),
            pn.Row(self.outlier_method, self.outlier_threshold),
            self.apply_outlier_btn,
            pn.pane.Markdown("---"),
            pn.pane.Markdown("### Missing Values"),
            self.impute_method,
            self.apply_impute_btn,
            pn.pane.Markdown("---"),
            pn.pane.Markdown("### Plot Columns"),
            self.plot_columns,
            pn.pane.Markdown("---"),
            pn.pane.Markdown("### Actions"),
            pn.Row(self.reset_btn, self.export_plot_btn, self.export_data_btn),
            sizing_mode="fixed",
            width=350,
        )

        self.plot_panel = pn.Column(
            pn.pane.Markdown("### Time Series"),
            self.time_series_plot,
            pn.pane.Markdown("### Distribution"),
            self.histogram_plot,
            sizing_mode="stretch_width",
        )

        self.stats_panel = pn.Column(
            pn.pane.Markdown("### Basic Statistics"),
            self.stats_display,
            pn.pane.Markdown("---"),
            pn.pane.Markdown("### Correlation Matrix"),
            self.correlation_display,
            pn.pane.Markdown("---"),
            pn.pane.Markdown("### Outlier Summary"),
            self.outlier_display,
            sizing_mode="fixed",
            width=400,
        )

        self.main_layout = pn.Row(
            self.control_panel,
            self.plot_panel,
            self.stats_panel,
            sizing_mode="stretch_width",
        )

    def _load_data(self, event):
        """Load data from uploaded Parquet file."""
        if event.new is None:
            return

        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
            tmp.write(event.new)
            tmp_path = tmp.name

        try:
            self.df = load_parquet(tmp_path)
            self.original_df = self.df.copy()
            self.filepath = tmp_path

            numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
            self.plot_columns.options = numeric_cols
            self.plot_columns.value = numeric_cols[:1] if numeric_cols else []

            self.start_time.value = self.df.index.min()
            self.end_time.value = self.df.index.max()

            self._update_plots()
            self._update_stats()
        except Exception as e:
            if pn.state.notifications:
                pn.state.notifications.error(f"Error loading data: {str(e)}")  # type: ignore

    def _apply_filter(self, event):
        """Apply time range filter."""
        if self.df is None:
            return

        try:
            start_val = self.start_time.value
            end_val = self.end_time.value
            if start_val is not None and end_val is not None:
                self.df = filter_by_time_range(self.df, str(start_val), str(end_val))
                self._update_plots()
                self._update_stats()
        except Exception as e:
            if pn.state.notifications:
                pn.state.notifications.error(f"Error applying filter: {str(e)}")  # type: ignore

    def _apply_resample(self, event):
        """Apply resampling."""
        if self.df is None:
            return

        try:
            freq_val = self.resample_freq.value
            agg_val = self.resample_agg.value
            if freq_val is not None and agg_val is not None:
                self.df = resample(self.df, freq_val, agg_val)
                self._update_plots()
                self._update_stats()
        except Exception as e:
            if pn.state.notifications:
                pn.state.notifications.error(f"Error applying resample: {str(e)}")  # type: ignore

    def _apply_rolling(self, event):
        """Apply rolling window."""
        if self.df is None:
            return

        try:
            window_val = self.rolling_window_size.value
            agg_val = self.rolling_agg.value
            if window_val is not None and agg_val is not None:
                self.df = rolling_window(self.df, window=window_val, agg_func=agg_val)  # type: ignore
                self._update_plots()
                self._update_stats()
        except Exception as e:
            if pn.state.notifications:
                pn.state.notifications.error(f"Error applying rolling: {str(e)}")  # type: ignore

    def _apply_outlier_detection(self, event):
        """Detect and mark outliers."""
        if self.df is None:
            return

        try:
            method_val = self.outlier_method.value
            threshold_val = self.outlier_threshold.value
            if method_val is not None and threshold_val is not None:
                outliers = detect_outliers(
                    self.df,
                    method=method_val,
                    threshold=float(threshold_val),  # type: ignore
                )

                outlier_counts = outliers.sum()
                total_rows = len(outliers)

                summary = "### Outlier Detection Results\n\n"
                for col, count in outlier_counts.items():
                    if count > 0:
                        percentage = (count / total_rows) * 100
                        summary += f"- **{col}**: {count} outliers ({percentage:.2f}%)\n"
                    else:
                        summary += f"- **{col}**: No outliers detected\n"

                self.outlier_display.object = summary
                if pn.state.notifications:
                    pn.state.notifications.success("Outlier detection complete")  # type: ignore
        except Exception as e:
            if pn.state.notifications:
                pn.state.notifications.error(f"Error detecting outliers: {str(e)}")  # type: ignore

    def _apply_impute(self, event):
        """Apply missing value imputation."""
        if self.df is None:
            return

        try:
            method_val = self.impute_method.value
            if method_val is not None:
                self.df = impute_missing(self.df, method=method_val)
                self._update_plots()
                self._update_stats()
                if pn.state.notifications:
                    pn.state.notifications.success("Missing values imputed")  # type: ignore
        except Exception as e:
            if pn.state.notifications:
                pn.state.notifications.error(f"Error imputing values: {str(e)}")  # type: ignore

    def _reset_data(self, event):
        """Reset data to original state."""
        if self.original_df is None:
            return

        self.df = self.original_df.copy()
        self._update_plots()
        self._update_stats()
        if pn.state.notifications:
            pn.state.notifications.success("Data reset to original")  # type: ignore

    def _update_plots(self):
        """Update time series and histogram plots."""
        if self.df is None or not self.plot_columns.value:  # type: ignore
            return

        plot_cols_val = self.plot_columns.value
        if not plot_cols_val:  # type: ignore
            return

        columns = plot_cols_val[:3] if plot_cols_val else []  # type: ignore

        if not columns:
            return

        df_to_plot = self.df[columns].iloc[-10000:] if len(self.df) > 10000 else self.df[columns]
        self.time_series_plot.object = hvplot.plot(
            df_to_plot,
            legend="top_left",
            responsive=True,
            tools=["pan", "zoom", "box_zoom", "reset"],
        )  # type: ignore

        for col in columns[:1]:
            self.histogram_plot.object = hvplot.plot(
                self.df[col].dropna(),
                kind="hist",  # type: ignore
                bins=50,
                title=f"Distribution: {col}",
                responsive=True,
            )

    def _update_stats(self):
        """Update statistics displays."""
        if self.df is None:
            return

        try:
            self.stats_display.object = basic_stats(self.df)
            self.correlation_display.object = correlation_matrix(self.df)
        except Exception as e:
            if pn.state.notifications:
                pn.state.notifications.error(f"Error computing stats: {str(e)}")  # type: ignore

    def _export_plot(self, event):
        """Export current plot as image."""
        if pn.state.notifications:
            pn.state.notifications.info("Plot export - use browser save or screenshot")  # type: ignore

    def _export_data(self, event):
        """Export current data as Parquet."""
        if self.df is None:
            return

        export_path = Path("data/exported_data.parquet")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(export_path)
        if pn.state.notifications:
            pn.state.notifications.success(f"Data exported to {export_path}")  # type: ignore

    def serve(self):
        """Serve the dashboard."""
        return self.main_layout.servable()


def launch_dashboard():
    """Launch the time series dashboard."""
    dashboard = TimeSeriesDashboard()
    return dashboard.serve()
