"""Main entry point for pygui application."""


def main():
    """Launch the pygui dashboard."""
    from pygui.viz.dashboard import launch_dashboard

    dashboard = launch_dashboard()

    print("=" * 60)
    print("pygui - Time Series Analysis Dashboard")
    print("=" * 60)
    print("Dashboard is running...")
    print("Open your browser to view the application")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    return dashboard


if __name__ == "__main__":
    main()
