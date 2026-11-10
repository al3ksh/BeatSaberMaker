"""Command line interface."""
import typer

app = typer.Typer(help="BeatSaberMaker: Beat Saber map generator.", no_args_is_help=True)
data_app = typer.Typer(help="Dataset commands.", no_args_is_help=True)
app.add_typer(data_app, name="data")
