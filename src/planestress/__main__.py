"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """planestress."""


if __name__ == "__main__":
    main(prog_name="planestress")  # pragma: no cover
