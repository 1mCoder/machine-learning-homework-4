from src.code.make_prediction import make_prediction
import click

@click.command(help="")
@click.option("--model", default="logreg_f1", type=str, help="model name ['baseline' | 'logreg_acc' | 'logreg_f1']")
@click.option("--output-folder", default="submissions", type=str, help="output folder")
@click.option("--text", default="Hello World", type=str, help="input text")
def main(model, output_folder, text):
    make_prediction(model, output_folder, text)

if __name__ == "__main__":
    main()