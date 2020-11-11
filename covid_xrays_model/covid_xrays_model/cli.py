import click
from covid_xrays_model.train_pipeline import run_training_sample, improve_saved_model,\
                                             plot_learning_rate


@click.group()
def main():
    pass

@main.command()
@click.option('--sample_size', default=600, help='Number of images to use')
@click.option('--image_size', default=224, help='Image size in pixels')
@click.option('--n_cycles', default=10, help='Number of cycles (epochs) trough the whole data')
@click.option('--focal_loss', is_flag=True, default=False, help='Use focal_loss function')
@click.option('--oversampling', is_flag=True, default=False, help='Use oversampling')
@click.option('--weighted_loss', is_flag=True, default=False, help='Use weighted loss function')
def run_training(sample_size, image_size, n_cycles, focal_loss, oversampling, weighted_loss):

    click.echo("Running model trainning...")
    run_training_sample(sample_size=sample_size, image_size=image_size, n_cycles=n_cycles,
                        with_focal_loss=focal_loss, with_oversampling=oversampling,
                        with_weighted_loss=weighted_loss)


@main.command()
@click.option('--sample_size', default=600, help='Number of images to use')
@click.option('--image_size', default=224, help='Image size in pixels')
def get_learning_rate(sample_size, image_size):

    click.echo("Generating learing rate plot...")
    plot_learning_rate(sample_size=sample_size, image_size=image_size)


@main.command()
@click.option('--sample_size', default=600, help='Number of images to use')
@click.option('--image_size', default=224, help='Image size in pixels')
@click.option('--n_cycles', default=2, help='Number of cycles (epochs) trough the whole data')
@click.option('--max_lr', default=slice(1e-6,1e-4), help='slice for learning rate. e.g., slice(1e-6,1e-4)')
@click.option('--save', is_flag=True, default=False, help='Save the new trained model, replacing the old one')
@click.option('--focal_loss', is_flag=True, default=False, help='Use focal_loss function')
@click.option('--oversampling', is_flag=True, default=False, help='Use oversampling')
@click.option('--weighted_loss', is_flag=True, default=False, help='Use weighted loss function')
def improve_model(sample_size, image_size, n_cycles, max_lr, save, focal_loss, oversampling, weighted_loss):

    click.echo("Running improve model...")
    improve_saved_model(sample_size=sample_size, image_size=image_size, n_cycles=n_cycles,
                        max_lr=max_lr, save=save, with_focal_loss=focal_loss,
                        with_oversampling=oversampling,
                        with_weighted_loss=weighted_loss)


if __name__ == "__main__":
    main()