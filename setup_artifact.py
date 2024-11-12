import wandb

wandb.init(project="autoadvex", entity="cavaokcava")

# Create a new artifact
artifact = wandb.Artifact('adversarial_results', type='results')

# Create the table
table = wandb.Table(columns=["confused_classes", "includes_true", "example_num", "notes", "image"])

# Add the table to the artifact
artifact.add(table, 'adversarial_examples')

# Log the artifact
wandb.log_artifact(artifact)
wandb.finish()