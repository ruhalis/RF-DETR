from rfdetr import RFDETRBase  # or RFDETRNano, RFDETRSmall, etc.

# Initialize model
model = RFDETRBase()  # Loads pretrained COCO weights automatically

# Train model
model.train(
    dataset_dir='filtered_dataset',  # Directory containing train/valid/test folders
    epochs=50,                            # Number of epochs
    batch_size=4,                         # Adjust based on your GPU
    grad_accum_steps=4,                   # Effective batch size = batch_size * grad_accum_steps
    lr=1e-4,                              # Learning rate
    output_dir='./output',                # Where to save checkpoints
    tensorboard=True,                     # Enable TensorBoard logging
    # Optional: W&B logging
    # wandb=True,
    # project='my-rfdetr-project',
    # run='experiment-1'
)