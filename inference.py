from rfdetr import RFDETRBase
from PIL import Image
import supervision as sv

# Load your trained model
model = RFDETRBase(pretrain_weights='./output/best_checkpoint.pth')

# Run inference
image = Image.open('test_image.jpg')
detections = model.predict(image, threshold=0.5)

# Visualize results
annotated_image = sv.BoxAnnotator().annotate(image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections)

# Save or display
annotated_image.save('result.jpg')