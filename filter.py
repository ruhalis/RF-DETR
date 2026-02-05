import json
import shutil
from pathlib import Path

def filter_coco_classes(input_json, output_json, output_image_dir, 
                        input_image_dir, target_classes):
    """
    Filter COCO dataset to include only specific classes
    
    Args:
        input_json: Path to original _annotations.coco.json
        output_json: Path to save filtered annotations
        output_image_dir: Directory to save filtered images
        input_image_dir: Directory with original images
        target_classes: List of class names to keep (e.g., ['person', 'car'])
    """
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Filter categories and deduplicate by name (same name -> single new id)
    new_categories = []
    new_cat_id = 0
    old_to_new_id = {}
    seen_names = set()

    for cat in data['categories']:
        if cat['name'] not in target_classes:
            continue
        if cat['name'] in seen_names:
            # Same name already added: map this old id to the existing new id
            for i, nc in enumerate(new_categories):
                if nc['name'] == cat['name']:
                    old_to_new_id[cat['id']] = i
                    break
            continue
        seen_names.add(cat['name'])
        new_cat = cat.copy()
        new_cat['id'] = new_cat_id
        new_categories.append(new_cat)
        old_to_new_id[cat['id']] = new_cat_id
        new_cat_id += 1
    
    # Filter annotations
    new_annotations = []
    valid_image_ids = set()
    ann_id = 0
    
    for ann in data['annotations']:
        if ann['category_id'] in old_to_new_id:
            new_ann = ann.copy()
            new_ann['id'] = ann_id
            new_ann['category_id'] = old_to_new_id[ann['category_id']]
            new_annotations.append(new_ann)
            valid_image_ids.add(ann['image_id'])
            ann_id += 1
    
    # Filter images
    new_images = []
    old_to_new_image_id = {}
    new_img_id = 0
    
    Path(output_image_dir).mkdir(parents=True, exist_ok=True)
    
    for img in data['images']:
        if img['id'] in valid_image_ids:
            new_img = img.copy()
            old_to_new_image_id[img['id']] = new_img_id
            new_img['id'] = new_img_id
            new_images.append(new_img)
            
            # Copy image file
            src = Path(input_image_dir) / img['file_name']
            dst = Path(output_image_dir) / img['file_name']
            if src.exists():
                shutil.copy2(src, dst)
            
            new_img_id += 1
    
    # Update image_id references in annotations
    for ann in new_annotations:
        ann['image_id'] = old_to_new_image_id[ann['image_id']]
    
    # Create new dataset
    filtered_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': new_categories,
        'images': new_images,
        'annotations': new_annotations
    }
    
    # Save filtered annotations
    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"Filtered dataset created:")
    print(f"  Classes: {len(new_categories)}")
    print(f"  Images: {len(new_images)}")
    print(f"  Annotations: {len(new_annotations)}")

# Example usage
if __name__ == "__main__":
    # Specify which classes you want
    target_classes = ['white', 'white-2']  # Change to your classes
    
    # Filter each split
    for split in ['train', 'valid', 'test']:
        filter_coco_classes(
            input_json=f'avtotime.coco/{split}/_annotations.coco.json',
            output_json=f'filtered_dataset/{split}/_annotations.coco.json',
            output_image_dir=f'filtered_dataset/{split}',
            input_image_dir=f'avtotime.coco/{split}',
            target_classes=target_classes
        )