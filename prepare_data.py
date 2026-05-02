import os
import shutil
import xml.etree.ElementTree as ET

def process_dataset(base_dir, output_dir):
    image_dir = os.path.join(base_dir, "images")
    annotation_dir = os.path.join(base_dir, "annotations")

    classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]

    # Create class folders
    for cls in classes:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    for file in os.listdir(annotation_dir):
        if file.endswith(".xml"):
            xml_path = os.path.join(annotation_dir, file)

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                obj = root.find("object")
                if obj is None:
                    continue

                label = obj.find("name").text.strip()
                label = label.replace(" ", "_")

                image_name = root.find("filename").text

                src = os.path.join(image_dir, image_name)
                dst = os.path.join(output_dir, label, image_name)

                if os.path.exists(src):
                    shutil.copy(src, dst)

            except Exception as e:
                print(f"Error processing {file}: {e}")

    print(f"{base_dir} done")


# Run for both datasets
process_dataset("train", "data/train")
process_dataset("validation", "data/validation")