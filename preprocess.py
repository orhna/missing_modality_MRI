import os
import glob
import numpy as np
import nibabel as nib
from monai.transforms import CropForegroundd
from tqdm import tqdm
import argparse

def process_brats_dataset(input_base_dir, output_base_dir):
    """
    Processes the BraTS dataset by combining MRI modalities, applying foreground cropping,
    and saving the results to a new directory structure.
    """
    output_images_dir = os.path.join(output_base_dir, "Images")
    output_labels_dir = os.path.join(output_base_dir, "Labels")

    # Create output directories if they don't exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    print(f"Output images will be saved to: {output_images_dir}")
    print(f"Output labels will be saved to: {output_labels_dir}")

    # Get all patient directories (e.g., "BraTS-GLI-00000-000")
    patient_dirs = sorted([d for d in glob.glob(os.path.join(input_base_dir, "BraTS-*-*")) if os.path.isdir(d)])

    if not patient_dirs:
        print(f"Error: No patient directories found in {input_base_dir}. Please check the path.")
        return

    # Initialize the MONAI transform for cropping.
    cropper = CropForegroundd(keys=['image', 'label'], source_key='image', margin=0)

    for p_dir in tqdm(patient_dirs, desc="Processing Patients"):
        patient_id = os.path.basename(p_dir)
        nii_files = glob.glob(os.path.join(p_dir, "*.nii.gz"))
        label_path = None
        image_paths = []

        for f in nii_files:
            if "seg" in os.path.basename(f):
                label_path = f
            else:
                image_paths.append(f)

        image_paths.sort()

        if not label_path or not image_paths:
            print(f"Warning: Skipping {patient_id} due to missing image or label files.")
            continue

        try:
            # 1. Load and Concatenate Images
            label_nii = nib.load(label_path)
            label_data = label_nii.get_fdata().astype(np.uint8)
            image_data_list = [nib.load(p).get_fdata() for p in image_paths]
            combined_image_data = np.stack(image_data_list, axis=-1)

            # 2. Prepare for MONAI Transform
            combined_image_monai = np.transpose(combined_image_data, (3, 0, 1, 2))
            label_monai = np.expand_dims(label_data, axis=0)
            data_dict = {'image': combined_image_monai, 'label': label_monai}

            # 3. Apply Foreground Cropping
            cropped_dict = cropper(data_dict)
            cropped_image_monai = cropped_dict['image']
            cropped_label_monai = cropped_dict['label']

            # 4. Prepare for Saving
            cropped_image_to_save = np.transpose(cropped_image_monai, (1, 2, 3, 0))
            cropped_label_to_save = np.squeeze(cropped_label_monai, axis=0)

            # 5. Save the Processed Files
            # Use the original affine from the label (since MONAI's CropForegroundd does not update affine)
            new_affine = label_nii.affine

            cropped_image_nii = nib.Nifti1Image(cropped_image_to_save, new_affine)
            cropped_label_nii = nib.Nifti1Image(cropped_label_to_save, new_affine)

            output_image_path = os.path.join(output_images_dir, f"{patient_id}.nii.gz")
            output_label_path = os.path.join(output_labels_dir, f"{patient_id}.nii.gz")

            nib.save(cropped_image_nii, output_image_path)
            nib.save(cropped_label_nii, output_label_path)

        except Exception as e:
            print(f"Error processing {patient_id}: {e}")

    print("\nData processing complete!")

def main():
    parser = argparse.ArgumentParser(description="Process BraTS dataset with cropping and save to new directory.")
    parser.add_argument("input_base_dir", type=str, help="Path to the input BraTS dataset directory")
    parser.add_argument("output_base_dir", type=str, help="Path to the output directory for processed data")
    args = parser.parse_args()

    process_brats_dataset(args.input_base_dir, args.output_base_dir)

if __name__ == "__main__":
    main()