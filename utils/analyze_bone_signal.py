import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

def analyze_bone_signals(base_path, output_path=None, show=False):
    # Rutas de los archivos (ajusta según tu estructura)
    ct_path = f"{base_path}/ct-label/ct.nii.gz"
    seg_path = f"{base_path}/ct-label/organ_seg.nii.gz"
    mri_path = f"{base_path}/features/mri_combined_in_phase.nii.gz"
    pet_path = f"{base_path}/features/nacpet.nii.gz"

    # Cargar los volúmenes
    print("Cargando imágenes...")
    ct_img = nib.load(ct_path).get_fdata()
    seg_img = nib.load(seg_path).get_fdata()
    mri_img = nib.load(mri_path).get_fdata()
    pet_img = nib.load(pet_path).get_fdata()

    # TotalSegmentator tiene IDs específicos para los huesos (ej. costillas, vértebras, fémur).
    # Si no estás seguro de los IDs, podemos usar un umbral en el CT real como aproximación:
    # bone_mask = ct_img > 300 
    
    # Asumiendo que has identificado los IDs de hueso en TotalSegmentator (ej. IDs del 1 al 24 suelen ser vértebras/costillas)
    # Aquí usaré un enfoque basado en umbral de CT (>300 HU) dentro de la máscara del cuerpo para asegurar que es hueso
    body_mask = nib.load(f"{base_path}/ct-label/body_seg.nii.gz").get_fdata() > 0
    bone_mask = (ct_img > 300) & body_mask

    # Extraer los valores de los píxeles dentro y fuera del hueso
    print("Extrayendo intensidades...")
    mri_bone_values = mri_img[bone_mask]
    mri_soft_values = mri_img[(~bone_mask) & body_mask] # Tejido blando (dentro del cuerpo, no hueso)

    pet_bone_values = pet_img[bone_mask]
    pet_soft_values = pet_img[(~bone_mask) & body_mask]

    # Visualización de los histogramas
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot MRI
    axs[0].hist(mri_soft_values, bins=50, alpha=0.5, density=True, label='Tejido Blando')
    axs[0].hist(mri_bone_values, bins=50, alpha=0.5, density=True, label='Hueso', color='red')
    axs[0].set_title('Distribución de Intensidad MRI In-Phase')
    axs[0].set_xlabel('Intensidad MRI')
    axs[0].legend()

    # Plot PET
    axs[1].hist(pet_soft_values, bins=50, alpha=0.5, density=True, label='Tejido Blando')
    axs[1].hist(pet_bone_values, bins=50, alpha=0.5, density=True, label='Hueso', color='red')
    axs[1].set_title('Distribución de Intensidad NAC-PET')
    axs[1].set_xlabel('Intensidad PET')
    axs[1].legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figura guardada en: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Imprimir estadísticas
    print(f"MRI Promedio en hueso: {np.mean(mri_bone_values):.2f} (Std: {np.std(mri_bone_values):.2f})")
    print(f"MRI Promedio en tejido blando: {np.mean(mri_soft_values):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze bone vs soft-tissue signals")
    parser.add_argument("--base_path", required=True, help="Ruta al sujeto, por ejemplo /data/pseudo_ct/train/sub-010")
    parser.add_argument("--output", default=None, help="Ruta opcional para guardar la figura")
    parser.add_argument("--show", action="store_true", help="Mostrar la figura en pantalla")

    args = parser.parse_args()
    analyze_bone_signals(args.base_path, output_path=args.output, show=args.show)