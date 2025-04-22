from PIL import Image
import argparse



def flip_Image(image: Image) -> Image:
    """
    Flips a PIL Image along the y-axis (vertically).
    
    Args:
        image (Image): PIL Image
        
    Returns:
        Image: Flipped image
    """
    return image.transpose(Image.FLIP_TOP_BOTTOM)



if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Flip an image vertically.")
    parser.add_argument("input_image", type=str, help="Path to the input image.")
    parser.add_argument("output_image", type=str, help="Path to save the flipped image.")
    args = parser.parse_args()
    
    # Load the image
    input_image = Image.open(args.input_image)
    # Flip the image
    flipped_image = flip_Image(input_image)
    # Save the flipped image
    flipped_image.save(args.output_image)
    print(f"Flipped image saved to {args.output_image}")


    """
    Example usage:
        python scripts/flip_image.py smplx_uv_altas_colored.png smplx_uv_altas_colored_no_flip.png
    """