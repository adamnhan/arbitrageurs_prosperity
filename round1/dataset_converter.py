import pandas as pd

def convert_semicolon_to_comma(input_path: str, output_path: str):
    """
    Converts a semicolon-separated CSV to a comma-separated CSV.

    Args:
        input_path (str): Path to the input .csv file using semicolons.
        output_path (str): Path to save the output .csv file using commas.
    """
    try:
        # Load the semicolon-separated file
        df = pd.read_csv(input_path, sep=';')
        
        # Save it using commas
        df.to_csv(output_path, index=False)
        print(f"✅ Converted and saved to {output_path}")
    
    except Exception as e:
        print(f"❌ Failed to convert file: {e}")

convert_semicolon_to_comma("datasets/tutorial_data.csv", "datasets/tutorial_data_converted.csv")
