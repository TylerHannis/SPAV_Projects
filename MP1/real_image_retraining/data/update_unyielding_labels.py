import os

# Define the directory containing label files
labels_dir = "datasets/Yield-Sign.v5i.yolov11/valid/labels"  # Change this to your actual path

# Loop through each label file
for filename in os.listdir(labels_dir):
    if filename.endswith(".txt"):  # Ensure we are only modifying label files
        file_path = os.path.join(labels_dir, filename)

        # Read the file content
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Modify class ID (replace 0 with 1)
        updated_lines = []
        for line in lines:
            parts = line.strip().split()  # Split by whitespace
            if parts and parts[0] == "0":  # Check if class ID is 0
                parts[0] = "1"  # Change it to 1
            updated_lines.append(" ".join(parts))  # Reconstruct line

        # Write updated content back to the file
        with open(file_path, "w") as file:
            file.write("\n".join(updated_lines) + "\n")  # Ensure newline at end

print("Label files updated: Class 0 → Class 1")
