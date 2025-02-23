import os
import shutil

def rename_and_collect_images(verbose=True):
    """
    This function:
      - Uses the directory containing this script as the 'working' (input) directory,
        where the 'person' folders are located.
      - Creates (or reuses) 'RenamedImages' in the *same* directory.
      - Recursively finds .jpg files in the structure:
          [personFolder]/
              [No Signs or With Signs]/
                  [environmentFolder]/
                      image.jpg
      - Renames them to: [SignFolder]_[EnvFolder]_[Counter].jpg
        (omitting the original filename).
      - Copies them into the 'RenamedImages' folder.
    """

    # The directory where this script is located (the "running" directory).
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # We'll treat that directory as our input directory where the person folders exist.
    input_dir = script_dir

    # Output folder: "RenamedImages" in the *same* directory
    output_dir = os.path.join(script_dir, "RenamedImages")

    if verbose:
        print(f"[INFO] Script (input) directory: {input_dir}")
        print(f"[INFO] Output directory will be: {output_dir}")

    # Ensure the output folder exists
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print("[INFO] Verified or created the 'RenamedImages' folder.")

    # For summary
    total_processed = 0
    total_copied = 0

    # We'll gather all subdirectories in the script_dir, assuming these are "person" folders
    person_folders = []
    for entry in os.listdir(input_dir):
        full_path = os.path.join(input_dir, entry)
        # Skip "RenamedImages" itself to avoid re-traversing our output folder
        if os.path.isdir(full_path) and entry != "RenamedImages":
            person_folders.append(entry)

    if verbose:
        print(f"[INFO] Found {len(person_folders)} potential person folders in the working directory.")

    # Global counter for producing unique filenames
    file_counter = 1

    # Traverse each 'person' folder
    for person_folder in person_folders:
        person_path = os.path.join(input_dir, person_folder)
        if verbose:
            print(f"\n[INFO] Entering person folder: {person_folder}")

        # Next, look for "No Signs"/"With Signs" subfolders (or similarly named sign folders)
        sign_folders = [
            f for f in os.listdir(person_path)
            if os.path.isdir(os.path.join(person_path, f))
        ]
        if verbose:
            print(f"[INFO]   Found {len(sign_folders)} sign subfolders under '{person_folder}'.")

        for sign_folder in sign_folders:
            sign_path = os.path.join(person_path, sign_folder)

            # Environment folders, e.g. "time(6)_fog(0)_rain(0)"
            env_folders = [
                f for f in os.listdir(sign_path)
                if os.path.isdir(os.path.join(sign_path, f))
            ]
            if verbose:
                print(f"[INFO]   Checking sign folder '{sign_folder}' → {len(env_folders)} environment subfolders found.")

            for env_folder in env_folders:
                env_path = os.path.join(sign_path, env_folder)

                # .jpg files
                all_files = os.listdir(env_path)
                jpg_files = [f for f in all_files if f.lower().endswith('.jpg')]

                if verbose:
                    print(f"[INFO]     Environment folder '{env_folder}' → {len(jpg_files)} .jpg files found.")

                for file_name in jpg_files:
                    total_processed += 1
                    original_file_path = os.path.join(env_path, file_name)

                    # Build the new filename WITHOUT the original base name
                    sign_str = sign_folder.replace(' ', '_')
                    env_str = env_folder.replace(' ', '_')
                    ext = os.path.splitext(file_name)[1]  # keep the original extension (.jpg)

                    # Use the global file_counter to ensure uniqueness
                    new_file_name = f"{sign_str}_{env_str}_{file_counter}{ext}"
                    file_counter += 1

                    output_file_path = os.path.join(output_dir, new_file_name)

                    if verbose:
                        print(f"[ACTION] Copying: {original_file_path} → {output_file_path}")

                    shutil.copy2(original_file_path, output_file_path)
                    total_copied += 1

    if verbose:
        print("\n[SUMMARY]")
        print(f"  Total JPG files found & processed: {total_processed}")
        print(f"  Total files successfully copied:   {total_copied}")
        print("  All images have been collected and renamed into 'RenamedImages' in the same directory.")

def main():
    rename_and_collect_images(verbose=True)

if __name__ == "__main__":
    main()
