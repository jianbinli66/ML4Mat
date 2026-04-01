import os
import glob
import shutil


def rename_pdf_files(directory="."):
    """Rename all PDF files in directory to 001.pdf, 002.pdf, etc."""

    # Store original directory
    original_dir = os.getcwd()

    try:
        # Convert to absolute path
        target_dir = os.path.abspath(directory)

        # Check if directory exists
        if not os.path.exists(target_dir):
            print(f"Error: Directory '{target_dir}' does not exist")
            return

        # Change to target directory
        os.chdir(target_dir)

        # Print current working directory for debugging
        print(f"Working in: {os.getcwd()}")

        # Get all PDF files (case insensitive)
        pdf_files = sorted([
            f for f in os.listdir('.')
            if f.lower().endswith('.pdf') and os.path.isfile(f)
        ])

        if not pdf_files:
            print("No PDF files found in directory")
            return

        print(f"Found {len(pdf_files)} PDF files:")
        for f in pdf_files:
            print(f"  - {f}")

        # Check for files with long paths
        long_paths = [f for f in pdf_files if len(os.path.join(target_dir, f)) > 200]
        if long_paths:
            print(f"\nWarning: Found {len(long_paths)} files with long paths:")
            for f in long_paths:
                print(f"  - {f} (length: {len(os.path.join(target_dir, f))} characters)")
            print("These files may fail to rename on Windows due to path length limits.")
            response = input("\nContinue anyway? (y/n): ").lower()
            if response != 'y':
                print("Operation cancelled")
                return

        # Confirm with user
        response = input("\nRename these files? (y/n): ").lower()
        if response != 'y':
            print("Operation cancelled")
            return

        # Use temporary naming in a subdirectory to avoid path length issues
        temp_dir = "_temp_rename"
        os.makedirs(temp_dir, exist_ok=True)

        # Move to temporary directory first
        for i, old_name in enumerate(pdf_files, 1):
            temp_path = os.path.join(temp_dir, f"_tmp_{i:04d}.pdf")
            try:
                shutil.move(old_name, temp_path)
                print(f"  {old_name} -> {temp_path}")
            except Exception as e:
                print(f"  Error moving {old_name}: {e}")
                # Try to copy instead if move fails
                try:
                    shutil.copy2(old_name, temp_path)
                    os.remove(old_name)
                    print(f"  Copied {old_name} to {temp_path}")
                except Exception as copy_error:
                    print(f"  Failed to handle {old_name}: {copy_error}")

        # Rename from temporary to final names
        temp_files = sorted(glob.glob(os.path.join(temp_dir, "_tmp_*.pdf")))
        for i, temp_file in enumerate(temp_files, 1):
            final_name = f"{i:04d}.pdf"
            try:
                shutil.move(temp_file, final_name)
                print(f"  {temp_file} -> {final_name}")
            except Exception as e:
                print(f"  Error renaming {temp_file}: {e}")

        # Remove temporary directory if empty
        try:
            os.rmdir(temp_dir)
        except OSError:
            # Directory not empty, keep it
            pass

        print(f"\nSuccessfully renamed {len(pdf_files)} files")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Return to original directory
        try:
            os.chdir(original_dir)
        except:
            pass


# Run the function
if __name__ == "__main__":
    # Use current directory or specify a path
    rename_pdf_files("../data/DAC_2000_2025")